import SwiftRosaCore
import Accelerate
import Foundation

/// Real-time streaming mel spectrogram processor.
///
/// `StreamingMel` maintains an internal ring buffer and computes mel spectrogram
/// frames as audio arrives. Push audio samples as they arrive, and pop
/// mel spectrogram frames as they become available.
///
/// This is suitable for:
/// - Real-time audio visualization
/// - Streaming audio analysis
/// - Low-latency feature extraction
///
/// ## Example Usage
///
/// ```swift
/// let streaming = await StreamingMel(sampleRate: 16000, nMels: 80)
///
/// // Audio callback (e.g., from AVAudioEngine)
/// func audioCallback(buffer: [Float]) async {
///     await streaming.push(buffer)
///
///     // Process available frames
///     let frames = await streaming.popFrames()
///     for frame in frames {
///         // frame is shape (nMels,) - single mel frame
///     }
/// }
///
/// // At end of stream
/// let remaining = await streaming.flush()
/// ```
public actor StreamingMel {
    /// Ring buffer for incoming samples.
    private var ringBuffer: [Float]

    /// Current write position in ring buffer.
    private var writePosition: Int = 0

    /// Number of samples currently in buffer.
    private var samplesInBuffer: Int = 0

    /// FFT size.
    public let nFFT: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Number of mel bands.
    public let nMels: Int

    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Pre-computed window.
    private let window: [Float]

    /// FFT setup wrapper.
    private let fftSetupWrapper: FFTSetupWrapper

    /// Pre-computed mel filterbank.
    private let filters: [[Float]]

    /// Statistics.
    private var totalSamplesReceived: Int = 0
    private var totalFramesOutput: Int = 0

    // MARK: - Initialization

    /// Create a streaming mel spectrogram processor.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: 512).
    ///   - nMels: Number of mel bands (default: 128).
    ///   - fMin: Minimum frequency (default: 0).
    ///   - fMax: Maximum frequency (default: sampleRate/2).
    ///   - windowType: Window type (default: hann).
    public init(
        sampleRate: Float = 22050,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        nMels: Int = 128,
        fMin: Float = 0,
        fMax: Float? = nil,
        windowType: WindowType = .hann
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels

        self.window = Windows.generate(windowType, length: nFFT, periodic: true)

        // Ring buffer sized for at least 4 FFT windows
        let bufferSize = nFFT * 4
        self.ringBuffer = [Float](repeating: 0, count: bufferSize)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)

        // Pre-compute mel filterbank
        let melFilterbank = MelFilterbank(
            nMels: nMels,
            nFFT: nFFT,
            sampleRate: sampleRate,
            fMin: fMin,
            fMax: fMax ?? sampleRate / 2,
            htk: false,
            norm: .slaney
        )
        self.filters = melFilterbank.filters()
    }

    // MARK: - Public Interface

    /// Push new audio samples into the buffer.
    ///
    /// - Parameter samples: Audio samples to add.
    public func push(_ samples: [Float]) {
        for sample in samples {
            ringBuffer[writePosition] = sample
            writePosition = (writePosition + 1) % ringBuffer.count
            samplesInBuffer = min(samplesInBuffer + 1, ringBuffer.count)
        }
        totalSamplesReceived += samples.count
    }

    /// Pop all available mel spectrogram frames.
    ///
    /// Each frame is an array of `nMels` values.
    ///
    /// - Returns: Array of mel frames, each of shape (nMels,).
    public func popFrames() -> [[Float]] {
        var frames: [[Float]] = []

        while samplesInBuffer >= nFFT {
            let frame = computeNextFrame()
            frames.append(frame)
            totalFramesOutput += 1

            // Advance read position by hop length
            samplesInBuffer -= hopLength
        }

        return frames
    }

    /// Pop frames as a 2D matrix (nMels, nFrames).
    ///
    /// - Returns: Mel spectrogram matrix.
    public func popMatrix() -> [[Float]] {
        let frames = popFrames()
        guard !frames.isEmpty else { return [] }

        // Transpose from (nFrames, nMels) to (nMels, nFrames)
        var result = [[Float]](repeating: [Float](repeating: 0, count: frames.count), count: nMels)

        for (t, frame) in frames.enumerated() {
            for m in 0..<nMels {
                result[m][t] = frame[m]
            }
        }

        return result
    }

    /// Number of complete frames available.
    public var availableFrameCount: Int {
        max(0, (samplesInBuffer - nFFT) / hopLength + 1)
    }

    /// Number of samples currently buffered.
    public var bufferedSamples: Int {
        samplesInBuffer
    }

    /// Minimum samples needed before a frame can be output.
    public var samplesNeededForFrame: Int {
        max(0, nFFT - samplesInBuffer)
    }

    /// Reset internal state.
    public func reset() {
        ringBuffer = [Float](repeating: 0, count: ringBuffer.count)
        writePosition = 0
        samplesInBuffer = 0
        totalSamplesReceived = 0
        totalFramesOutput = 0
    }

    /// Flush remaining samples with zero-padding.
    ///
    /// - Returns: Array of remaining mel frames.
    public func flush() -> [[Float]] {
        var frames: [[Float]] = []

        // Process complete frames
        frames.append(contentsOf: popFrames())

        // Process remaining samples with zero-padding
        if samplesInBuffer > 0 {
            var remaining = extractSamples(count: samplesInBuffer)

            if remaining.count < nFFT {
                remaining.append(contentsOf: [Float](repeating: 0, count: nFFT - remaining.count))
            }

            let frame = computeMelFrame(remaining)
            frames.append(frame)
            samplesInBuffer = 0
        }

        return frames
    }

    // MARK: - Statistics

    /// Statistics about the streaming processor.
    public struct Stats: Sendable {
        public let totalSamplesReceived: Int
        public let totalFramesOutput: Int
        public let currentBufferLevel: Int
        public let bufferCapacity: Int
    }

    /// Get current statistics.
    public var stats: Stats {
        Stats(
            totalSamplesReceived: totalSamplesReceived,
            totalFramesOutput: totalFramesOutput,
            currentBufferLevel: samplesInBuffer,
            bufferCapacity: ringBuffer.count
        )
    }

    // MARK: - Private Implementation

    /// Compute the next mel frame from the buffer.
    private func computeNextFrame() -> [Float] {
        let frame = extractSamples(count: nFFT)
        return computeMelFrame(frame)
    }

    /// Extract samples from ring buffer (oldest first).
    private func extractSamples(count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)
        let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count

        for i in 0..<count {
            let idx = (readPosition + i) % ringBuffer.count
            samples[i] = ringBuffer[idx]
        }

        return samples
    }

    /// Compute mel spectrogram for a single frame.
    private func computeMelFrame(_ frame: [Float]) -> [Float] {
        // Apply window
        var windowedFrame = [Float](repeating: 0, count: nFFT)
        vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(nFFT))

        // Compute power spectrum
        let powerSpec = computePowerSpectrum(windowedFrame)

        // Apply mel filterbank
        return applyMelFilterbank(powerSpec)
    }

    /// Compute power spectrum using FFT.
    private func computePowerSpectrum(_ input: [Float]) -> [Float] {
        let n = nFFT
        let log2n = vDSP_Length(log2(Double(n)))
        let nFreqs = n / 2 + 1

        var realInput = [Float](repeating: 0, count: n / 2)
        var imagInput = [Float](repeating: 0, count: n / 2)

        realInput.withUnsafeMutableBufferPointer { realPtr in
            imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                input.withUnsafeBufferPointer { inputPtr in
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(n / 2)
                    )
                }

                vDSP_fft_zrip(fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                var scale: Float = 0.5
                vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, vDSP_Length(n / 2))
                vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, vDSP_Length(n / 2))
            }
        }

        // Compute power spectrum
        var powerSpec = [Float](repeating: 0, count: nFreqs)

        // DC component
        powerSpec[0] = realInput[0] * realInput[0]

        // Other frequencies
        for i in 1..<(n / 2) {
            powerSpec[i] = realInput[i] * realInput[i] + imagInput[i] * imagInput[i]
        }

        // Nyquist
        powerSpec[n / 2] = imagInput[0] * imagInput[0]

        return powerSpec
    }

    /// Apply mel filterbank to power spectrum.
    private func applyMelFilterbank(_ powerSpec: [Float]) -> [Float] {
        var melFrame = [Float](repeating: 0, count: nMels)

        for m in 0..<nMels {
            var sum: Float = 0
            for f in 0..<min(powerSpec.count, filters[m].count) {
                sum += filters[m][f] * powerSpec[f]
            }
            melFrame[m] = sum
        }

        return melFrame
    }
}

// MARK: - Presets

extension StreamingMel {
    /// Configuration for speech recognition (16kHz, 80 mels).
    public static func speechRecognition() async -> StreamingMel {
        StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80,
            fMin: 20,
            fMax: 8000,
            windowType: .hann
        )
    }

    /// Configuration for music analysis.
    public static func musicAnalysis() async -> StreamingMel {
        StreamingMel(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            fMin: 0,
            fMax: nil,
            windowType: .hann
        )
    }

    /// Configuration matching Whisper model input.
    public static func whisperCompatible() async -> StreamingMel {
        StreamingMel(
            sampleRate: 16000,
            nFFT: 400,
            hopLength: 160,
            nMels: 80,
            fMin: 0,
            fMax: 8000,
            windowType: .hann
        )
    }

    /// Process a complete signal (for testing).
    public func processComplete(_ signal: [Float]) -> [[Float]] {
        reset()
        push(signal)
        return flush()
    }
}
