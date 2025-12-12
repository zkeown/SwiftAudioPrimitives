import Accelerate
import Foundation

/// Real-time streaming STFT processor.
///
/// `StreamingSTFT` maintains an internal ring buffer for processing audio
/// in a streaming fashion. Push audio samples as they arrive, and pop
/// STFT frames as they become available.
///
/// This is suitable for:
/// - Real-time audio processing
/// - Low-latency applications
/// - Processing audio from microphone or network streams
///
/// ## Example Usage
///
/// ```swift
/// let streaming = StreamingSTFT(config: STFTConfig(nFFT: 1024, hopLength: 256))
///
/// // Audio callback (e.g., from AVAudioEngine)
/// func audioCallback(buffer: [Float]) async {
///     await streaming.push(buffer)
///
///     // Process available frames
///     let frames = await streaming.popFrames()
///     for frame in frames {
///         // Process each STFT frame
///     }
/// }
///
/// // At end of stream, flush remaining samples
/// let remaining = await streaming.flush()
/// ```
public actor StreamingSTFT {
    /// Ring buffer for incoming samples.
    private var ringBuffer: [Float]

    /// Current write position in ring buffer.
    private var writePosition: Int = 0

    /// Number of samples currently in buffer.
    private var samplesInBuffer: Int = 0

    /// STFT configuration.
    public let config: STFTConfig

    /// Pre-computed window.
    private let window: [Float]

    /// FFT setup wrapper.
    private let fftSetupWrapper: FFTSetupWrapper

    /// Total samples received (for debugging/stats).
    private var totalSamplesReceived: Int = 0

    /// Total frames output.
    private var totalFramesOutput: Int = 0

    // MARK: - Initialization

    /// Create a streaming STFT processor.
    ///
    /// - Parameter config: STFT configuration.
    public init(config: STFTConfig = STFTConfig()) {
        self.config = config
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Ring buffer sized for at least 2 FFT windows worth of data
        let bufferSize = config.nFFT * 4
        self.ringBuffer = [Float](repeating: 0, count: bufferSize)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)
    }

    // MARK: - Public Interface

    /// Push new audio samples into the buffer.
    ///
    /// - Parameter samples: Audio samples to add.
    /// - Note: If the buffer would overflow, it automatically expands to accommodate all samples.
    public func push(_ samples: [Float]) {
        // Check if we need to expand the buffer
        let neededCapacity = samplesInBuffer + samples.count
        if neededCapacity > ringBuffer.count {
            // Expand buffer - at least double or enough for the new samples
            let newSize = max(ringBuffer.count * 2, neededCapacity + config.nFFT)
            var newBuffer = [Float](repeating: 0, count: newSize)

            // Copy existing samples to new buffer (oldest first)
            let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count
            for i in 0..<samplesInBuffer {
                let oldIdx = (readPosition + i) % ringBuffer.count
                newBuffer[i] = ringBuffer[oldIdx]
            }

            ringBuffer = newBuffer
            writePosition = samplesInBuffer
        }

        for sample in samples {
            ringBuffer[writePosition] = sample
            writePosition = (writePosition + 1) % ringBuffer.count
            samplesInBuffer += 1
        }
        totalSamplesReceived += samples.count
    }

    /// Pop all available STFT frames.
    ///
    /// Returns frames computed from samples received so far.
    /// Each call removes the corresponding samples from the buffer.
    ///
    /// - Returns: Array of complex spectrogram frames.
    public func popFrames() -> [ComplexMatrix] {
        var frames: [ComplexMatrix] = []

        while samplesInBuffer >= config.nFFT {
            if let frame = computeNextFrame() {
                frames.append(frame)
                totalFramesOutput += 1
            }

            // Advance read position by hop length
            samplesInBuffer -= config.hopLength
        }

        return frames
    }

    /// Number of complete frames available.
    public var availableFrameCount: Int {
        max(0, (samplesInBuffer - config.nFFT) / config.hopLength + 1)
    }

    /// Number of samples currently buffered.
    public var bufferedSamples: Int {
        samplesInBuffer
    }

    /// Minimum samples needed before a frame can be output.
    public var samplesNeededForFrame: Int {
        max(0, config.nFFT - samplesInBuffer)
    }

    /// Reset internal state.
    public func reset() {
        ringBuffer = [Float](repeating: 0, count: ringBuffer.count)
        writePosition = 0
        samplesInBuffer = 0
        totalSamplesReceived = 0
        totalFramesOutput = 0
    }

    /// Flush remaining samples, padding with zeros if necessary.
    ///
    /// Call this at the end of a stream to process any remaining samples.
    ///
    /// - Returns: Array of remaining STFT frames (may be partial/zero-padded).
    public func flush() -> [ComplexMatrix] {
        var frames: [ComplexMatrix] = []

        // Process any complete frames
        frames.append(contentsOf: popFrames())

        // If there are remaining samples, zero-pad and compute final frame
        if samplesInBuffer > 0 {
            // Extract remaining samples
            var remaining = extractSamples(count: samplesInBuffer)

            // Zero-pad to nFFT
            if remaining.count < config.nFFT {
                remaining.append(contentsOf: [Float](repeating: 0, count: config.nFFT - remaining.count))
            }

            // Compute final frame
            let frame = computeFFTFrame(remaining)
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

    /// Compute the next STFT frame from the buffer.
    private func computeNextFrame() -> ComplexMatrix? {
        guard samplesInBuffer >= config.nFFT else { return nil }

        // Extract frame worth of samples
        let frame = extractSamples(count: config.nFFT)
        return computeFFTFrame(frame)
    }

    /// Extract samples from ring buffer (oldest first).
    private func extractSamples(count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)

        // Calculate read position (oldest sample)
        let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count

        for i in 0..<count {
            let idx = (readPosition + i) % ringBuffer.count
            samples[i] = ringBuffer[idx]
        }

        return samples
    }

    /// Compute FFT of a single frame.
    private func computeFFTFrame(_ frame: [Float]) -> ComplexMatrix {
        let n = config.nFFT

        // Apply window
        var windowedFrame = [Float](repeating: 0, count: n)
        for i in 0..<min(frame.count, config.winLength) {
            windowedFrame[i] = frame[i] * window[i]
        }

        // Compute FFT
        let (fftReal, fftImag) = computeFFT(windowedFrame)

        // Return as ComplexMatrix with shape (nFreqs, 1) - single frame
        let real = fftReal.map { [$0] }
        let imag = fftImag.map { [$0] }

        return ComplexMatrix(real: real, imag: imag)
    }

    /// Compute FFT using vDSP.
    private func computeFFT(_ input: [Float]) -> (real: [Float], imag: [Float]) {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

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

        // Unpack to positive frequencies
        var fftReal = [Float](repeating: 0, count: n / 2 + 1)
        var fftImag = [Float](repeating: 0, count: n / 2 + 1)

        fftReal[0] = realInput[0]
        fftImag[0] = 0

        for i in 1..<(n / 2) {
            fftReal[i] = realInput[i]
            fftImag[i] = imagInput[i]
        }

        fftReal[n / 2] = imagInput[0]
        fftImag[n / 2] = 0

        return (fftReal, fftImag)
    }
}

// MARK: - Convenience Extensions

extension StreamingSTFT {
    /// Create a streaming STFT with Banquet-compatible settings.
    public static var forBanquet: StreamingSTFT {
        StreamingSTFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: false  // Streaming mode doesn't use centering
        ))
    }

    /// Process a complete signal through streaming interface.
    ///
    /// This is mainly for testing; for non-streaming use, prefer the regular STFT.
    ///
    /// - Parameter signal: Complete audio signal.
    /// - Returns: All STFT frames.
    public func processComplete(_ signal: [Float]) -> [ComplexMatrix] {
        reset()
        push(signal)
        return flush()
    }
}
