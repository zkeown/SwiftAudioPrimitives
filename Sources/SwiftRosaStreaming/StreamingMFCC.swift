import SwiftRosaCore
import SwiftRosaEffects
import Accelerate
import Foundation

/// Real-time streaming MFCC processor.
///
/// `StreamingMFCC` maintains an internal ring buffer and computes MFCC
/// frames as audio arrives. Push audio samples as they arrive, and pop
/// MFCC frames as they become available.
///
/// The MFCC pipeline is:
/// 1. Compute power spectrum via FFT
/// 2. Apply mel filterbank
/// 3. Log compression
/// 4. DCT-II to get cepstral coefficients
/// 5. Optional liftering
///
/// ## Example Usage
///
/// ```swift
/// let streaming = await StreamingMFCC(sampleRate: 16000, nMFCC: 13)
///
/// // Audio callback
/// func audioCallback(buffer: [Float]) async {
///     await streaming.push(buffer)
///
///     // Process available frames
///     let frames = await streaming.popFrames()
///     for frame in frames {
///         // frame is shape (nMFCC,)
///     }
/// }
///
/// // Get frames with deltas
/// let (mfcc, delta, deltaDelta) = await streaming.popFramesWithDeltas()
/// ```
public actor StreamingMFCC {
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

    /// Number of MFCC coefficients.
    public let nMFCC: Int

    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Liftering coefficient.
    public let lifter: Int

    /// Pre-computed window.
    private let window: [Float]

    /// FFT setup wrapper.
    private let fftSetupWrapper: FFTSetupWrapper

    /// Pre-computed mel filterbank.
    private let filters: [[Float]]

    /// Pre-computed DCT matrix.
    private let dctMatrix: [[Float]]

    /// Pre-computed lifter weights.
    private let lifterWeights: [Float]?

    /// Buffer for delta computation.
    private var frameHistory: [[Float]] = []

    /// Delta context width.
    private let deltaWidth: Int

    /// Statistics.
    private var totalSamplesReceived: Int = 0
    private var totalFramesOutput: Int = 0

    // MARK: - Initialization

    /// Create a streaming MFCC processor.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 16000).
    ///   - nFFT: FFT size (default: 512).
    ///   - hopLength: Hop between frames (default: 160).
    ///   - nMels: Number of mel bands (default: 40).
    ///   - nMFCC: Number of MFCC coefficients (default: 13).
    ///   - fMin: Minimum frequency (default: 20).
    ///   - fMax: Maximum frequency (default: sampleRate/2).
    ///   - windowType: Window type (default: hann).
    ///   - lifter: Liftering coefficient (default: 22, 0 = disabled).
    ///   - deltaWidth: Context width for delta features (default: 9).
    public init(
        sampleRate: Float = 16000,
        nFFT: Int = 512,
        hopLength: Int = 160,
        nMels: Int = 40,
        nMFCC: Int = 13,
        fMin: Float = 20,
        fMax: Float? = nil,
        windowType: WindowType = .hann,
        lifter: Int = 22,
        deltaWidth: Int = 9
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.nMFCC = min(nMFCC, nMels)
        self.lifter = lifter
        self.deltaWidth = deltaWidth

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

        // Pre-compute DCT matrix
        self.dctMatrix = Self.computeDCTMatrix(nMels: nMels, nMFCC: self.nMFCC)

        // Pre-compute lifter weights
        if lifter > 0 {
            self.lifterWeights = Self.computeLifterWeights(nMFCC: self.nMFCC, lifter: lifter)
        } else {
            self.lifterWeights = nil
        }
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

    /// Pop all available MFCC frames.
    ///
    /// Each frame is an array of `nMFCC` coefficients.
    ///
    /// - Returns: Array of MFCC frames, each of shape (nMFCC,).
    public func popFrames() -> [[Float]] {
        var frames: [[Float]] = []

        while samplesInBuffer >= nFFT {
            let frame = computeNextFrame()
            frames.append(frame)

            // Store for delta computation
            frameHistory.append(frame)
            if frameHistory.count > deltaWidth * 2 + 1 {
                frameHistory.removeFirst()
            }

            totalFramesOutput += 1
            samplesInBuffer -= hopLength
        }

        return frames
    }

    /// Pop frames as a 2D matrix (nMFCC, nFrames).
    ///
    /// - Returns: MFCC matrix.
    public func popMatrix() -> [[Float]] {
        let frames = popFrames()
        guard !frames.isEmpty else { return [] }

        // Transpose from (nFrames, nMFCC) to (nMFCC, nFrames)
        var result = [[Float]](repeating: [Float](repeating: 0, count: frames.count), count: nMFCC)

        for (t, frame) in frames.enumerated() {
            for m in 0..<nMFCC {
                result[m][t] = frame[m]
            }
        }

        return result
    }

    /// Pop frames with delta and delta-delta features.
    ///
    /// - Returns: Tuple of (mfcc, delta, deltaDelta) matrices, each (nMFCC, nFrames).
    public func popFramesWithDeltas() -> ([[Float]], [[Float]], [[Float]]) {
        let mfcc = popMatrix()
        guard !mfcc.isEmpty else { return ([], [], []) }

        let delta = MFCC.delta(mfcc, width: deltaWidth)
        let deltaDelta = MFCC.delta(delta, width: deltaWidth)

        return (mfcc, delta, deltaDelta)
    }

    /// Pop frames with stacked deltas (nMFCC * 3, nFrames).
    ///
    /// - Returns: Stacked feature matrix.
    public func popFramesStacked() -> [[Float]] {
        let (mfcc, delta, deltaDelta) = popFramesWithDeltas()
        guard !mfcc.isEmpty else { return [] }

        var stacked = mfcc
        stacked.append(contentsOf: delta)
        stacked.append(contentsOf: deltaDelta)

        return stacked
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
        frameHistory = []
    }

    /// Flush remaining samples with zero-padding.
    ///
    /// - Returns: Array of remaining MFCC frames.
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

            let frame = computeMFCCFrame(remaining)
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

    /// Compute the next MFCC frame from the buffer.
    private func computeNextFrame() -> [Float] {
        let frame = extractSamples(count: nFFT)
        return computeMFCCFrame(frame)
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

    /// Compute MFCC for a single frame.
    private func computeMFCCFrame(_ frame: [Float]) -> [Float] {
        // 1. Apply window
        var windowedFrame = [Float](repeating: 0, count: nFFT)
        vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(nFFT))

        // 2. Compute power spectrum
        let powerSpec = computePowerSpectrum(windowedFrame)

        // 3. Apply mel filterbank
        let melFrame = applyMelFilterbank(powerSpec)

        // 4. Log compression
        let logMel = applyLogCompression(melFrame)

        // 5. Apply DCT
        var mfcc = applyDCT(logMel)

        // 6. Apply liftering if configured
        if let weights = lifterWeights {
            for i in 0..<nMFCC {
                mfcc[i] *= weights[i]
            }
        }

        return mfcc
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

        powerSpec[0] = realInput[0] * realInput[0]

        for i in 1..<(n / 2) {
            powerSpec[i] = realInput[i] * realInput[i] + imagInput[i] * imagInput[i]
        }

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

    /// Apply log compression.
    private func applyLogCompression(_ melFrame: [Float]) -> [Float] {
        let amin: Float = 1e-10
        return melFrame.map { log(max($0, amin)) }
    }

    /// Apply DCT-II.
    private func applyDCT(_ logMel: [Float]) -> [Float] {
        var mfcc = [Float](repeating: 0, count: nMFCC)

        for k in 0..<nMFCC {
            var sum: Float = 0
            for m in 0..<nMels {
                sum += dctMatrix[k][m] * logMel[m]
            }
            mfcc[k] = sum
        }

        return mfcc
    }

    /// Compute DCT-II matrix.
    private static func computeDCTMatrix(nMels: Int, nMFCC: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(nMFCC)

        let scale = sqrt(2.0 / Float(nMels))

        for k in 0..<nMFCC {
            var row = [Float](repeating: 0, count: nMels)

            for m in 0..<nMels {
                let angle = Float.pi * Float(k) * (Float(m) + 0.5) / Float(nMels)
                row[m] = scale * cos(angle)
            }

            // First coefficient gets different scaling
            if k == 0 {
                var firstScale = Float(1.0 / sqrt(2.0))
                vDSP_vsmul(row, 1, &firstScale, &row, 1, vDSP_Length(nMels))
            }

            matrix.append(row)
        }

        return matrix
    }

    /// Compute lifter weights.
    private static func computeLifterWeights(nMFCC: Int, lifter: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: nMFCC)

        for n in 0..<nMFCC {
            weights[n] = 1.0 + Float(lifter) / 2.0 * sin(Float.pi * Float(n) / Float(lifter))
        }

        return weights
    }
}

// MARK: - Presets

extension StreamingMFCC {
    /// Standard 13-coefficient MFCC for speech recognition.
    public static func speechRecognition() async -> StreamingMFCC {
        StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 40,
            nMFCC: 13,
            fMin: 20,
            fMax: 8000,
            windowType: .hann,
            lifter: 22,
            deltaWidth: 9
        )
    }

    /// MFCC configuration for music analysis.
    public static func musicAnalysis() async -> StreamingMFCC {
        StreamingMFCC(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 20,
            fMin: 0,
            fMax: nil,
            windowType: .hann,
            lifter: 0,
            deltaWidth: 9
        )
    }

    /// Kaldi-compatible configuration.
    public static func kaldiCompatible() async -> StreamingMFCC {
        StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 23,
            nMFCC: 13,
            fMin: 20,
            fMax: 7800,
            windowType: .hamming,
            lifter: 22,
            deltaWidth: 9
        )
    }

    /// Process a complete signal (for testing).
    public func processComplete(_ signal: [Float]) -> [[Float]] {
        reset()
        push(signal)
        return flush()
    }
}
