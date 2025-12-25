import SwiftRosaCore
import Accelerate
import Foundation

/// Real-time streaming ISTFT processor.
///
/// `StreamingISTFT` performs overlap-add reconstruction in a streaming fashion.
/// Push STFT frames as they arrive, and receive reconstructed audio samples
/// as they become available.
///
/// This is the inverse counterpart to `StreamingSTFT`, suitable for:
/// - Real-time audio synthesis
/// - Low-latency audio processing pipelines
/// - Streaming source separation
///
/// ## Example Usage
///
/// ```swift
/// let streamingISTFT = StreamingISTFT(config: stftConfig)
///
/// // Process STFT frames from streaming source
/// for frame in stftFrames {
///     let audioChunk = await streamingISTFT.push([frame])
///     // Output audioChunk to speakers/file
/// }
///
/// // Flush remaining audio at end of stream
/// let remaining = await streamingISTFT.flush()
/// ```
public actor StreamingISTFT {
    /// Default maximum buffer size (10 MB of Float samples = ~2.5M samples).
    public static let defaultMaxBufferSize = 10 * 1024 * 1024 / MemoryLayout<Float>.size

    /// Maximum allowed buffer size in samples.
    public let maxBufferSize: Int

    /// Overlap-add buffer for accumulating reconstructed samples.
    private var overlapBuffer: [Float]

    /// Window sum buffer for normalization.
    private var windowSumBuffer: [Float]

    /// Current read position (samples ready for output).
    private var readPosition: Int = 0

    /// Current write position (next overlap-add position).
    private var writePosition: Int = 0

    /// Frames processed so far.
    private var framesProcessed: Int = 0

    /// STFT configuration.
    public let config: STFTConfig

    /// Pre-computed window.
    private let window: [Float]

    /// Squared window for normalization.
    private let windowSquared: [Float]

    /// FFT setup wrapper.
    private let fftSetupWrapper: FFTSetupWrapper

    /// Minimum normalization threshold.
    private let epsilon: Float = 1e-8

    // MARK: - Initialization

    /// Create a streaming ISTFT processor.
    ///
    /// - Parameters:
    ///   - config: STFT configuration (should match the STFT used).
    ///   - maxBufferSize: Maximum buffer size in samples. Defaults to ~2.5M samples (10 MB).
    public init(config: STFTConfig = .default, maxBufferSize: Int = StreamingISTFT.defaultMaxBufferSize) {
        self.config = config
        self.maxBufferSize = maxBufferSize
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Pre-compute squared window
        var squared = [Float](repeating: 0, count: config.winLength)
        vDSP_vsq(window, 1, &squared, 1, vDSP_Length(config.winLength))
        self.windowSquared = squared

        // Buffer sized for several frames of overlap
        let bufferSize = config.nFFT * 8
        self.overlapBuffer = [Float](repeating: 0, count: bufferSize)
        self.windowSumBuffer = [Float](repeating: 0, count: bufferSize)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)
    }

    // MARK: - Public Interface

    /// Push STFT frames and get reconstructed audio.
    ///
    /// - Parameter frames: Array of STFT frames (each as ComplexMatrix with shape (nFreqs, 1)).
    /// - Returns: Reconstructed audio samples that are ready for output.
    /// - Throws: `StreamingError.bufferOverflow` if the buffer would exceed `maxBufferSize`.
    public func push(_ frames: [ComplexMatrix]) throws -> [Float] {
        // Check if processing these frames would exceed our buffer limit
        let projectedWritePosition = (framesProcessed + frames.count) * config.hopLength + config.winLength
        let requiredBufferSize = projectedWritePosition - readPosition
        if requiredBufferSize > maxBufferSize {
            throw StreamingError.bufferOverflow(requested: requiredBufferSize, max: maxBufferSize)
        }

        for frame in frames {
            processFrame(frame)
        }

        return collectReadySamples()
    }

    /// Number of audio samples available for output.
    public var availableSamples: Int {
        let samplesInBuffer = writePosition - readPosition
        // Only output samples that have been fully accumulated
        return max(0, samplesInBuffer - config.nFFT)
    }

    /// Reset internal state.
    public func reset() {
        overlapBuffer = [Float](repeating: 0, count: overlapBuffer.count)
        windowSumBuffer = [Float](repeating: 0, count: windowSumBuffer.count)
        readPosition = 0
        writePosition = 0
        framesProcessed = 0
    }

    /// Flush all remaining audio from the buffer.
    ///
    /// Call this at the end of a stream to get any buffered samples.
    ///
    /// - Returns: All remaining audio samples.
    public func flush() -> [Float] {
        var output: [Float] = []

        // Collect all samples up to write position
        while readPosition < writePosition {
            let idx = readPosition % overlapBuffer.count
            var sample = overlapBuffer[idx]

            // Normalize
            let windowSum = windowSumBuffer[idx]
            if windowSum > epsilon {
                sample /= windowSum
            }

            output.append(sample)

            // Clear buffer position for potential reuse
            overlapBuffer[idx] = 0
            windowSumBuffer[idx] = 0

            readPosition += 1
        }

        return output
    }

    // MARK: - Statistics

    /// Statistics about the streaming processor.
    public struct Stats: Sendable {
        public let framesProcessed: Int
        public let samplesOutput: Int
        public let currentLatency: Int  // Samples of latency
    }

    /// Get current statistics.
    public var stats: Stats {
        Stats(
            framesProcessed: framesProcessed,
            samplesOutput: readPosition,
            currentLatency: writePosition - readPosition
        )
    }

    // MARK: - Private Implementation

    /// Process a single STFT frame.
    private func processFrame(_ frame: ComplexMatrix) {
        // Extract real and imaginary parts (frame should be nFreqs x 1)
        var frameReal = [Float](repeating: 0, count: config.nFreqs)
        var frameImag = [Float](repeating: 0, count: config.nFreqs)

        for i in 0..<min(frame.rows, config.nFreqs) {
            frameReal[i] = frame.real[i][0]
            frameImag[i] = frame.imag[i][0]
        }

        // Compute inverse FFT
        let reconstructedFrame = computeIFFT(real: frameReal, imag: frameImag)

        // Apply synthesis window
        var windowedFrame = [Float](repeating: 0, count: config.winLength)
        for i in 0..<config.winLength {
            windowedFrame[i] = reconstructedFrame[i] * window[i]
        }

        // Overlap-add into buffer
        let frameStart = framesProcessed * config.hopLength

        for i in 0..<config.winLength {
            let bufferIdx = (frameStart + i) % overlapBuffer.count

            // Handle buffer wraparound
            if frameStart + i >= writePosition + overlapBuffer.count {
                // Buffer overflow - would need to expand or output more
                continue
            }

            overlapBuffer[bufferIdx] += windowedFrame[i]
            windowSumBuffer[bufferIdx] += windowSquared[i]
        }

        // Update write position
        writePosition = max(writePosition, frameStart + config.winLength)
        framesProcessed += 1
    }

    /// Collect samples that are ready for output.
    private func collectReadySamples() -> [Float] {
        var output: [Float] = []

        // Output samples that have been fully accumulated (at least nFFT samples behind write position)
        let readyUpTo = writePosition - config.nFFT

        while readPosition < readyUpTo {
            let idx = readPosition % overlapBuffer.count
            var sample = overlapBuffer[idx]

            // Normalize
            let windowSum = windowSumBuffer[idx]
            if windowSum > epsilon {
                sample /= windowSum
            }

            output.append(sample)

            // Clear buffer position
            overlapBuffer[idx] = 0
            windowSumBuffer[idx] = 0

            readPosition += 1
        }

        return output
    }

    /// Compute inverse FFT.
    private func computeIFFT(real: [Float], imag: [Float]) -> [Float] {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

        // Pack into format expected by vDSP
        var fullReal = [Float](repeating: 0, count: n / 2)
        var fullImag = [Float](repeating: 0, count: n / 2)

        fullReal[0] = real[0]
        fullImag[0] = real[n / 2]  // Nyquist in imag[0]

        for i in 1..<(n / 2) {
            fullReal[i] = real[i]
            fullImag[i] = imag[i]
        }

        // Perform inverse FFT
        fullReal.withUnsafeMutableBufferPointer { realPtr in
            fullImag.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_fft_zrip(fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))
            }
        }

        // Unpack to real output
        var output = [Float](repeating: 0, count: n)

        fullReal.withUnsafeBufferPointer { realPtr in
            fullImag.withUnsafeBufferPointer { imagPtr in
                var split = DSPSplitComplex(
                    realp: UnsafeMutablePointer(mutating: realPtr.baseAddress!),
                    imagp: UnsafeMutablePointer(mutating: imagPtr.baseAddress!)
                )

                output.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_ztoc(
                        &split,
                        1,
                        UnsafeMutablePointer<DSPComplex>(OpaquePointer(outPtr.baseAddress!)),
                        2,
                        vDSP_Length(n / 2)
                    )
                }
            }
        }

        // Scale
        var scale = 1.0 / Float(n)
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(n))

        return output
    }
}

// MARK: - Convenience Extensions

extension StreamingISTFT {
    /// Create a streaming ISTFT with Banquet-compatible settings.
    public static var forBanquet: StreamingISTFT {
        StreamingISTFT(config: STFTConfig(
            uncheckedNFFT: 2048,
            hopLength: 512,
            winLength: 2048,
            windowType: .hann,
            center: false
        ))
    }

    /// Get the inherent latency in samples.
    ///
    /// This is the minimum delay between input and output due to buffering.
    public var latencySamples: Int {
        config.nFFT
    }

    /// Get the inherent latency in seconds.
    ///
    /// - Parameter sampleRate: Audio sample rate.
    /// - Returns: Latency in seconds.
    public func latencySeconds(sampleRate: Float) -> Float {
        Float(latencySamples) / sampleRate
    }
}

// MARK: - Combined Streaming Processor

/// Combined streaming STFT/ISTFT processor for real-time audio effects.
///
/// Useful for processing audio in the frequency domain with minimal latency.
public actor StreamingProcessor {
    private let streamingSTFT: StreamingSTFT
    private let streamingISTFT: StreamingISTFT

    /// Create a streaming audio processor.
    ///
    /// - Parameter config: STFT configuration.
    public init(config: STFTConfig = .default) {
        // Use non-centered config for streaming
        let streamConfig = STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.winLength,
            windowType: config.windowType,
            center: false,
            padMode: config.padMode
        )
        self.streamingSTFT = StreamingSTFT(config: streamConfig)
        self.streamingISTFT = StreamingISTFT(config: streamConfig)
    }

    /// Process audio samples through a spectral transformation.
    ///
    /// - Parameters:
    ///   - samples: Input audio samples.
    ///   - transform: Function to transform STFT frames.
    /// - Returns: Processed audio samples.
    /// - Throws: `StreamingError.bufferOverflow` if either buffer would exceed its limit.
    public func process(
        _ samples: [Float],
        transform: @Sendable ([ComplexMatrix]) -> [ComplexMatrix]
    ) async throws -> [Float] {
        // Push to STFT
        try await streamingSTFT.push(samples)

        // Get available frames
        let frames = await streamingSTFT.popFrames()

        // Apply transformation
        let transformedFrames = transform(frames)

        // Push to ISTFT and get output
        return try await streamingISTFT.push(transformedFrames)
    }

    /// Flush remaining audio at end of stream.
    ///
    /// - Throws: `StreamingError.bufferOverflow` if the ISTFT buffer would exceed its limit.
    public func flush() async throws -> [Float] {
        let remainingFrames = await streamingSTFT.flush()
        var output = try await streamingISTFT.push(remainingFrames)
        output.append(contentsOf: await streamingISTFT.flush())
        return output
    }

    /// Reset both processors.
    public func reset() async {
        await streamingSTFT.reset()
        await streamingISTFT.reset()
    }
}
