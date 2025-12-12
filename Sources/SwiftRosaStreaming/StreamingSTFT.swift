import SwiftRosaCore
import Accelerate
import Foundation

/// Reusable workspace for streaming FFT computation to minimize allocations.
final class StreamingFFTWorkspace: @unchecked Sendable {
    /// Split complex buffers for FFT input.
    var realInput: [Float]
    var imagInput: [Float]
    /// Output buffers for FFT result.
    var fftReal: [Float]
    var fftImag: [Float]
    /// Windowed frame buffer.
    var windowedFrame: [Float]
    /// Frame extraction buffer.
    var frameBuffer: [Float]

    /// Create a workspace for the given FFT size.
    init(nFFT: Int) {
        let halfN = nFFT / 2
        self.realInput = [Float](repeating: 0, count: halfN)
        self.imagInput = [Float](repeating: 0, count: halfN)
        self.fftReal = [Float](repeating: 0, count: halfN + 1)
        self.fftImag = [Float](repeating: 0, count: halfN + 1)
        self.windowedFrame = [Float](repeating: 0, count: nFFT)
        self.frameBuffer = [Float](repeating: 0, count: nFFT)
    }

    /// Reset FFT buffers to zero.
    func resetFFTBuffers() {
        _ = realInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
        _ = imagInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
    }
}

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

    /// Reusable workspace for FFT computation.
    private let workspace: StreamingFFTWorkspace

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

        // Create reusable workspace
        self.workspace = StreamingFFTWorkspace(nFFT: config.nFFT)
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

            // Copy existing samples to new buffer using block copies
            let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count
            if readPosition + samplesInBuffer <= ringBuffer.count {
                // Contiguous: single block copy
                _ = ringBuffer.withUnsafeBufferPointer { srcPtr in
                    newBuffer.withUnsafeMutableBufferPointer { destPtr in
                        memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, samplesInBuffer * MemoryLayout<Float>.size)
                    }
                }
            } else {
                // Wrap-around: two block copies
                let firstChunkSize = ringBuffer.count - readPosition
                let secondChunkSize = samplesInBuffer - firstChunkSize
                _ = ringBuffer.withUnsafeBufferPointer { srcPtr in
                    newBuffer.withUnsafeMutableBufferPointer { destPtr in
                        memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, firstChunkSize * MemoryLayout<Float>.size)
                        memcpy(destPtr.baseAddress! + firstChunkSize, srcPtr.baseAddress!, secondChunkSize * MemoryLayout<Float>.size)
                    }
                }
            }

            ringBuffer = newBuffer
            writePosition = samplesInBuffer
        }

        // Block copy new samples to ring buffer
        let sampleCount = samples.count
        let bufferSize = ringBuffer.count

        samples.withUnsafeBufferPointer { srcPtr in
            ringBuffer.withUnsafeMutableBufferPointer { destPtr in
                if writePosition + sampleCount <= bufferSize {
                    // Single contiguous copy
                    memcpy(destPtr.baseAddress! + writePosition, srcPtr.baseAddress!, sampleCount * MemoryLayout<Float>.size)
                } else {
                    // Wrap-around: two copies
                    let firstChunkSize = bufferSize - writePosition
                    let secondChunkSize = sampleCount - firstChunkSize
                    memcpy(destPtr.baseAddress! + writePosition, srcPtr.baseAddress!, firstChunkSize * MemoryLayout<Float>.size)
                    memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + firstChunkSize, secondChunkSize * MemoryLayout<Float>.size)
                }
            }
        }

        writePosition = (writePosition + sampleCount) % bufferSize
        samplesInBuffer += sampleCount
        totalSamplesReceived += sampleCount
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

    /// Compute the next STFT frame from the buffer using workspace buffers.
    private func computeNextFrame() -> ComplexMatrix? {
        guard samplesInBuffer >= config.nFFT else { return nil }

        // Extract frame directly into workspace buffer (zero allocation)
        extractSamplesIntoWorkspace()
        return computeFFTFrameWithWorkspace()
    }

    /// Extract samples from ring buffer (oldest first) using block copies.
    private func extractSamples(count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)

        // Calculate read position (oldest sample)
        let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count

        ringBuffer.withUnsafeBufferPointer { srcPtr in
            samples.withUnsafeMutableBufferPointer { destPtr in
                if readPosition + count <= ringBuffer.count {
                    // Contiguous: single block copy
                    memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, count * MemoryLayout<Float>.size)
                } else {
                    // Wrap-around: two block copies
                    let firstChunkSize = ringBuffer.count - readPosition
                    let secondChunkSize = count - firstChunkSize
                    memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, firstChunkSize * MemoryLayout<Float>.size)
                    memcpy(destPtr.baseAddress! + firstChunkSize, srcPtr.baseAddress!, secondChunkSize * MemoryLayout<Float>.size)
                }
            }
        }

        return samples
    }

    /// Extract samples from ring buffer directly into workspace buffer.
    private func extractSamplesIntoWorkspace() {
        let count = config.nFFT

        // Calculate read position (oldest sample)
        let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count

        ringBuffer.withUnsafeBufferPointer { srcPtr in
            workspace.frameBuffer.withUnsafeMutableBufferPointer { destPtr in
                if readPosition + count <= ringBuffer.count {
                    // Contiguous: single block copy
                    memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, count * MemoryLayout<Float>.size)
                } else {
                    // Wrap-around: two block copies
                    let firstChunkSize = ringBuffer.count - readPosition
                    let secondChunkSize = count - firstChunkSize
                    memcpy(destPtr.baseAddress!, srcPtr.baseAddress! + readPosition, firstChunkSize * MemoryLayout<Float>.size)
                    memcpy(destPtr.baseAddress! + firstChunkSize, srcPtr.baseAddress!, secondChunkSize * MemoryLayout<Float>.size)
                }
            }
        }
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

    /// Compute FFT using workspace buffers (zero allocation hot path).
    private func computeFFTFrameWithWorkspace() -> ComplexMatrix {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

        // Zero out workspace windowed frame buffer
        _ = workspace.windowedFrame.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }

        // Apply window using vDSP (vectorized multiply)
        workspace.frameBuffer.withUnsafeBufferPointer { framePtr in
            window.withUnsafeBufferPointer { windowPtr in
                workspace.windowedFrame.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_vmul(framePtr.baseAddress!, 1, windowPtr.baseAddress!, 1, outPtr.baseAddress!, 1, vDSP_Length(config.winLength))
                }
            }
        }

        // Reset FFT buffers
        workspace.resetFFTBuffers()

        // Compute FFT using workspace buffers
        workspace.realInput.withUnsafeMutableBufferPointer { realPtr in
            workspace.imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                workspace.windowedFrame.withUnsafeBufferPointer { inputPtr in
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

        // Unpack to positive frequencies into workspace output buffers
        workspace.fftReal[0] = workspace.realInput[0]
        workspace.fftImag[0] = 0

        for i in 1..<(n / 2) {
            workspace.fftReal[i] = workspace.realInput[i]
            workspace.fftImag[i] = workspace.imagInput[i]
        }

        workspace.fftReal[n / 2] = workspace.imagInput[0]
        workspace.fftImag[n / 2] = 0

        // Return as ComplexMatrix with shape (nFreqs, 1) - single frame
        // Note: We still create the ComplexMatrix structure here, but the FFT computation
        // itself used no allocations
        let real = workspace.fftReal.map { [$0] }
        let imag = workspace.fftImag.map { [$0] }

        return ComplexMatrix(real: real, imag: imag)
    }

    /// Compute FFT using vDSP (legacy method for flush()).
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
