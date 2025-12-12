import Accelerate
import Foundation

/// Padding mode for signal centering.
public enum PadMode: Sendable {
    /// Reflect signal at boundaries.
    case reflect
    /// Pad with constant value (default, matches librosa's default).
    case constant(Float)
    /// Pad with edge values.
    case edge
}

/// Configuration for STFT computation.
public struct STFTConfig: Sendable {
    /// FFT size. Default: 2048.
    public let nFFT: Int

    /// Hop length between frames. Default: nFFT/4.
    public let hopLength: Int

    /// Window length. Default: nFFT.
    public let winLength: Int

    /// Window function type. Default: hann.
    public let windowType: WindowType

    /// Whether to center the signal by padding. Default: true.
    public let center: Bool

    /// Padding mode when center is true. Default: reflect.
    public let padMode: PadMode

    /// Create STFT configuration.
    ///
    /// - Parameters:
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: nFFT/4).
    ///   - winLength: Window length (default: nFFT).
    ///   - windowType: Window function (default: hann).
    ///   - center: Pad signal for centered frames (default: true).
    ///   - padMode: Padding mode (default: constant(0), matching librosa).
    public init(
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        windowType: WindowType = .hann,
        center: Bool = true,
        padMode: PadMode = .constant(0)
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength ?? (nFFT / 4)
        self.winLength = winLength ?? nFFT
        self.windowType = windowType
        self.center = center
        self.padMode = padMode
    }

    /// Number of frequency bins in output (nFFT/2 + 1 for real input).
    public var nFreqs: Int { nFFT / 2 + 1 }
}

/// Wrapper for FFTSetup to make it Sendable
/// Note: FFTSetup is thread-safe for read operations after creation
public final class FFTSetupWrapper: @unchecked Sendable {
    public let setup: FFTSetup

    public init(log2n: vDSP_Length) {
        self.setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }
}

/// Reusable workspace for STFT computation to avoid per-frame allocations.
///
/// This class holds pre-allocated buffers that are reused across frames,
/// significantly reducing memory allocations in hot paths.
public final class STFTWorkspace: @unchecked Sendable {
    /// Split complex buffers for FFT input.
    public var realInput: [Float]
    public var imagInput: [Float]
    /// Output buffers for FFT result.
    public var fftReal: [Float]
    public var fftImag: [Float]
    /// Windowed frame buffer.
    public var windowedFrame: [Float]

    /// Create a workspace for the given FFT size.
    public init(nFFT: Int) {
        let halfN = nFFT / 2
        self.realInput = [Float](repeating: 0, count: halfN)
        self.imagInput = [Float](repeating: 0, count: halfN)
        self.fftReal = [Float](repeating: 0, count: halfN + 1)
        self.fftImag = [Float](repeating: 0, count: halfN + 1)
        self.windowedFrame = [Float](repeating: 0, count: nFFT)
    }

    /// Reset all buffers to zero.
    public func reset() {
        realInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
        imagInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
    }
}

/// Short-Time Fourier Transform with automatic GPU/CPU dispatch.
///
/// Uses Metal GPU acceleration for large signals and optimized
/// vDSP operations for CPU fallback.
public struct STFT: Sendable {
    public let config: STFTConfig

    private let window: [Float]
    private let fftSetupWrapper: FFTSetupWrapper

    /// Optional Metal engine for GPU acceleration.
    private let metalEngine: MetalEngine?

    /// Whether to prefer GPU acceleration when available.
    public let useGPU: Bool

    /// Create STFT processor.
    ///
    /// - Parameters:
    ///   - config: STFT configuration.
    ///   - useGPU: Prefer GPU acceleration when available (default: true).
    public init(config: STFTConfig = STFTConfig(), useGPU: Bool = true) {
        self.config = config
        self.useGPU = useGPU
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)

        // Initialize Metal engine if GPU is preferred and available
        if useGPU && MetalEngine.isAvailable {
            self.metalEngine = try? MetalEngine()
        } else {
            self.metalEngine = nil
        }
    }

    /// Compute STFT of a signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram of shape (nFreqs, nFrames).
    public func transform(_ signal: [Float]) async -> ComplexMatrix {
        let paddedSignal = config.center ? padSignal(signal) : signal

        // Get frame start indices instead of copying frames
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else {
            return ComplexMatrix(real: [], imag: [])
        }

        let nFFT = config.nFFT
        let nFreqs = config.nFreqs
        let winLength = config.winLength
        let halfN = nFFT / 2
        let log2n = vDSP_Length(log2(Double(nFFT)))

        // OPTIMIZED: Use frame-major output layout for cache-friendly writes
        // Then transpose at the end (single pass)
        var realFrameMajor = [Float](repeating: 0, count: nFrames * nFreqs)
        var imagFrameMajor = [Float](repeating: 0, count: nFrames * nFreqs)

        // Process frames in parallel using GCD for better performance
        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        let framesPerThread = (nFrames + processorCount - 1) / processorCount

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            window.withUnsafeBufferPointer { windowPtr in
                realFrameMajor.withUnsafeMutableBufferPointer { outRealPtr in
                    imagFrameMajor.withUnsafeMutableBufferPointer { outImagPtr in
                        DispatchQueue.concurrentPerform(iterations: processorCount) { threadIdx in
                            let startFrame = threadIdx * framesPerThread
                            let endFrame = min(startFrame + framesPerThread, nFrames)

                            if startFrame >= endFrame { return }

                            // Thread-local workspace
                            var windowedFrame = [Float](repeating: 0, count: nFFT)
                            var splitReal = [Float](repeating: 0, count: halfN)
                            var splitImag = [Float](repeating: 0, count: halfN)
                            var scale: Float = 0.5

                            windowedFrame.withUnsafeMutableBufferPointer { framePtr in
                                splitReal.withUnsafeMutableBufferPointer { realPtr in
                                    splitImag.withUnsafeMutableBufferPointer { imagPtr in
                                        var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                                        for frameIdx in startFrame..<endFrame {
                                            let startIdx = indices[frameIdx]

                                            // Zero the frame buffer
                                            vDSP_vclr(framePtr.baseAddress!, 1, vDSP_Length(nFFT))

                                            // Window the frame
                                            let availableSamples = min(startIdx + winLength, paddedSignal.count) - startIdx
                                            vDSP_vmul(
                                                signalPtr.baseAddress! + startIdx, 1,
                                                windowPtr.baseAddress!, 1,
                                                framePtr.baseAddress!, 1,
                                                vDSP_Length(min(availableSamples, winLength))
                                            )

                                            // Pack into split complex
                                            vDSP_ctoz(
                                                UnsafePointer<DSPComplex>(OpaquePointer(framePtr.baseAddress!)),
                                                2,
                                                &split,
                                                1,
                                                vDSP_Length(halfN)
                                            )

                                            // FFT
                                            vDSP_fft_zrip(self.fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                                            vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, vDSP_Length(halfN))
                                            vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, vDSP_Length(halfN))

                                            // OPTIMIZED: Write to frame-major layout (contiguous per frame)
                                            let frameOffset = frameIdx * nFreqs

                                            // DC component
                                            outRealPtr[frameOffset] = split.realp[0]
                                            outImagPtr[frameOffset] = 0

                                            // Positive frequencies (contiguous copy)
                                            memcpy(outRealPtr.baseAddress! + frameOffset + 1,
                                                   realPtr.baseAddress! + 1,
                                                   (halfN - 1) * MemoryLayout<Float>.size)
                                            memcpy(outImagPtr.baseAddress! + frameOffset + 1,
                                                   imagPtr.baseAddress! + 1,
                                                   (halfN - 1) * MemoryLayout<Float>.size)

                                            // Nyquist
                                            outRealPtr[frameOffset + halfN] = split.imagp[0]
                                            outImagPtr[frameOffset + halfN] = 0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Transpose from frame-major (nFrames × nFreqs) to freq-major (nFreqs × nFrames)
        // Use vDSP_mtrans for efficient transpose, then slice
        var transposedReal = [Float](repeating: 0, count: nFreqs * nFrames)
        var transposedImag = [Float](repeating: 0, count: nFreqs * nFrames)

        vDSP_mtrans(realFrameMajor, 1, &transposedReal, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))
        vDSP_mtrans(imagFrameMajor, 1, &transposedImag, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))

        // Build result arrays using slicing (single allocation per row via Array init)
        var realResult = [[Float]]()
        var imagResult = [[Float]]()
        realResult.reserveCapacity(nFreqs)
        imagResult.reserveCapacity(nFreqs)

        transposedReal.withUnsafeBufferPointer { realPtr in
            transposedImag.withUnsafeBufferPointer { imagPtr in
                for f in 0..<nFreqs {
                    let start = f * nFrames
                    realResult.append(Array(UnsafeBufferPointer(start: realPtr.baseAddress! + start, count: nFrames)))
                    imagResult.append(Array(UnsafeBufferPointer(start: imagPtr.baseAddress! + start, count: nFrames)))
                }
            }
        }

        return ComplexMatrix(real: realResult, imag: imagResult)
    }

    /// Compute magnitude spectrogram.
    ///
    /// Uses GPU acceleration for large signals when available.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nFreqs, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let paddedSignal = config.center ? padSignal(signal) : signal
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else { return [] }

        let nFFT = config.nFFT
        let nFreqs = config.nFreqs
        let elementCount = nFFT * nFrames

        // Try GPU path for large data
        if let engine = metalEngine,
           MetalEngine.shouldUseGPU(elementCount: elementCount, operationType: .fft)
        {
            // Prepare windowed frames for GPU
            var flatFrames = [Float](repeating: 0, count: nFFT * nFrames)

            paddedSignal.withUnsafeBufferPointer { signalPtr in
                window.withUnsafeBufferPointer { windowPtr in
                    flatFrames.withUnsafeMutableBufferPointer { framesPtr in
                        for (frameIdx, startIdx) in indices.enumerated() {
                            let frameStart = frameIdx * nFFT
                            let availableSamples = min(startIdx + config.winLength, paddedSignal.count) - startIdx

                            // Window the frame directly into flat array
                            vDSP_vmul(
                                signalPtr.baseAddress! + startIdx, 1,
                                windowPtr.baseAddress!, 1,
                                framesPtr.baseAddress! + frameStart, 1,
                                vDSP_Length(min(availableSamples, config.winLength))
                            )
                        }
                    }
                }
            }

            // Use GPU STFT magnitude
            do {
                let contiguousMag = try await engine.stftMagnitude(
                    frames: flatFrames,
                    nFFT: nFFT,
                    nFrames: nFrames
                )

                // Convert to [[Float]] format
                var result = [[Float]]()
                result.reserveCapacity(nFreqs)
                for f in 0..<nFreqs {
                    let start = f * nFrames
                    result.append(Array(contiguousMag.data[start..<(start + nFrames)]))
                }
                return result
            } catch {
                // Fall through to CPU on GPU error
            }
        }

        // CPU fallback
        let spectrogram = await transform(signal)
        return spectrogram.magnitude
    }

    /// Compute power spectrogram (magnitude squared).
    ///
    /// Uses GPU acceleration for large signals when available.
    /// Optimized to compute power directly without intermediate sqrt.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram of shape (nFreqs, nFrames).
    public func power(_ signal: [Float]) async -> [[Float]] {
        let paddedSignal = config.center ? padSignal(signal) : signal
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else { return [] }

        let nFFT = config.nFFT
        let nFreqs = config.nFreqs
        let winLength = config.winLength
        let halfN = nFFT / 2
        let log2n = vDSP_Length(log2(Double(nFFT)))

        // OPTIMIZED: Compute power directly (real² + imag²) without sqrt
        // Use frame-major output layout for cache-friendly writes
        var powerFrameMajor = [Float](repeating: 0, count: nFrames * nFreqs)

        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        let framesPerThread = (nFrames + processorCount - 1) / processorCount

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            window.withUnsafeBufferPointer { windowPtr in
                powerFrameMajor.withUnsafeMutableBufferPointer { outPtr in
                    DispatchQueue.concurrentPerform(iterations: processorCount) { threadIdx in
                        let startFrame = threadIdx * framesPerThread
                        let endFrame = min(startFrame + framesPerThread, nFrames)

                        if startFrame >= endFrame { return }

                        // Thread-local workspace (pre-allocate all buffers)
                        var windowedFrame = [Float](repeating: 0, count: nFFT)
                        var splitReal = [Float](repeating: 0, count: halfN)
                        var splitImag = [Float](repeating: 0, count: halfN)
                        var powerBuf = [Float](repeating: 0, count: nFreqs)
                        var scale: Float = 0.5

                        windowedFrame.withUnsafeMutableBufferPointer { framePtr in
                            splitReal.withUnsafeMutableBufferPointer { realPtr in
                                splitImag.withUnsafeMutableBufferPointer { imagPtr in
                                    powerBuf.withUnsafeMutableBufferPointer { powerPtr in
                                        var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                                        for frameIdx in startFrame..<endFrame {
                                            let startIdx = indices[frameIdx]

                                            // Zero the frame buffer
                                            vDSP_vclr(framePtr.baseAddress!, 1, vDSP_Length(nFFT))

                                            // Window the frame
                                            let availableSamples = min(startIdx + winLength, paddedSignal.count) - startIdx
                                            vDSP_vmul(
                                                signalPtr.baseAddress! + startIdx, 1,
                                                windowPtr.baseAddress!, 1,
                                                framePtr.baseAddress!, 1,
                                                vDSP_Length(min(availableSamples, winLength))
                                            )

                                            // Pack into split complex
                                            vDSP_ctoz(
                                                UnsafePointer<DSPComplex>(OpaquePointer(framePtr.baseAddress!)),
                                                2,
                                                &split,
                                                1,
                                                vDSP_Length(halfN)
                                            )

                                            // FFT
                                            vDSP_fft_zrip(self.fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                                            vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, vDSP_Length(halfN))
                                            vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, vDSP_Length(halfN))

                                            // Compute power: r² + i² using vDSP_zvmags (squared magnitude)
                                            vDSP_zvmags(&split, 1, powerPtr.baseAddress!, 1, vDSP_Length(halfN))

                                            // DC is correct, but Nyquist is packed in imagp[0]
                                            let nyq = split.imagp[0]
                                            powerPtr[halfN] = nyq * nyq

                                            // Copy to output (frame-major)
                                            let frameOffset = frameIdx * nFreqs
                                            memcpy(outPtr.baseAddress! + frameOffset, powerPtr.baseAddress!, nFreqs * MemoryLayout<Float>.size)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Transpose to freq-major and build [[Float]] result
        var transposed = [Float](repeating: 0, count: nFreqs * nFrames)
        vDSP_mtrans(powerFrameMajor, 1, &transposed, 1, vDSP_Length(nFreqs), vDSP_Length(nFrames))

        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        transposed.withUnsafeBufferPointer { ptr in
            for f in 0..<nFreqs {
                result.append(Array(UnsafeBufferPointer(start: ptr.baseAddress! + f * nFrames, count: nFrames)))
            }
        }

        return result
    }

    // MARK: - Private Implementation

    private func padSignal(_ signal: [Float]) -> [Float] {
        let padLength = config.nFFT / 2

        switch config.padMode {
        case .reflect:
            return reflectPad(signal, padLength: padLength)
        case .constant(let value):
            let padding = [Float](repeating: value, count: padLength)
            return padding + signal + padding
        case .edge:
            let leftPad = [Float](repeating: signal.first ?? 0, count: padLength)
            let rightPad = [Float](repeating: signal.last ?? 0, count: padLength)
            return leftPad + signal + rightPad
        }
    }

    private func reflectPad(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var result = [Float]()
        result.reserveCapacity(signal.count + 2 * padLength)

        // Left padding (reflect)
        for i in (1...padLength).reversed() {
            let idx = i % signal.count
            result.append(signal[idx])
        }

        // Original signal
        result.append(contentsOf: signal)

        // Right padding (reflect)
        for i in 1...padLength {
            let idx = (signal.count - 1) - (i % signal.count)
            result.append(signal[max(0, idx)])
        }

        return result
    }

    /// Frame the signal into overlapping frames.
    /// Returns frame start indices to avoid copying data.
    private func frameIndices(_ signalLength: Int) -> [Int] {
        guard signalLength >= config.nFFT else { return [] }
        let nFrames = 1 + (signalLength - config.nFFT) / config.hopLength
        var indices = [Int]()
        indices.reserveCapacity(nFrames)

        for i in 0..<nFrames {
            indices.append(i * config.hopLength)
        }

        return indices
    }

    /// Legacy frame extraction - still needed for compatibility.
    /// Consider using frameIndices + direct signal access for better performance.
    private func frameSignal(_ signal: [Float]) -> [[Float]] {
        guard signal.count >= config.nFFT else { return [] }
        let nFrames = 1 + (signal.count - config.nFFT) / config.hopLength
        var frames = [[Float]]()
        frames.reserveCapacity(nFrames)

        for i in 0..<nFrames {
            let start = i * config.hopLength
            let end = min(start + config.nFFT, signal.count)
            var frame = Array(signal[start..<end])

            // Zero-pad if necessary
            if frame.count < config.nFFT {
                frame.append(contentsOf: [Float](repeating: 0, count: config.nFFT - frame.count))
            }

            frames.append(frame)
        }

        return frames
    }

    /// Compute FFT using pre-allocated workspace buffers.
    ///
    /// This method reuses workspace buffers across frames to avoid allocations.
    /// Results are stored directly in workspace.fftReal and workspace.fftImag.
    private func computeFFTWithWorkspace(_ workspace: STFTWorkspace) {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

        // Reset workspace buffers
        workspace.realInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
        workspace.imagInput.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }

        // Convert to split complex format for FFT
        workspace.realInput.withUnsafeMutableBufferPointer { realPtr in
            workspace.imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                // Pack real data into split complex format
                workspace.windowedFrame.withUnsafeBufferPointer { inputPtr in
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(n / 2)
                    )
                }

                // Perform FFT
                vDSP_fft_zrip(fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                // Scale by 1/2 (vDSP FFT convention)
                var scale: Float = 0.5
                vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, vDSP_Length(n / 2))
                vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, vDSP_Length(n / 2))
            }
        }

        // Unpack to full complex spectrum (positive frequencies only)
        // DC component
        workspace.fftReal[0] = workspace.realInput[0]
        workspace.fftImag[0] = 0

        // Positive frequencies
        for i in 1..<(n / 2) {
            workspace.fftReal[i] = workspace.realInput[i]
            workspace.fftImag[i] = workspace.imagInput[i]
        }

        // Nyquist
        workspace.fftReal[n / 2] = workspace.imagInput[0]
        workspace.fftImag[n / 2] = 0
    }

    /// Legacy method for backward compatibility (allocates per call).
    private func computeFFT(_ input: [Float]) -> (real: [Float], imag: [Float]) {
        let workspace = STFTWorkspace(nFFT: config.nFFT)
        workspace.windowedFrame = input
        computeFFTWithWorkspace(workspace)
        return (workspace.fftReal, workspace.fftImag)
    }

    // MARK: - Contiguous Output Methods

    /// Compute STFT with contiguous output format.
    ///
    /// This method outputs directly to contiguous memory layout, avoiding
    /// the intermediate [[Float]] allocations. Preferred for performance-critical
    /// code paths.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram in contiguous format.
    public func transformContiguous(_ signal: [Float]) async -> ContiguousComplexMatrix {
        let paddedSignal = config.center ? padSignal(signal) : signal
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else {
            return ContiguousComplexMatrix(rows: 0, cols: 0)
        }

        let nFreqs = config.nFreqs
        let workspace = STFTWorkspace(nFFT: config.nFFT)

        // Single allocation for output (row-major: freq × frames)
        var realData = [Float](repeating: 0, count: nFreqs * nFrames)
        var imagData = [Float](repeating: 0, count: nFreqs * nFrames)

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            realData.withUnsafeMutableBufferPointer { realPtr in
                imagData.withUnsafeMutableBufferPointer { imagPtr in
                    for (frameIdx, startIdx) in indices.enumerated() {
                        // Zero windowed frame buffer
                        workspace.windowedFrame.withUnsafeMutableBytes {
                            memset($0.baseAddress!, 0, $0.count)
                        }

                        // Window directly from signal
                        let frameEnd = min(startIdx + config.winLength, paddedSignal.count)
                        let availableSamples = frameEnd - startIdx

                        for i in 0..<min(availableSamples, config.winLength) {
                            workspace.windowedFrame[i] = signalPtr[startIdx + i] * window[i]
                        }

                        // Compute FFT
                        computeFFTWithWorkspace(workspace)

                        // Store results in row-major order (row = freq, col = frame)
                        for freqIdx in 0..<nFreqs {
                            let idx = freqIdx * nFrames + frameIdx
                            realPtr[idx] = workspace.fftReal[freqIdx]
                            imagPtr[idx] = workspace.fftImag[freqIdx]
                        }
                    }
                }
            }
        }

        return ContiguousComplexMatrix(real: realData, imag: imagData, rows: nFreqs, cols: nFrames)
    }

    /// Compute magnitude spectrogram with contiguous output.
    ///
    /// This method computes magnitude directly in contiguous memory,
    /// avoiding both [[Float]] and intermediate complex storage.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram in contiguous format.
    public func magnitudeContiguous(_ signal: [Float]) async -> ContiguousMatrix {
        let paddedSignal = config.center ? padSignal(signal) : signal
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else {
            return ContiguousMatrix(rows: 0, cols: 0)
        }

        let nFreqs = config.nFreqs
        let workspace = STFTWorkspace(nFFT: config.nFFT)

        // Single allocation for magnitude output
        var magData = [Float](repeating: 0, count: nFreqs * nFrames)

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            magData.withUnsafeMutableBufferPointer { magPtr in
                for (frameIdx, startIdx) in indices.enumerated() {
                    // Zero windowed frame buffer
                    workspace.windowedFrame.withUnsafeMutableBytes {
                        memset($0.baseAddress!, 0, $0.count)
                    }

                    // Window directly from signal
                    let frameEnd = min(startIdx + config.winLength, paddedSignal.count)
                    let availableSamples = frameEnd - startIdx

                    for i in 0..<min(availableSamples, config.winLength) {
                        workspace.windowedFrame[i] = signalPtr[startIdx + i] * window[i]
                    }

                    // Compute FFT
                    computeFFTWithWorkspace(workspace)

                    // Compute magnitude and store in row-major order
                    workspace.fftReal.withUnsafeBufferPointer { realPtr in
                        workspace.fftImag.withUnsafeBufferPointer { imagPtr in
                            for freqIdx in 0..<nFreqs {
                                let r = realPtr[freqIdx]
                                let i = imagPtr[freqIdx]
                                magPtr[freqIdx * nFrames + frameIdx] = sqrtf(r * r + i * i)
                            }
                        }
                    }
                }
            }
        }

        return ContiguousMatrix(data: magData, rows: nFreqs, cols: nFrames)
    }

    /// Compute power spectrogram with contiguous output.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram (magnitude squared) in contiguous format.
    public func powerContiguous(_ signal: [Float]) async -> ContiguousMatrix {
        let paddedSignal = config.center ? padSignal(signal) : signal
        let indices = frameIndices(paddedSignal.count)
        let nFrames = indices.count

        guard nFrames > 0 else {
            return ContiguousMatrix(rows: 0, cols: 0)
        }

        let nFreqs = config.nFreqs
        let workspace = STFTWorkspace(nFFT: config.nFFT)

        // Single allocation for power output
        var powerData = [Float](repeating: 0, count: nFreqs * nFrames)

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            powerData.withUnsafeMutableBufferPointer { powerPtr in
                for (frameIdx, startIdx) in indices.enumerated() {
                    // Zero windowed frame buffer
                    workspace.windowedFrame.withUnsafeMutableBytes {
                        memset($0.baseAddress!, 0, $0.count)
                    }

                    // Window directly from signal
                    let frameEnd = min(startIdx + config.winLength, paddedSignal.count)
                    let availableSamples = frameEnd - startIdx

                    for i in 0..<min(availableSamples, config.winLength) {
                        workspace.windowedFrame[i] = signalPtr[startIdx + i] * window[i]
                    }

                    // Compute FFT
                    computeFFTWithWorkspace(workspace)

                    // Compute power (mag^2) and store in row-major order
                    workspace.fftReal.withUnsafeBufferPointer { realPtr in
                        workspace.fftImag.withUnsafeBufferPointer { imagPtr in
                            for freqIdx in 0..<nFreqs {
                                let r = realPtr[freqIdx]
                                let i = imagPtr[freqIdx]
                                powerPtr[freqIdx * nFrames + frameIdx] = r * r + i * i
                            }
                        }
                    }
                }
            }
        }

        return ContiguousMatrix(data: powerData, rows: nFreqs, cols: nFrames)
    }
}
