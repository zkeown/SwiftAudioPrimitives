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

/// Short-Time Fourier Transform.
public struct STFT: Sendable {
    public let config: STFTConfig

    private let window: [Float]
    private let fftSetupWrapper: FFTSetupWrapper

    /// Create STFT processor.
    ///
    /// - Parameter config: STFT configuration.
    public init(config: STFTConfig = STFTConfig()) {
        self.config = config
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)
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

        // Create workspace once for all frames (major optimization)
        let workspace = STFTWorkspace(nFFT: config.nFFT)

        // Apply window to each frame and compute FFT
        var realResult = [[Float]]()
        var imagResult = [[Float]]()
        realResult.reserveCapacity(config.nFreqs)
        imagResult.reserveCapacity(config.nFreqs)

        // Initialize output arrays
        for _ in 0..<config.nFreqs {
            realResult.append([Float](repeating: 0, count: nFrames))
            imagResult.append([Float](repeating: 0, count: nFrames))
        }

        // Process each frame with reusable workspace
        // Access signal directly via indices to avoid frame copies
        paddedSignal.withUnsafeBufferPointer { signalPtr in
            for (frameIdx, startIdx) in indices.enumerated() {
                // Zero out the windowed frame buffer first
                workspace.windowedFrame.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }

                // Window directly from signal (no intermediate copy)
                let frameEnd = min(startIdx + config.winLength, paddedSignal.count)
                let availableSamples = frameEnd - startIdx

                for i in 0..<min(availableSamples, config.winLength) {
                    workspace.windowedFrame[i] = signalPtr[startIdx + i] * window[i]
                }

                // Compute FFT using workspace buffers
                computeFFTWithWorkspace(workspace)

                // Store results (only positive frequencies)
                for freqIdx in 0..<config.nFreqs {
                    realResult[freqIdx][frameIdx] = workspace.fftReal[freqIdx]
                    imagResult[freqIdx][frameIdx] = workspace.fftImag[freqIdx]
                }
            }
        }

        return ComplexMatrix(real: realResult, imag: imagResult)
    }

    /// Compute magnitude spectrogram.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nFreqs, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let spectrogram = await transform(signal)
        return spectrogram.magnitude
    }

    /// Compute power spectrogram (magnitude squared).
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram of shape (nFreqs, nFrames).
    public func power(_ signal: [Float]) async -> [[Float]] {
        let spectrogram = await transform(signal)

        // Compute power directly from real/imag as r² + i² (avoids sqrt precision loss)
        var result = [[Float]]()
        result.reserveCapacity(spectrogram.rows)

        for f in 0..<spectrogram.rows {
            var powerRow = [Float](repeating: 0, count: spectrogram.cols)
            for t in 0..<spectrogram.cols {
                let r = spectrogram.real[f][t]
                let i = spectrogram.imag[f][t]
                powerRow[t] = r * r + i * i
            }
            result.append(powerRow)
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
