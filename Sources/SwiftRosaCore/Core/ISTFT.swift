import Accelerate
import Foundation

/// Inverse Short-Time Fourier Transform with automatic GPU/CPU dispatch.
///
/// Uses Metal GPU acceleration for large spectrograms and optimized
/// vDSP operations for CPU fallback.
public struct ISTFT: Sendable {
    public let config: STFTConfig

    private let window: [Float]
    private let fftSetupWrapper: FFTSetupWrapper

    /// Pre-computed squared window for normalization.
    private let windowSquared: [Float]

    /// Optional Metal engine for GPU acceleration.
    private let metalEngine: MetalEngine?

    /// Whether to prefer GPU acceleration when available.
    public let useGPU: Bool

    /// Create ISTFT processor.
    ///
    /// - Parameters:
    ///   - config: STFT configuration (should match the STFT used to create the spectrogram).
    ///   - useGPU: Prefer GPU acceleration when available (default: true).
    public init(config: STFTConfig = STFTConfig(), useGPU: Bool = true) {
        self.config = config
        self.useGPU = useGPU
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Pre-compute squared window for normalization
        var squared = [Float](repeating: 0, count: config.winLength)
        vDSP_vsq(window, 1, &squared, 1, vDSP_Length(config.winLength))
        self.windowSquared = squared

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

    /// Reconstruct signal from complex spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Complex spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length. If nil, computed from spectrogram shape.
    /// - Returns: Reconstructed audio signal.
    public func transform(_ spectrogram: ComplexMatrix, length: Int? = nil) async -> [Float] {
        // Convert to contiguous and use optimized implementation
        return await transform(spectrogram.toContiguous(), length: length)
    }

    /// Reconstruct signal from contiguous complex spectrogram (optimized).
    ///
    /// This is the high-performance implementation using transpose-based access
    /// for cache-friendly memory patterns.
    ///
    /// - Parameters:
    ///   - spectrogram: Contiguous complex spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length. If nil, computed from spectrogram shape.
    /// - Returns: Reconstructed audio signal.
    public func transform(_ spectrogram: ContiguousComplexMatrix, length: Int? = nil) async -> [Float] {
        let nFrames = spectrogram.cols
        let nFreqs = config.nFreqs
        let nFFT = config.nFFT
        let winLength = config.winLength
        let hopLength = config.hopLength

        guard nFrames > 0 else { return [] }

        // Compute output length
        let expectedLength: Int
        if let length = length {
            expectedLength = length
        } else {
            let uncenteredLength = nFFT + (nFrames - 1) * hopLength
            expectedLength = config.center ? uncenteredLength - nFFT : uncenteredLength
        }

        // Full output buffer length (with padding if centered)
        let bufferLength = config.center
            ? expectedLength + nFFT
            : nFFT + (nFrames - 1) * hopLength

        let halfN = nFFT / 2
        let log2n = vDSP_Length(log2(Double(nFFT)))

        // OPTIMIZATION: Transpose spectrogram from (nFreqs, nFrames) to (nFrames, nFreqs)
        // This converts column-wise access (cache-hostile) to row-wise access (cache-friendly)
        var transposedReal = [Float](repeating: 0, count: nFreqs * nFrames)
        var transposedImag = [Float](repeating: 0, count: nFreqs * nFrames)

        spectrogram.real.withUnsafeBufferPointer { realPtr in
            spectrogram.imag.withUnsafeBufferPointer { imagPtr in
                // vDSP_mtrans: transpose from (nFreqs rows, nFrames cols) to (nFrames rows, nFreqs cols)
                vDSP_mtrans(realPtr.baseAddress!, 1, &transposedReal, 1,
                            vDSP_Length(nFrames), vDSP_Length(nFreqs))
                vDSP_mtrans(imagPtr.baseAddress!, 1, &transposedImag, 1,
                            vDSP_Length(nFrames), vDSP_Length(nFreqs))
            }
        }

        // Use contiguous memory for frame results: nFrames × winLength
        var frameResults = [Float](repeating: 0, count: nFrames * winLength)

        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        let framesPerThread = (nFrames + processorCount - 1) / processorCount

        // Parallel IFFT processing with cache-friendly access
        frameResults.withUnsafeMutableBufferPointer { resultsPtr in
            transposedReal.withUnsafeBufferPointer { transRealPtr in
                transposedImag.withUnsafeBufferPointer { transImagPtr in
                    DispatchQueue.concurrentPerform(iterations: processorCount) { threadIdx in
                        let startFrame = threadIdx * framesPerThread
                        let endFrame = min(startFrame + framesPerThread, nFrames)

                        if startFrame >= endFrame { return }

                        // Thread-local workspace
                        var fullReal = [Float](repeating: 0, count: halfN)
                        var fullImag = [Float](repeating: 0, count: halfN)
                        var reconstructed = [Float](repeating: 0, count: nFFT)
                        var scale = 1.0 / Float(nFFT)

                        fullReal.withUnsafeMutableBufferPointer { fullRealPtr in
                            fullImag.withUnsafeMutableBufferPointer { fullImagPtr in
                                reconstructed.withUnsafeMutableBufferPointer { reconPtr in
                                    for frameIdx in startFrame..<endFrame {
                                        // After transpose: frame data is at row frameIdx, contiguous in memory
                                        let frameOffset = frameIdx * nFreqs

                                        // DC and Nyquist - now contiguous access
                                        fullRealPtr[0] = transRealPtr[frameOffset]
                                        fullImagPtr[0] = transRealPtr[frameOffset + halfN]  // Nyquist stored in real

                                        // Positive frequencies - bulk copy from contiguous memory
                                        memcpy(fullRealPtr.baseAddress! + 1,
                                               transRealPtr.baseAddress! + frameOffset + 1,
                                               (halfN - 1) * MemoryLayout<Float>.size)
                                        memcpy(fullImagPtr.baseAddress! + 1,
                                               transImagPtr.baseAddress! + frameOffset + 1,
                                               (halfN - 1) * MemoryLayout<Float>.size)

                                        // Inverse FFT
                                        var split = DSPSplitComplex(realp: fullRealPtr.baseAddress!, imagp: fullImagPtr.baseAddress!)
                                        vDSP_fft_zrip(self.fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))

                                        // Unpack to real
                                        vDSP_ztoc(
                                            &split, 1,
                                            UnsafeMutablePointer<DSPComplex>(OpaquePointer(reconPtr.baseAddress!)),
                                            2,
                                            vDSP_Length(halfN)
                                        )

                                        // Scale
                                        vDSP_vsmul(reconPtr.baseAddress!, 1, &scale, reconPtr.baseAddress!, 1, vDSP_Length(nFFT))

                                        // Apply window and store to contiguous output
                                        let resultOffset = frameIdx * winLength
                                        self.window.withUnsafeBufferPointer { windowPtr in
                                            vDSP_vmul(reconPtr.baseAddress!, 1, windowPtr.baseAddress!, 1, resultsPtr.baseAddress! + resultOffset, 1, vDSP_Length(winLength))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sequential overlap-add
        var output = [Float](repeating: 0, count: bufferLength)
        var windowSum = [Float](repeating: 0, count: bufferLength)

        output.withUnsafeMutableBufferPointer { outPtr in
            windowSum.withUnsafeMutableBufferPointer { sumPtr in
                frameResults.withUnsafeBufferPointer { framePtr in
                    for frameIdx in 0..<nFrames {
                        let startIdx = frameIdx * hopLength
                        let addLen = min(winLength, bufferLength - startIdx)
                        let frameOffset = frameIdx * winLength

                        vDSP_vadd(
                            outPtr.baseAddress! + startIdx, 1,
                            framePtr.baseAddress! + frameOffset, 1,
                            outPtr.baseAddress! + startIdx, 1,
                            vDSP_Length(addLen)
                        )

                        self.windowSquared.withUnsafeBufferPointer { sqPtr in
                            vDSP_vadd(
                                sumPtr.baseAddress! + startIdx, 1,
                                sqPtr.baseAddress!, 1,
                                sumPtr.baseAddress! + startIdx, 1,
                                vDSP_Length(addLen)
                            )
                        }
                    }
                }
            }
        }

        // Normalize by squared window sum
        let epsilon: Float = 1e-8
        for i in 0..<bufferLength {
            if windowSum[i] > epsilon {
                output[i] /= windowSum[i]
            }
        }

        // Remove padding if centered
        if config.center {
            let padLength = nFFT / 2
            let startIdx = padLength
            let endIdx = min(startIdx + expectedLength, bufferLength)
            return Array(output[startIdx..<endIdx])
        } else {
            return Array(output.prefix(expectedLength))
        }
    }

    /// CPU overlap-add with window normalization.
    private func overlapAddCPU(frames: [[Float]], bufferLength: Int) -> [Float] {
        var output = [Float](repeating: 0, count: bufferLength)
        var windowSum = [Float](repeating: 0, count: bufferLength)

        for (frameIdx, frame) in frames.enumerated() {
            let startIdx = frameIdx * config.hopLength

            // Apply window and accumulate
            for i in 0..<min(frame.count, config.winLength) {
                let outputIdx = startIdx + i
                if outputIdx < bufferLength {
                    output[outputIdx] += frame[i] * window[i]
                    windowSum[outputIdx] += windowSquared[i]
                }
            }
        }

        // Normalize by squared window sum
        let epsilon: Float = 1e-8
        for i in 0..<bufferLength {
            if windowSum[i] > epsilon {
                output[i] /= windowSum[i]
            }
        }

        return output
    }

    /// Normalize GPU overlap-add output by window sum.
    private func normalizeOutput(_ output: [Float], nFrames: Int, bufferLength: Int) -> [Float] {
        // Compute window sum for normalization
        var windowSum = [Float](repeating: 0, count: bufferLength)

        for frameIdx in 0..<nFrames {
            let startIdx = frameIdx * config.hopLength
            for i in 0..<config.winLength {
                let outputIdx = startIdx + i
                if outputIdx < bufferLength {
                    windowSum[outputIdx] += windowSquared[i]
                }
            }
        }

        // Normalize
        var result = output
        let epsilon: Float = 1e-8
        for i in 0..<min(result.count, bufferLength) {
            if windowSum[i] > epsilon {
                result[i] /= windowSum[i]
            }
        }

        return result
    }

    // MARK: - Private Implementation

    private func computeIFFT(real: [Float], imag: [Float]) -> [Float] {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

        // Prepare full complex spectrum for inverse FFT
        var fullReal = [Float](repeating: 0, count: n / 2)
        var fullImag = [Float](repeating: 0, count: n / 2)

        // Pack DC and Nyquist into packed format
        fullReal[0] = real[0]
        fullImag[0] = real[n / 2]  // Nyquist stored in imag[0] for packed format

        // Copy positive frequencies
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

        // Scale (vDSP inverse FFT convention: divide by 2)
        var scale = 1.0 / Float(n)
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(n))

        return output
    }
}

// MARK: - Convenience Extensions

extension STFT {
    /// Create an ISTFT with matching configuration.
    public var inverse: ISTFT {
        ISTFT(config: config)
    }
}
