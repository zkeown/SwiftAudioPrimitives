import Accelerate
import Foundation

/// Inverse Short-Time Fourier Transform.
public struct ISTFT: Sendable {
    public let config: STFTConfig

    private let window: [Float]
    private let fftSetupWrapper: FFTSetupWrapper

    /// Create ISTFT processor.
    ///
    /// - Parameter config: STFT configuration (should match the STFT used to create the spectrogram).
    public init(config: STFTConfig = STFTConfig()) {
        self.config = config
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)
    }

    /// Reconstruct signal from complex spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Complex spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length. If nil, computed from spectrogram shape.
    /// - Returns: Reconstructed audio signal.
    public func transform(_ spectrogram: ComplexMatrix, length: Int? = nil) async -> [Float] {
        let nFrames = spectrogram.cols
        guard nFrames > 0 else { return [] }

        // Compute output length
        let expectedLength: Int
        if let length = length {
            expectedLength = length
        } else {
            // Estimate length from spectrogram dimensions
            let uncenteredLength = config.nFFT + (nFrames - 1) * config.hopLength
            expectedLength = config.center ? uncenteredLength - config.nFFT : uncenteredLength
        }

        // Full output buffer length (with padding if centered)
        let bufferLength = config.center
            ? expectedLength + config.nFFT
            : config.nFFT + (nFrames - 1) * config.hopLength

        // Initialize output buffer and normalization buffer
        var output = [Float](repeating: 0, count: bufferLength)
        var windowSum = [Float](repeating: 0, count: bufferLength)

        // Squared window for normalization
        var windowSquared = [Float](repeating: 0, count: config.winLength)
        vDSP_vsq(window, 1, &windowSquared, 1, vDSP_Length(config.winLength))

        // Process each frame
        for frameIdx in 0..<nFrames {
            // Extract complex frame
            var frameReal = [Float](repeating: 0, count: config.nFreqs)
            var frameImag = [Float](repeating: 0, count: config.nFreqs)

            for freqIdx in 0..<config.nFreqs {
                frameReal[freqIdx] = spectrogram.real[freqIdx][frameIdx]
                frameImag[freqIdx] = spectrogram.imag[freqIdx][frameIdx]
            }

            // Compute inverse FFT
            let reconstructedFrame = computeIFFT(real: frameReal, imag: frameImag)

            // Apply synthesis window
            var windowedFrame = [Float](repeating: 0, count: config.winLength)
            for i in 0..<config.winLength {
                windowedFrame[i] = reconstructedFrame[i] * window[i]
            }

            // Overlap-add
            let startIdx = frameIdx * config.hopLength
            for i in 0..<config.winLength {
                let outputIdx = startIdx + i
                if outputIdx < bufferLength {
                    output[outputIdx] += windowedFrame[i]
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

        // Remove padding if centered
        if config.center {
            let padLength = config.nFFT / 2
            let startIdx = padLength
            let endIdx = min(startIdx + expectedLength, bufferLength)
            return Array(output[startIdx..<endIdx])
        } else {
            return Array(output.prefix(expectedLength))
        }
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
