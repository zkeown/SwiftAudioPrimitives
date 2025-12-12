import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for Harmonic-Percussive Source Separation.
public struct HPSSConfig: Sendable {
    /// Kernel size for median filtering (harmonic, percussive).
    public let kernelSize: (harmonic: Int, percussive: Int)

    /// Power for soft masking (2.0 = Wiener filtering).
    public let power: Float

    /// Margin for mask separation (1.0 = H + P = S, >1.0 allows residual).
    public let margin: Float

    /// Create HPSS configuration.
    ///
    /// - Parameters:
    ///   - kernelSize: Median filter kernel sizes (default: (31, 31)).
    ///   - power: Soft mask power (default: 2.0).
    ///   - margin: Separation margin (default: 1.0).
    public init(
        kernelSize: (harmonic: Int, percussive: Int) = (31, 31),
        power: Float = 2.0,
        margin: Float = 1.0
    ) {
        // Ensure odd kernel sizes
        self.kernelSize = (
            harmonic: kernelSize.harmonic | 1,
            percussive: kernelSize.percussive | 1
        )
        self.power = power
        self.margin = margin
    }
}

/// Result of HPSS separation.
public struct HPSSResult: Sendable {
    /// Harmonic component (sustained tones).
    public let harmonic: [Float]

    /// Percussive component (transients).
    public let percussive: [Float]

    /// Residual component (when margin > 1).
    public let residual: [Float]?
}

/// HPSS soft masks for spectrogram-domain processing.
public struct HPSSMasks: Sendable {
    /// Mask for harmonic component.
    public let harmonicMask: [[Float]]

    /// Mask for percussive component.
    public let percussiveMask: [[Float]]
}

/// Harmonic-Percussive Source Separation.
///
/// HPSS separates an audio signal into harmonic (sustained) and percussive
/// (transient) components using median filtering in the time-frequency domain.
///
/// ## Algorithm Overview
///
/// 1. Compute STFT of input signal
/// 2. Apply median filter along time axis → harmonic-enhanced spectrogram
/// 3. Apply median filter along frequency axis → percussive-enhanced spectrogram
/// 4. Compute soft masks using Wiener filtering
/// 5. Apply masks to original spectrogram
/// 6. Reconstruct audio via inverse STFT
///
/// ## Example Usage
///
/// ```swift
/// let hpss = HPSS()
///
/// // Separate into harmonic and percussive
/// let result = await hpss.separate(audioSignal)
///
/// // Get only harmonic component
/// let harmonic = await hpss.harmonic(audioSignal)
/// ```
public struct HPSS: Sendable {
    /// HPSS configuration.
    public let config: HPSSConfig

    /// STFT configuration.
    public let stftConfig: STFTConfig

    private let stft: STFT
    private let istft: ISTFT

    /// Create an HPSS processor.
    ///
    /// - Parameters:
    ///   - hpssConfig: HPSS configuration.
    ///   - stftConfig: STFT configuration.
    public init(
        hpssConfig: HPSSConfig = HPSSConfig(),
        stftConfig: STFTConfig = STFTConfig()
    ) {
        self.config = hpssConfig
        self.stftConfig = stftConfig
        self.stft = STFT(config: stftConfig)
        self.istft = ISTFT(config: stftConfig)
    }

    /// Separate signal into harmonic and percussive components.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: HPSSResult with harmonic, percussive, and optional residual.
    public func separate(_ signal: [Float]) async -> HPSSResult {
        guard !signal.isEmpty else {
            return HPSSResult(harmonic: [], percussive: [], residual: nil)
        }

        let targetLength = signal.count

        // Compute STFT
        let spectrogram = await stft.transform(signal)

        // Get masks
        let masks = separateMasks(spectrogram: spectrogram)

        // Apply masks to complex spectrogram
        let harmonicSpec = applyMask(spectrogram, mask: masks.harmonicMask)
        let percussiveSpec = applyMask(spectrogram, mask: masks.percussiveMask)

        // Reconstruct audio
        var harmonic = await istft.transform(harmonicSpec)
        var percussive = await istft.transform(percussiveSpec)

        // Match output length to input length
        harmonic = matchLength(harmonic, targetLength: targetLength)
        percussive = matchLength(percussive, targetLength: targetLength)

        // Compute residual if margin > 1
        var residual: [Float]? = nil
        if config.margin > 1.0 {
            residual = [Float](repeating: 0, count: targetLength)
            for i in 0..<targetLength {
                residual![i] = signal[i] - harmonic[i] - percussive[i]
            }
        }

        return HPSSResult(
            harmonic: harmonic,
            percussive: percussive,
            residual: residual
        )
    }

    /// Match array length to target by trimming or zero-padding.
    private func matchLength(_ array: [Float], targetLength: Int) -> [Float] {
        if array.count == targetLength {
            return array
        } else if array.count > targetLength {
            return Array(array.prefix(targetLength))
        } else {
            var result = array
            result.append(contentsOf: [Float](repeating: 0, count: targetLength - array.count))
            return result
        }
    }

    /// Get only harmonic component.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Harmonic audio signal.
    public func harmonic(_ signal: [Float]) async -> [Float] {
        let result = await separate(signal)
        return result.harmonic
    }

    /// Get only percussive component.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Percussive audio signal.
    public func percussive(_ signal: [Float]) async -> [Float] {
        let result = await separate(signal)
        return result.percussive
    }

    /// Compute separation masks from magnitude spectrogram.
    ///
    /// - Parameter magnitude: Magnitude spectrogram.
    /// - Returns: Harmonic and percussive masks.
    public func separateMasks(magnitude: [[Float]]) -> HPSSMasks {
        guard !magnitude.isEmpty && !magnitude[0].isEmpty else {
            return HPSSMasks(harmonicMask: [], percussiveMask: [])
        }

        let nFreqs = magnitude.count
        let nFrames = magnitude[0].count

        // Harmonic enhancement: median filter along time (horizontal)
        let harmonicEnhanced = medianFilterHorizontal(magnitude, kernelSize: config.kernelSize.harmonic)

        // Percussive enhancement: median filter along frequency (vertical)
        let percussiveEnhanced = medianFilterVertical(magnitude, kernelSize: config.kernelSize.percussive)

        // Compute soft masks using Wiener filtering
        var harmonicMask = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)
        var percussiveMask = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)

        let margin = config.margin
        let power = config.power

        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                let hVal = pow(harmonicEnhanced[f][t], power)
                let pVal = pow(percussiveEnhanced[f][t], power)

                // Apply margin
                let hMarg = hVal * margin
                let pMarg = pVal * margin

                let total = hMarg + pMarg
                if total > 0 {
                    harmonicMask[f][t] = hMarg / total
                    percussiveMask[f][t] = pMarg / total
                } else {
                    harmonicMask[f][t] = 0.5
                    percussiveMask[f][t] = 0.5
                }
            }
        }

        return HPSSMasks(harmonicMask: harmonicMask, percussiveMask: percussiveMask)
    }

    /// Compute separation masks from complex spectrogram.
    private func separateMasks(spectrogram: ComplexMatrix) -> HPSSMasks {
        let magnitude = spectrogram.magnitude
        return separateMasks(magnitude: magnitude)
    }

    // MARK: - Private Implementation

    /// Apply mask to complex spectrogram.
    private func applyMask(_ spectrogram: ComplexMatrix, mask: [[Float]]) -> ComplexMatrix {
        let nFreqs = spectrogram.rows
        let nFrames = spectrogram.cols

        var realMasked = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)
        var imagMasked = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)

        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                let m = mask[f][t]
                realMasked[f][t] = spectrogram.real[f][t] * m
                imagMasked[f][t] = spectrogram.imag[f][t] * m
            }
        }

        return ComplexMatrix(real: realMasked, imag: imagMasked)
    }

    /// Median filter along horizontal (time) axis - for harmonic enhancement.
    private func medianFilterHorizontal(_ matrix: [[Float]], kernelSize: Int) -> [[Float]] {
        let nFreqs = matrix.count
        let nFrames = matrix[0].count
        let halfKernel = kernelSize / 2

        var result = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)

        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                // Collect values in kernel window
                var window = [Float]()
                window.reserveCapacity(kernelSize)

                for dt in -halfKernel...halfKernel {
                    let tIdx = t + dt
                    if tIdx >= 0 && tIdx < nFrames {
                        window.append(matrix[f][tIdx])
                    }
                }

                // Compute median
                window.sort()
                result[f][t] = window[window.count / 2]
            }
        }

        return result
    }

    /// Median filter along vertical (frequency) axis - for percussive enhancement.
    private func medianFilterVertical(_ matrix: [[Float]], kernelSize: Int) -> [[Float]] {
        let nFreqs = matrix.count
        let nFrames = matrix[0].count
        let halfKernel = kernelSize / 2

        var result = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)

        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                // Collect values in kernel window
                var window = [Float]()
                window.reserveCapacity(kernelSize)

                for df in -halfKernel...halfKernel {
                    let fIdx = f + df
                    if fIdx >= 0 && fIdx < nFreqs {
                        window.append(matrix[fIdx][t])
                    }
                }

                // Compute median
                window.sort()
                result[f][t] = window[window.count / 2]
            }
        }

        return result
    }
}

// MARK: - Presets

extension HPSS {
    /// Default configuration suitable for most audio.
    public static var standard: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (31, 31),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 2048,
                hopLength: 512,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Configuration optimized for separating vocals from drums.
    public static var vocalsFromDrums: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (15, 15),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 4096,
                hopLength: 1024,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Configuration with residual extraction.
    public static var withResidual: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (31, 31),
                power: 2.0,
                margin: 2.0
            ),
            stftConfig: STFTConfig(
                nFFT: 2048,
                hopLength: 512,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Faster configuration with smaller kernel.
    public static var fast: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (11, 11),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 1024,
                hopLength: 256,
                windowType: .hann,
                center: true
            )
        )
    }
}
