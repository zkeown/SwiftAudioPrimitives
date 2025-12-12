import Accelerate
import Foundation

/// SpecAugment configuration.
public struct SpecAugmentConfig: Sendable {
    /// Maximum width of frequency masks (F parameter).
    public let frequencyMaskParam: Int

    /// Maximum width of time masks (T parameter).
    public let timeMaskParam: Int

    /// Number of frequency masks to apply.
    public let numFrequencyMasks: Int

    /// Number of time masks to apply.
    public let numTimeMasks: Int

    /// Value to fill masked regions (nil = use mean of spectrogram).
    public let maskValue: Float?

    /// Create SpecAugment configuration.
    ///
    /// - Parameters:
    ///   - frequencyMaskParam: Max frequency mask width (default: 27).
    ///   - timeMaskParam: Max time mask width (default: 100).
    ///   - numFrequencyMasks: Number of frequency masks (default: 1).
    ///   - numTimeMasks: Number of time masks (default: 1).
    ///   - maskValue: Fill value for masks (default: nil = mean).
    public init(
        frequencyMaskParam: Int = 27,
        timeMaskParam: Int = 100,
        numFrequencyMasks: Int = 1,
        numTimeMasks: Int = 1,
        maskValue: Float? = nil
    ) {
        self.frequencyMaskParam = frequencyMaskParam
        self.timeMaskParam = timeMaskParam
        self.numFrequencyMasks = numFrequencyMasks
        self.numTimeMasks = numTimeMasks
        self.maskValue = maskValue
    }
}

/// SpecAugment for spectrogram augmentation.
///
/// SpecAugment is a simple but effective data augmentation technique for
/// speech recognition, introduced by Google. It applies random time and
/// frequency masks to spectrograms during training.
///
/// ## Algorithm
///
/// 1. Apply `numFrequencyMasks` random frequency masks of width up to `frequencyMaskParam`
/// 2. Apply `numTimeMasks` random time masks of width up to `timeMaskParam`
/// 3. Masked regions are filled with `maskValue` (or spectrogram mean)
///
/// ## Example Usage
///
/// ```swift
/// let specAugment = SpecAugment(config: SpecAugmentConfig(
///     frequencyMaskParam: 27,
///     timeMaskParam: 100
/// ))
///
/// let augmented = specAugment.transform(melSpectrogram)
/// ```
///
/// ## References
///
/// Park et al., "SpecAugment: A Simple Data Augmentation Method for
/// Automatic Speech Recognition", Interspeech 2019.
public struct SpecAugment: Sendable {
    /// Configuration.
    public let config: SpecAugmentConfig

    /// Create a SpecAugment processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: SpecAugmentConfig = SpecAugmentConfig()) {
        self.config = config
    }

    /// Apply SpecAugment to spectrogram.
    ///
    /// - Parameter spectrogram: Input spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Augmented spectrogram of same shape.
    public func transform(_ spectrogram: [[Float]]) -> [[Float]] {
        guard !spectrogram.isEmpty && !spectrogram[0].isEmpty else { return spectrogram }

        var augmented = spectrogram
        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        // Compute fill value
        let fillValue: Float
        if let value = config.maskValue {
            fillValue = value
        } else {
            // Use mean of spectrogram
            var total: Float = 0
            var count = 0
            for row in spectrogram {
                for val in row {
                    total += val
                    count += 1
                }
            }
            fillValue = count > 0 ? total / Float(count) : 0
        }

        // Apply frequency masks
        for _ in 0..<config.numFrequencyMasks {
            let maskWidth = Int.random(in: 0...min(config.frequencyMaskParam, nFreqs))
            if maskWidth > 0 {
                let f0 = Int.random(in: 0...(nFreqs - maskWidth))

                for f in f0..<(f0 + maskWidth) {
                    for t in 0..<nFrames {
                        augmented[f][t] = fillValue
                    }
                }
            }
        }

        // Apply time masks
        for _ in 0..<config.numTimeMasks {
            let maskWidth = Int.random(in: 0...min(config.timeMaskParam, nFrames))
            if maskWidth > 0 {
                let t0 = Int.random(in: 0...(nFrames - maskWidth))

                for f in 0..<nFreqs {
                    for t in t0..<(t0 + maskWidth) {
                        augmented[f][t] = fillValue
                    }
                }
            }
        }

        return augmented
    }

    /// Apply only frequency masking.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram.
    ///   - numMasks: Number of masks (default: from config).
    ///   - maxWidth: Maximum mask width (default: from config).
    /// - Returns: Augmented spectrogram.
    public func frequencyMask(
        _ spectrogram: [[Float]],
        numMasks: Int? = nil,
        maxWidth: Int? = nil
    ) -> [[Float]] {
        guard !spectrogram.isEmpty && !spectrogram[0].isEmpty else { return spectrogram }

        var augmented = spectrogram
        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        let fillValue = computeFillValue(spectrogram)
        let masks = numMasks ?? config.numFrequencyMasks
        let width = maxWidth ?? config.frequencyMaskParam

        for _ in 0..<masks {
            let maskWidth = Int.random(in: 0...min(width, nFreqs))
            if maskWidth > 0 {
                let f0 = Int.random(in: 0...(nFreqs - maskWidth))

                for f in f0..<(f0 + maskWidth) {
                    for t in 0..<nFrames {
                        augmented[f][t] = fillValue
                    }
                }
            }
        }

        return augmented
    }

    /// Apply only time masking.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram.
    ///   - numMasks: Number of masks (default: from config).
    ///   - maxWidth: Maximum mask width (default: from config).
    /// - Returns: Augmented spectrogram.
    public func timeMask(
        _ spectrogram: [[Float]],
        numMasks: Int? = nil,
        maxWidth: Int? = nil
    ) -> [[Float]] {
        guard !spectrogram.isEmpty && !spectrogram[0].isEmpty else { return spectrogram }

        var augmented = spectrogram
        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        let fillValue = computeFillValue(spectrogram)
        let masks = numMasks ?? config.numTimeMasks
        let width = maxWidth ?? config.timeMaskParam

        for _ in 0..<masks {
            let maskWidth = Int.random(in: 0...min(width, nFrames))
            if maskWidth > 0 {
                let t0 = Int.random(in: 0...(nFrames - maskWidth))

                for f in 0..<nFreqs {
                    for t in t0..<(t0 + maskWidth) {
                        augmented[f][t] = fillValue
                    }
                }
            }
        }

        return augmented
    }

    /// Apply time warping (simple implementation).
    ///
    /// Warps the spectrogram along the time axis by a random amount.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram.
    ///   - warpParam: Maximum warp distance (default: 80).
    /// - Returns: Warped spectrogram.
    public func timeWarp(
        _ spectrogram: [[Float]],
        warpParam: Int = 80
    ) -> [[Float]] {
        guard !spectrogram.isEmpty && !spectrogram[0].isEmpty else { return spectrogram }

        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        guard nFrames > 2 * warpParam else { return spectrogram }

        // Choose a random source point and warp distance
        let sourcePoint = Int.random(in: warpParam...(nFrames - warpParam))
        let warpDistance = Int.random(in: -warpParam...warpParam)
        let destPoint = sourcePoint + warpDistance

        var augmented = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFreqs)

        // Simple linear warping (proper implementation would use sparse_image_warp)
        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                // Map output time to input time
                let inputT: Float
                if t < destPoint {
                    inputT = Float(t) * Float(sourcePoint) / Float(destPoint)
                } else {
                    inputT = Float(sourcePoint) + Float(t - destPoint) * Float(nFrames - sourcePoint) / Float(nFrames - destPoint)
                }

                // Linear interpolation
                let t0 = Int(inputT)
                let t1 = min(t0 + 1, nFrames - 1)
                let alpha = inputT - Float(t0)

                augmented[f][t] = spectrogram[f][t0] * (1 - alpha) + spectrogram[f][t1] * alpha
            }
        }

        return augmented
    }

    // MARK: - Private Helpers

    private func computeFillValue(_ spectrogram: [[Float]]) -> Float {
        if let value = config.maskValue {
            return value
        }

        var total: Float = 0
        var count = 0
        for row in spectrogram {
            for val in row {
                total += val
                count += 1
            }
        }
        return count > 0 ? total / Float(count) : 0
    }
}

// MARK: - Presets

extension SpecAugment {
    /// LibriSpeech Double (LD) policy from original paper.
    public static var libriSpeech: SpecAugment {
        SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 27,
            timeMaskParam: 100,
            numFrequencyMasks: 2,
            numTimeMasks: 2,
            maskValue: nil
        ))
    }

    /// LibriSpeech Basic (LB) policy.
    public static var libriSpeechBasic: SpecAugment {
        SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 27,
            timeMaskParam: 100,
            numFrequencyMasks: 1,
            numTimeMasks: 1,
            maskValue: nil
        ))
    }

    /// Mild augmentation for fine-tuning.
    public static var mild: SpecAugment {
        SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 15,
            timeMaskParam: 50,
            numFrequencyMasks: 1,
            numTimeMasks: 1,
            maskValue: nil
        ))
    }

    /// Aggressive augmentation for robust training.
    public static var aggressive: SpecAugment {
        SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 40,
            timeMaskParam: 150,
            numFrequencyMasks: 3,
            numTimeMasks: 3,
            maskValue: nil
        ))
    }

    /// No augmentation (identity transform).
    public static var none: SpecAugment {
        SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 0,
            timeMaskParam: 0,
            numFrequencyMasks: 0,
            numTimeMasks: 0,
            maskValue: nil
        ))
    }
}
