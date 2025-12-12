import SwiftRosaCore
import Accelerate
import Foundation

/// Feature normalization utilities.
///
/// Provides various normalization techniques for audio features (MFCCs, mel spectrograms, etc.)
/// commonly used to improve model robustness and generalization.
///
/// ## Example Usage
///
/// ```swift
/// // Per-utterance mean normalization
/// let normalized = FeatureNormalization.meanNormalize(features)
///
/// // Cepstral Mean and Variance Normalization (CMVN)
/// let cmvnNormalized = FeatureNormalization.cmvn(features)
///
/// // Using pre-computed global statistics
/// let stats = FeatureNormalization.computeStatistics(trainingFeatures)
/// let normalized = FeatureNormalization.cmvn(testFeatures, globalMean: stats.mean, globalVar: stats.variance)
/// ```
public struct FeatureNormalization: Sendable {
    private init() {}

    // MARK: - Per-Utterance Normalization

    /// Per-utterance mean normalization.
    ///
    /// Subtracts the mean of each feature dimension.
    ///
    /// - Parameter features: Feature matrix of shape (nFeatures, nFrames).
    /// - Returns: Mean-normalized features.
    public static func meanNormalize(_ features: [[Float]]) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var normalized = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)

        for f in 0..<nFeatures {
            // Compute mean
            var mean: Float = 0
            vDSP_meanv(features[f], 1, &mean, vDSP_Length(nFrames))

            // Subtract mean
            var negMean = -mean
            vDSP_vsadd(features[f], 1, &negMean, &normalized[f], 1, vDSP_Length(nFrames))
        }

        return normalized
    }

    /// Per-utterance mean and variance normalization.
    ///
    /// Normalizes each feature dimension to zero mean and unit variance.
    ///
    /// - Parameter features: Feature matrix of shape (nFeatures, nFrames).
    /// - Returns: Mean-variance normalized features.
    public static func mvNormalize(_ features: [[Float]]) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var normalized = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)

        for f in 0..<nFeatures {
            // Compute mean
            var mean: Float = 0
            vDSP_meanv(features[f], 1, &mean, vDSP_Length(nFrames))

            // Compute variance
            var variance: Float = 0
            var squared = [Float](repeating: 0, count: nFrames)
            var negMean = -mean
            vDSP_vsadd(features[f], 1, &negMean, &squared, 1, vDSP_Length(nFrames))
            vDSP_vsq(squared, 1, &squared, 1, vDSP_Length(nFrames))
            vDSP_meanv(squared, 1, &variance, vDSP_Length(nFrames))

            let stdDev = sqrt(max(variance, 1e-10))

            // Normalize: (x - mean) / std
            for t in 0..<nFrames {
                normalized[f][t] = (features[f][t] - mean) / stdDev
            }
        }

        return normalized
    }

    /// Cepstral Mean and Variance Normalization (CMVN).
    ///
    /// Standard technique for speaker normalization in speech recognition.
    /// Can use either per-utterance or global (pre-computed) statistics.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - globalMean: Optional global mean per feature (uses utterance mean if nil).
    ///   - globalVar: Optional global variance per feature (uses utterance variance if nil).
    /// - Returns: CMVN-normalized features.
    public static func cmvn(
        _ features: [[Float]],
        globalMean: [Float]? = nil,
        globalVar: [Float]? = nil
    ) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var normalized = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)

        for f in 0..<nFeatures {
            // Get mean (global or per-utterance)
            let mean: Float
            if let global = globalMean, f < global.count {
                mean = global[f]
            } else {
                var m: Float = 0
                vDSP_meanv(features[f], 1, &m, vDSP_Length(nFrames))
                mean = m
            }

            // Get variance (global or per-utterance)
            let variance: Float
            if let global = globalVar, f < global.count {
                variance = global[f]
            } else {
                var squared = [Float](repeating: 0, count: nFrames)
                var negMean = -mean
                vDSP_vsadd(features[f], 1, &negMean, &squared, 1, vDSP_Length(nFrames))
                vDSP_vsq(squared, 1, &squared, 1, vDSP_Length(nFrames))
                var v: Float = 0
                vDSP_meanv(squared, 1, &v, vDSP_Length(nFrames))
                variance = v
            }

            let stdDev = sqrt(max(variance, 1e-10))

            // Normalize
            for t in 0..<nFrames {
                normalized[f][t] = (features[f][t] - mean) / stdDev
            }
        }

        return normalized
    }

    // MARK: - Scaling

    /// Min-max scaling to [0, 1] range.
    ///
    /// - Parameter features: Feature matrix of shape (nFeatures, nFrames).
    /// - Returns: Scaled features in [0, 1] range.
    public static func minMaxScale(_ features: [[Float]]) -> [[Float]] {
        return minMaxScale(features, min: 0, max: 1)
    }

    /// Min-max scaling to custom range.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - min: Target minimum value.
    ///   - max: Target maximum value.
    /// - Returns: Scaled features in [min, max] range.
    public static func minMaxScale(
        _ features: [[Float]],
        min targetMin: Float,
        max targetMax: Float
    ) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var normalized = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)

        for f in 0..<nFeatures {
            // Find min and max
            var minVal: Float = 0
            var maxVal: Float = 0
            vDSP_minv(features[f], 1, &minVal, vDSP_Length(nFrames))
            vDSP_maxv(features[f], 1, &maxVal, vDSP_Length(nFrames))

            let range = maxVal - minVal
            let targetRange = targetMax - targetMin

            if range > 0 {
                // Scale to target range
                for t in 0..<nFrames {
                    normalized[f][t] = targetMin + (features[f][t] - minVal) * targetRange / range
                }
            } else {
                // All values same, map to middle of target range
                let midpoint = (targetMin + targetMax) / 2
                for t in 0..<nFrames {
                    normalized[f][t] = midpoint
                }
            }
        }

        return normalized
    }

    /// L2 normalization (unit norm per frame).
    ///
    /// - Parameter features: Feature matrix of shape (nFeatures, nFrames).
    /// - Returns: L2-normalized features.
    public static func l2Normalize(_ features: [[Float]]) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var normalized = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)

        for t in 0..<nFrames {
            // Compute L2 norm for this frame
            var sumSquares: Float = 0
            for f in 0..<nFeatures {
                sumSquares += features[f][t] * features[f][t]
            }

            let norm = sqrt(max(sumSquares, 1e-10))

            // Normalize
            for f in 0..<nFeatures {
                normalized[f][t] = features[f][t] / norm
            }
        }

        return normalized
    }

    // MARK: - Statistics Computation

    /// Compute mean and variance statistics from features.
    ///
    /// Use this to compute global statistics from a training set for use
    /// with CMVN normalization.
    ///
    /// - Parameter features: Feature matrix of shape (nFeatures, nFrames).
    /// - Returns: Per-feature mean and variance.
    public static func computeStatistics(
        _ features: [[Float]]
    ) -> (mean: [Float], variance: [Float]) {
        guard !features.isEmpty && !features[0].isEmpty else {
            return ([], [])
        }

        let nFeatures = features.count
        let nFrames = features[0].count

        var mean = [Float](repeating: 0, count: nFeatures)
        var variance = [Float](repeating: 0, count: nFeatures)

        for f in 0..<nFeatures {
            // Compute mean
            vDSP_meanv(features[f], 1, &mean[f], vDSP_Length(nFrames))

            // Compute variance
            var squared = [Float](repeating: 0, count: nFrames)
            var negMean = -mean[f]
            vDSP_vsadd(features[f], 1, &negMean, &squared, 1, vDSP_Length(nFrames))
            vDSP_vsq(squared, 1, &squared, 1, vDSP_Length(nFrames))
            vDSP_meanv(squared, 1, &variance[f], vDSP_Length(nFrames))
        }

        return (mean, variance)
    }

    /// Compute statistics from multiple utterances.
    ///
    /// - Parameter allFeatures: Array of feature matrices.
    /// - Returns: Global per-feature mean and variance.
    public static func computeGlobalStatistics(
        _ allFeatures: [[[Float]]]
    ) -> (mean: [Float], variance: [Float]) {
        guard !allFeatures.isEmpty else { return ([], []) }

        // Get feature dimension from first utterance
        guard let first = allFeatures.first, !first.isEmpty else {
            return ([], [])
        }

        let nFeatures = first.count

        var sumX = [Float](repeating: 0, count: nFeatures)
        var sumX2 = [Float](repeating: 0, count: nFeatures)
        var totalFrames = 0

        for features in allFeatures {
            guard features.count == nFeatures else { continue }

            let nFrames = features[0].count
            totalFrames += nFrames

            for f in 0..<nFeatures {
                for t in 0..<nFrames {
                    sumX[f] += features[f][t]
                    sumX2[f] += features[f][t] * features[f][t]
                }
            }
        }

        guard totalFrames > 0 else { return ([], []) }

        var mean = [Float](repeating: 0, count: nFeatures)
        var variance = [Float](repeating: 0, count: nFeatures)

        let n = Float(totalFrames)
        for f in 0..<nFeatures {
            mean[f] = sumX[f] / n
            variance[f] = (sumX2[f] / n) - (mean[f] * mean[f])
        }

        return (mean, variance)
    }
}

// MARK: - Streaming CMVN

/// Streaming CMVN accumulator.
///
/// Maintains running statistics for online CMVN normalization.
///
/// ## Example Usage
///
/// ```swift
/// let cmvn = StreamingCMVN(nFeatures: 13)
///
/// // During streaming
/// for frame in audioFrames {
///     let features = extractFeatures(frame)
///     await cmvn.update(features)
///     let normalized = await cmvn.normalize(features)
/// }
/// ```
public actor StreamingCMVN {
    private var sumX: [Float]
    private var sumX2: [Float]
    private var count: Int
    private let nFeatures: Int

    /// Minimum samples before normalization is applied.
    public let minSamples: Int

    /// Create a streaming CMVN accumulator.
    ///
    /// - Parameters:
    ///   - nFeatures: Number of feature dimensions.
    ///   - minSamples: Minimum frames before normalization starts (default: 10).
    public init(nFeatures: Int, minSamples: Int = 10) {
        self.nFeatures = nFeatures
        self.minSamples = minSamples
        self.sumX = [Float](repeating: 0, count: nFeatures)
        self.sumX2 = [Float](repeating: 0, count: nFeatures)
        self.count = 0
    }

    /// Update running statistics with a new frame.
    ///
    /// - Parameter frame: Feature vector of length nFeatures.
    public func update(_ frame: [Float]) {
        guard frame.count == nFeatures else { return }

        for f in 0..<nFeatures {
            sumX[f] += frame[f]
            sumX2[f] += frame[f] * frame[f]
        }
        count += 1
    }

    /// Normalize a frame using current running statistics.
    ///
    /// - Parameter frame: Feature vector of length nFeatures.
    /// - Returns: Normalized feature vector.
    public func normalize(_ frame: [Float]) -> [Float] {
        guard frame.count == nFeatures else { return frame }
        guard count >= minSamples else { return frame }

        var normalized = [Float](repeating: 0, count: nFeatures)
        let n = Float(count)

        for f in 0..<nFeatures {
            let mean = sumX[f] / n
            let variance = (sumX2[f] / n) - (mean * mean)
            let stdDev = sqrt(max(variance, 1e-10))

            normalized[f] = (frame[f] - mean) / stdDev
        }

        return normalized
    }

    /// Get current mean values.
    public var currentMean: [Float] {
        guard count > 0 else { return [Float](repeating: 0, count: nFeatures) }
        let n = Float(count)
        return sumX.map { $0 / n }
    }

    /// Get current variance values.
    public var currentVariance: [Float] {
        guard count > 0 else { return [Float](repeating: 1, count: nFeatures) }
        let n = Float(count)
        let mean = currentMean
        return (0..<nFeatures).map { f in
            (sumX2[f] / n) - (mean[f] * mean[f])
        }
    }

    /// Reset statistics.
    public func reset() {
        sumX = [Float](repeating: 0, count: nFeatures)
        sumX2 = [Float](repeating: 0, count: nFeatures)
        count = 0
    }

    /// Number of frames seen.
    public var frameCount: Int { count }
}
