import Accelerate
import Foundation
import SwiftRosaCore

/// Aggregation method for nearest-neighbor filtering.
public enum NNAggregation: Sendable {
    /// Median of neighbors (robust to outliers).
    case median
    /// Mean of neighbors.
    case mean
    /// Maximum of neighbors.
    case max
    /// Minimum of neighbors.
    case min
}

/// Configuration for nearest-neighbor filtering.
public struct NNFilterConfig: Sendable {
    /// Number of nearest neighbors to consider.
    public let k: Int

    /// Temporal window width (neighbors within ±width frames).
    public let width: Int

    /// Aggregation method.
    public let aggregation: NNAggregation

    /// Distance metric.
    public let metric: DistanceMetric

    /// Create NN filter configuration.
    ///
    /// - Parameters:
    ///   - k: Number of neighbors (default: 5).
    ///   - width: Temporal window width (default: 31 frames on each side).
    ///   - aggregation: Aggregation method (default: median).
    ///   - metric: Distance metric (default: euclidean).
    public init(
        k: Int = 5,
        width: Int = 31,
        aggregation: NNAggregation = .median,
        metric: DistanceMetric = .euclidean
    ) {
        self.k = k
        self.width = width
        self.aggregation = aggregation
        self.metric = metric
    }
}

/// Distance metric for neighbor computation.
public enum DistanceMetric: Sendable {
    /// Euclidean distance.
    case euclidean
    /// Cosine distance (1 - cosine similarity).
    case cosine
    /// Manhattan distance (L1 norm).
    case manhattan
}

/// Nearest-neighbor filtering for spectrograms.
///
/// Filters spectrogram frames by replacing each frame with an aggregation
/// of its k nearest neighbors in the feature space. This is useful for:
///
/// - **Denoising**: Remove transient noise by replacing with median of similar frames
/// - **Smoothing**: Smooth temporal variations while preserving spectral structure
/// - **Soft masking**: Create time-frequency masks based on neighbor similarity
///
/// ## Example Usage
///
/// ```swift
/// let nnFilter = NNFilter(config: NNFilterConfig(k: 7, aggregation: .median))
///
/// // Filter a spectrogram
/// let filtered = nnFilter.filter(spectrogram)
///
/// // Compute soft mask
/// let mask = nnFilter.softMask(spectrogram)
///
/// // Apply to original
/// let masked = nnFilter.apply(mask: mask, to: spectrogram)
/// ```
///
/// ## Algorithm
///
/// For each frame j:
/// 1. Compute distances to all frames within temporal window
/// 2. Select k nearest neighbors
/// 3. Aggregate neighbor values (median, mean, etc.)
/// 4. Replace original frame with aggregated result
public struct NNFilter: Sendable {
    /// Configuration.
    public let config: NNFilterConfig

    /// Create NN filter.
    ///
    /// - Parameter config: Configuration.
    public init(config: NNFilterConfig = NNFilterConfig()) {
        self.config = config
    }

    /// Filter a spectrogram using nearest-neighbor aggregation.
    ///
    /// - Parameter features: Input spectrogram of shape (nFeatures, nFrames).
    /// - Returns: Filtered spectrogram of shape (nFeatures, nFrames).
    public func filter(_ features: [[Float]]) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        // Find nearest neighbors for each frame
        let neighbors = findNeighbors(features: features)

        // Apply aggregation
        var result = [[Float]]()
        result.reserveCapacity(nFeatures)

        for f in 0..<nFeatures {
            var row = [Float](repeating: 0, count: nFrames)

            for j in 0..<nFrames {
                // Gather values from neighbors
                var values = [Float]()
                values.reserveCapacity(config.k)

                for neighborIdx in neighbors[j] {
                    values.append(features[f][neighborIdx])
                }

                row[j] = aggregate(values)
            }

            result.append(row)
        }

        return result
    }

    /// Compute a soft mask based on neighbor similarity.
    ///
    /// The mask value for each time-frequency bin is based on how similar
    /// the frame is to its neighbors. Frames that are outliers get lower mask values.
    ///
    /// - Parameter features: Input spectrogram of shape (nFeatures, nFrames).
    /// - Returns: Soft mask of shape (nFeatures, nFrames) with values in [0, 1].
    public func softMask(_ features: [[Float]]) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        // Find nearest neighbors and their distances
        let (neighbors, _) = findNeighborsWithDistances(features: features)

        // Compute mask based on local variance
        var result = [[Float]]()
        result.reserveCapacity(nFeatures)

        for f in 0..<nFeatures {
            var row = [Float](repeating: 0, count: nFrames)

            for j in 0..<nFrames {
                // Gather values from neighbors
                var values = [Float]()
                for neighborIdx in neighbors[j] {
                    values.append(features[f][neighborIdx])
                }

                // Compute median and MAD (Median Absolute Deviation)
                let median = computeMedian(values)
                let absDeviations = values.map { abs($0 - median) }
                let mad = computeMedian(absDeviations) + 1e-10

                // Current value's deviation
                let currentValue = features[f][j]
                let deviation = abs(currentValue - median)

                // Mask: 1.0 for values close to median, decreasing for outliers
                // Using sigmoid-like function
                let normalizedDeviation = deviation / mad
                let maskValue = 1.0 / (1.0 + normalizedDeviation)

                row[j] = maskValue
            }

            result.append(row)
        }

        return result
    }

    /// Apply a mask to a spectrogram.
    ///
    /// - Parameters:
    ///   - mask: Mask of shape (nFeatures, nFrames).
    ///   - features: Spectrogram to mask.
    /// - Returns: Masked spectrogram.
    public func apply(mask: [[Float]], to features: [[Float]]) -> [[Float]] {
        guard mask.count == features.count else { return features }
        guard !mask.isEmpty, mask[0].count == features[0].count else { return features }

        var result = [[Float]]()
        result.reserveCapacity(features.count)

        for (f, row) in features.enumerated() {
            var maskedRow = [Float](repeating: 0, count: row.count)
            for j in 0..<row.count {
                maskedRow[j] = row[j] * mask[f][j]
            }
            result.append(maskedRow)
        }

        return result
    }

    /// Decompose spectrogram into "repeating" and "non-repeating" components.
    ///
    /// Repeating elements are frames similar to their neighbors.
    /// Non-repeating elements are transients/outliers.
    ///
    /// - Parameter features: Input spectrogram.
    /// - Returns: Tuple of (repeating, nonRepeating) spectrograms.
    public func decompose(_ features: [[Float]]) -> (repeating: [[Float]], nonRepeating: [[Float]]) {
        let filtered = filter(features)
        let mask = softMask(features)

        // Repeating: filtered values (similar to neighbors)
        let repeating = filtered

        // Non-repeating: original minus filtered, scaled by inverse mask
        var nonRepeating = [[Float]]()
        nonRepeating.reserveCapacity(features.count)

        for (f, row) in features.enumerated() {
            var diffRow = [Float](repeating: 0, count: row.count)
            for j in 0..<row.count {
                // Residual weighted by how much of an outlier this frame is
                let inverseMask = 1.0 - mask[f][j]
                diffRow[j] = max(0, row[j] - filtered[f][j]) * (1.0 + inverseMask)
            }
            nonRepeating.append(diffRow)
        }

        return (repeating, nonRepeating)
    }

    // MARK: - Private Implementation

    /// Find k nearest neighbors for each frame.
    private func findNeighbors(features: [[Float]]) -> [[Int]] {
        let (neighbors, _) = findNeighborsWithDistances(features: features)
        return neighbors
    }

    /// Find k nearest neighbors and their distances.
    private func findNeighborsWithDistances(
        features: [[Float]]
    ) -> (neighbors: [[Int]], distances: [[Float]]) {
        let nFeatures = features.count
        let nFrames = features[0].count

        var allNeighbors = [[Int]]()
        var allDistances = [[Float]]()
        allNeighbors.reserveCapacity(nFrames)
        allDistances.reserveCapacity(nFrames)

        for j in 0..<nFrames {
            // Window bounds
            let windowStart = max(0, j - config.width)
            let windowEnd = min(nFrames, j + config.width + 1)

            // Compute distances to all frames in window
            var frameDists: [(idx: Int, dist: Float)] = []

            for i in windowStart..<windowEnd {
                if i == j { continue }  // Skip self

                let dist = computeDistance(
                    frame1: j, frame2: i,
                    features: features,
                    nFeatures: nFeatures, nFrames: nFrames
                )
                frameDists.append((i, dist))
            }

            // Sort by distance and take k nearest
            frameDists.sort { $0.dist < $1.dist }
            let k = min(config.k, frameDists.count)

            var neighbors = [Int]()
            var distances = [Float]()

            for i in 0..<k {
                neighbors.append(frameDists[i].idx)
                distances.append(frameDists[i].dist)
            }

            // Pad with self if not enough neighbors
            while neighbors.count < config.k {
                neighbors.append(j)
                distances.append(0)
            }

            allNeighbors.append(neighbors)
            allDistances.append(distances)
        }

        return (allNeighbors, allDistances)
    }

    /// Compute distance between two frames.
    private func computeDistance(
        frame1: Int, frame2: Int,
        features: [[Float]],
        nFeatures: Int, nFrames: Int
    ) -> Float {
        switch config.metric {
        case .euclidean:
            var sumSq: Float = 0
            for f in 0..<nFeatures {
                let diff = features[f][frame1] - features[f][frame2]
                sumSq += diff * diff
            }
            return sqrt(sumSq)

        case .manhattan:
            var sum: Float = 0
            for f in 0..<nFeatures {
                sum += abs(features[f][frame1] - features[f][frame2])
            }
            return sum

        case .cosine:
            var dot: Float = 0
            var norm1: Float = 0
            var norm2: Float = 0

            for f in 0..<nFeatures {
                let v1 = features[f][frame1]
                let v2 = features[f][frame2]
                dot += v1 * v2
                norm1 += v1 * v1
                norm2 += v2 * v2
            }

            let denom = sqrt(norm1) * sqrt(norm2)
            if denom > 0 {
                return 1.0 - (dot / denom)
            }
            return 1.0
        }
    }

    /// Aggregate values based on configuration.
    private func aggregate(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }

        switch config.aggregation {
        case .median:
            return computeMedian(values)

        case .mean:
            var sum: Float = 0
            for v in values { sum += v }
            return sum / Float(values.count)

        case .max:
            return values.max() ?? 0

        case .min:
            return values.min() ?? 0
        }
    }

    /// Compute median of values.
    private func computeMedian(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }

        let sorted = values.sorted()
        let mid = sorted.count / 2

        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }
}

// MARK: - Presets

extension NNFilter {
    /// Standard NN filter.
    public static var standard: NNFilter {
        NNFilter(config: NNFilterConfig(
            k: 5,
            width: 31,
            aggregation: .median
        ))
    }

    /// NN filter for percussive/harmonic separation.
    public static var hpss: NNFilter {
        NNFilter(config: NNFilterConfig(
            k: 7,
            width: 31,
            aggregation: .median
        ))
    }

    /// NN filter for denoising (larger neighborhood).
    public static var denoising: NNFilter {
        NNFilter(config: NNFilterConfig(
            k: 11,
            width: 51,
            aggregation: .median
        ))
    }

    /// NN filter for smoothing (smaller neighborhood, mean).
    public static var smoothing: NNFilter {
        NNFilter(config: NNFilterConfig(
            k: 3,
            width: 15,
            aggregation: .mean
        ))
    }
}
