import Accelerate
import Foundation

/// Aggregation method for synchronizing features.
public enum SyncAggregation: Sendable {
    /// Arithmetic mean.
    case mean
    /// Median value.
    case median
    /// Maximum value.
    case max
    /// Minimum value.
    case min
    /// Sum of values.
    case sum
    /// First value in segment.
    case first
    /// Last value in segment.
    case last
}

/// Feature synchronization utilities.
///
/// Aggregates frame-level features to segment boundaries (e.g., beat-synchronous features).
/// This is useful for aligning features to musical beats or other temporal boundaries.
///
/// ## Example Usage
///
/// ```swift
/// // Sync MFCC features to beat times
/// let beatFrames = SyncFeatures.timesToFrames(beatTimes, sampleRate: 22050, hopLength: 512)
/// let syncedMFCC = SyncFeatures.sync(mfccFeatures, boundaries: beatFrames)
///
/// // Use median aggregation for robustness
/// let syncedChroma = SyncFeatures.sync(chromaFeatures, boundaries: beatFrames, aggregation: .median)
/// ```
public struct SyncFeatures {
    private init() {}

    /// Synchronize features to segment boundaries.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - boundaries: Frame indices of segment boundaries (sorted).
    ///   - aggregation: How to aggregate frames within segments (default: mean).
    /// - Returns: Synchronized features of shape (nFeatures, nSegments).
    public static func sync(
        _ features: [[Float]],
        boundaries: [Int],
        aggregation: SyncAggregation = .mean
    ) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return [] }
        guard !boundaries.isEmpty else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        // Ensure boundaries are sorted and include start/end
        var sortedBoundaries = boundaries.sorted()

        // Add start boundary if not present
        if sortedBoundaries.first != 0 {
            sortedBoundaries.insert(0, at: 0)
        }

        // Add end boundary if not present
        if sortedBoundaries.last != nFrames {
            sortedBoundaries.append(nFrames)
        }

        let nSegments = sortedBoundaries.count - 1
        guard nSegments > 0 else { return features }

        var result = [[Float]]()
        result.reserveCapacity(nFeatures)

        for f in 0..<nFeatures {
            var row = [Float](repeating: 0, count: nSegments)

            for seg in 0..<nSegments {
                let start = max(0, sortedBoundaries[seg])
                let end = min(nFrames, sortedBoundaries[seg + 1])

                guard start < end else {
                    row[seg] = 0
                    continue
                }

                let segmentValues = Array(features[f][start..<end])
                row[seg] = aggregate(segmentValues, method: aggregation)
            }

            result.append(row)
        }

        return result
    }

    /// Synchronize a 1D feature array to segment boundaries.
    ///
    /// - Parameters:
    ///   - feature: Feature array of length nFrames.
    ///   - boundaries: Frame indices of segment boundaries (sorted).
    ///   - aggregation: How to aggregate frames within segments (default: mean).
    /// - Returns: Synchronized feature of length nSegments.
    public static func sync(
        _ feature: [Float],
        boundaries: [Int],
        aggregation: SyncAggregation = .mean
    ) -> [Float] {
        let result = sync([feature], boundaries: boundaries, aggregation: aggregation)
        return result.first ?? []
    }

    /// Convert time values to frame indices.
    ///
    /// - Parameters:
    ///   - times: Time values in seconds.
    ///   - sampleRate: Sample rate in Hz.
    ///   - hopLength: Hop length used for frame computation.
    /// - Returns: Frame indices corresponding to times.
    public static func timesToFrames(
        _ times: [Float],
        sampleRate: Float,
        hopLength: Int
    ) -> [Int] {
        return times.map { time in
            Int(time * sampleRate / Float(hopLength))
        }
    }

    /// Convert frame indices to time values.
    ///
    /// - Parameters:
    ///   - frames: Frame indices.
    ///   - sampleRate: Sample rate in Hz.
    ///   - hopLength: Hop length used for frame computation.
    /// - Returns: Time values in seconds.
    public static func framesToTimes(
        _ frames: [Int],
        sampleRate: Float,
        hopLength: Int
    ) -> [Float] {
        return frames.map { frame in
            Float(frame) * Float(hopLength) / sampleRate
        }
    }

    /// Convert sample indices to frame indices.
    ///
    /// - Parameters:
    ///   - samples: Sample indices.
    ///   - hopLength: Hop length used for frame computation.
    /// - Returns: Frame indices.
    public static func samplesToFrames(_ samples: [Int], hopLength: Int) -> [Int] {
        return samples.map { $0 / hopLength }
    }

    /// Convert frame indices to sample indices.
    ///
    /// - Parameters:
    ///   - frames: Frame indices.
    ///   - hopLength: Hop length used for frame computation.
    /// - Returns: Sample indices (frame starts).
    public static func framesToSamples(_ frames: [Int], hopLength: Int) -> [Int] {
        return frames.map { $0 * hopLength }
    }

    // MARK: - Private Implementation

    private static func aggregate(_ values: [Float], method: SyncAggregation) -> Float {
        guard !values.isEmpty else { return 0 }

        switch method {
        case .mean:
            var result: Float = 0
            vDSP_meanv(values, 1, &result, vDSP_Length(values.count))
            return result

        case .median:
            let sorted = values.sorted()
            let mid = sorted.count / 2
            if sorted.count % 2 == 0 {
                return (sorted[mid - 1] + sorted[mid]) / 2
            } else {
                return sorted[mid]
            }

        case .max:
            var result: Float = 0
            vDSP_maxv(values, 1, &result, vDSP_Length(values.count))
            return result

        case .min:
            var result: Float = 0
            vDSP_minv(values, 1, &result, vDSP_Length(values.count))
            return result

        case .sum:
            var result: Float = 0
            vDSP_sve(values, 1, &result, vDSP_Length(values.count))
            return result

        case .first:
            return values.first ?? 0

        case .last:
            return values.last ?? 0
        }
    }
}
