import Accelerate
import Foundation

/// Mode for handling edge frames when computing delta features.
public enum DeltaEdgeMode: Sendable {
    /// Interpolate at boundaries (librosa default).
    case interp
    /// Replicate edge values.
    case nearest
    /// Mirror at boundaries.
    case mirror
    /// Wrap around (circular).
    case wrap
}

/// Generalized delta (derivative) feature computation.
///
/// Computes local estimate of the derivative of a feature sequence using
/// a Savitzky-Golay-like filter. This is useful for computing MFCCs with
/// deltas, or any other feature where time derivatives are informative.
///
/// ## Example Usage
///
/// ```swift
/// // Compute first-order delta (velocity)
/// let delta1 = Delta.compute(mfccFeatures, width: 9)
///
/// // Compute second-order delta (acceleration)
/// let delta2 = Delta.compute(mfccFeatures, order: 2)
///
/// // Stack original features with deltas
/// let stacked = Delta.stack(mfccFeatures, orders: [1, 2])
/// ```
///
/// ## Formula
///
/// For order 1: `delta[t] = sum(n * (x[t+n] - x[t-n])) / (2 * sum(n²))` for n ∈ [1, width/2]
///
/// Higher orders are computed by iteratively applying the delta operator.
public struct Delta {
    private init() {}

    /// Compute delta features (first derivative estimate).
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width for delta computation (default: 9, must be odd).
    ///   - order: Order of the difference operator (default: 1).
    ///   - mode: Edge handling mode (default: interp).
    /// - Returns: Delta features of same shape.
    public static func compute(
        _ features: [[Float]],
        width: Int = 9,
        order: Int = 1,
        mode: DeltaEdgeMode = .interp
    ) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return [] }
        guard order >= 1 else { return features }

        // Apply delta operator iteratively for higher orders
        var result = features
        for _ in 0..<order {
            result = computeFirstOrder(result, width: width, mode: mode)
        }

        return result
    }

    /// Stack original features with their deltas.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - orders: Delta orders to compute and stack (default: [1, 2]).
    ///   - width: Context width for delta computation (default: 9).
    /// - Returns: Stacked features of shape (nFeatures * (1 + len(orders)), nFrames).
    public static func stack(
        _ features: [[Float]],
        orders: [Int] = [1, 2],
        width: Int = 9
    ) -> [[Float]] {
        guard !features.isEmpty else { return [] }

        var stacked = features

        for order in orders {
            let delta = compute(features, width: width, order: order)
            stacked.append(contentsOf: delta)
        }

        return stacked
    }

    // MARK: - Private Implementation

    private static func computeFirstOrder(
        _ features: [[Float]],
        width: Int,
        mode: DeltaEdgeMode
    ) -> [[Float]] {
        let nFeatures = features.count
        let nFrames = features[0].count
        let halfWidth = width / 2

        // Pre-compute normalization factor
        var norm: Float = 0
        for n in 1...halfWidth {
            norm += Float(n * n)
        }
        norm *= 2

        var deltaFeatures = [[Float]]()
        deltaFeatures.reserveCapacity(nFeatures)

        for f in 0..<nFeatures {
            var deltaRow = [Float](repeating: 0, count: nFrames)

            for t in 0..<nFrames {
                var sum: Float = 0

                for n in 1...halfWidth {
                    let leftIdx = getIndex(t - n, count: nFrames, mode: mode)
                    let rightIdx = getIndex(t + n, count: nFrames, mode: mode)
                    sum += Float(n) * (features[f][rightIdx] - features[f][leftIdx])
                }

                deltaRow[t] = sum / norm
            }

            deltaFeatures.append(deltaRow)
        }

        return deltaFeatures
    }

    private static func getIndex(_ idx: Int, count: Int, mode: DeltaEdgeMode) -> Int {
        if idx >= 0 && idx < count {
            return idx
        }

        switch mode {
        case .interp, .nearest:
            // Clamp to valid range
            return max(0, min(count - 1, idx))

        case .mirror:
            // Mirror at boundaries
            if idx < 0 {
                return min(count - 1, -idx)
            } else {
                return max(0, 2 * (count - 1) - idx)
            }

        case .wrap:
            // Circular wrap
            var wrapped = idx % count
            if wrapped < 0 {
                wrapped += count
            }
            return wrapped
        }
    }
}

// MARK: - Convenience Extensions

extension Delta {
    /// Compute delta-delta (second derivative) features.
    ///
    /// Convenience method equivalent to `compute(features, order: 2)`.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width (default: 9).
    /// - Returns: Delta-delta features of same shape.
    public static func deltaDelta(
        _ features: [[Float]],
        width: Int = 9
    ) -> [[Float]] {
        return compute(features, width: width, order: 2)
    }

    /// Compute delta and delta-delta in one call.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width (default: 9).
    /// - Returns: Tuple of (delta, delta-delta) features.
    public static func deltaAndDeltaDelta(
        _ features: [[Float]],
        width: Int = 9
    ) -> (delta: [[Float]], deltaDelta: [[Float]]) {
        let delta = compute(features, width: width, order: 1)
        let deltaDelta = compute(features, width: width, order: 2)
        return (delta, deltaDelta)
    }
}
