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

    /// Pre-compute Savitzky-Golay-like filter coefficients for first derivative.
    ///
    /// The delta filter is: [-halfWidth, ..., -1, 0, 1, ..., halfWidth] / norm
    /// where norm = 2 * sum(n²) for n in 1...halfWidth
    private static func computeFilterCoefficients(width: Int) -> [Float] {
        let halfWidth = width / 2

        // Normalization factor: 2 * sum(n²)
        var norm: Float = 0
        for n in 1...halfWidth {
            norm += Float(n * n)
        }
        norm *= 2

        // Build filter: [-halfWidth, ..., -1, 0, 1, ..., halfWidth]
        var filter = [Float](repeating: 0, count: width)
        for n in 1...halfWidth {
            filter[halfWidth - n] = Float(-n) / norm  // Left side (negative)
            filter[halfWidth + n] = Float(n) / norm   // Right side (positive)
        }
        // Center element is 0 (already initialized)

        return filter
    }

    /// Optimized first-order delta using vectorized operations.
    ///
    /// This implementation uses:
    /// - vDSP_vsub for vectorized differences (x[t+n] - x[t-n])
    /// - Pre-computed weights
    /// - In-place accumulation
    ///
    /// The delta formula is: delta[t] = sum(n * (x[t+n] - x[t-n])) / norm
    /// We compute this efficiently by processing each offset n separately.
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

        // Pre-compute weights: n / norm for each offset
        var weights = [Float](repeating: 0, count: halfWidth)
        for n in 1...halfWidth {
            weights[n - 1] = Float(n) / norm
        }

        var deltaFeatures = [[Float]]()
        deltaFeatures.reserveCapacity(nFeatures)

        // Temporary buffers for padded row and differences
        let paddedLen = halfWidth + nFrames + halfWidth
        var padded = [Float](repeating: 0, count: paddedLen)
        var diff = [Float](repeating: 0, count: nFrames)

        // Process each feature row
        for f in 0..<nFeatures {
            let row = features[f]

            // Pad the row once - use memcpy for center data
            row.withUnsafeBufferPointer { rowPtr in
                padded.withUnsafeMutableBufferPointer { paddedPtr in
                    // Copy center data
                    memcpy(paddedPtr.baseAddress! + halfWidth, rowPtr.baseAddress!, nFrames * MemoryLayout<Float>.size)
                }
            }

            // Pad edges (small number of elements, loop is fine)
            for i in 0..<halfWidth {
                let leftIdx = getIndex(-halfWidth + i, count: nFrames, mode: mode)
                padded[i] = row[leftIdx]
                let rightIdx = getIndex(nFrames + i, count: nFrames, mode: mode)
                padded[halfWidth + nFrames + i] = row[rightIdx]
            }

            // Initialize delta row to zero
            var deltaRow = [Float](repeating: 0, count: nFrames)

            // Accumulate weighted differences for each offset
            padded.withUnsafeBufferPointer { paddedPtr in
                deltaRow.withUnsafeMutableBufferPointer { deltaPtr in
                    let paddedBase = paddedPtr.baseAddress!
                    let deltaBase = deltaPtr.baseAddress!

                    for n in 1...halfWidth {
                        let weight = weights[n - 1]
                        let leftOffset = halfWidth - n   // Points to x[t-n] for t=0
                        let rightOffset = halfWidth + n  // Points to x[t+n] for t=0

                        // Compute x[t+n] - x[t-n] for all t
                        diff.withUnsafeMutableBufferPointer { diffPtr in
                            vDSP_vsub(
                                paddedBase + leftOffset, 1,   // x[t-n]
                                paddedBase + rightOffset, 1,  // x[t+n]
                                diffPtr.baseAddress!, 1,
                                vDSP_Length(nFrames)
                            )

                            // Accumulate: delta += weight * diff
                            var w = weight
                            vDSP_vsma(
                                diffPtr.baseAddress!, 1,
                                &w,
                                deltaBase, 1,
                                deltaBase, 1,
                                vDSP_Length(nFrames)
                            )
                        }
                    }
                }
            }

            deltaFeatures.append(deltaRow)
        }

        return deltaFeatures
    }

    /// Pad a row of features according to the edge mode.
    private static func padRow(_ row: [Float], halfWidth: Int, mode: DeltaEdgeMode) -> [Float] {
        let nFrames = row.count
        var padded = [Float](repeating: 0, count: halfWidth + nFrames + halfWidth)

        // Copy center data
        for i in 0..<nFrames {
            padded[halfWidth + i] = row[i]
        }

        // Pad left edge
        for i in 0..<halfWidth {
            let idx = getIndex(-halfWidth + i, count: nFrames, mode: mode)
            padded[i] = row[idx]
        }

        // Pad right edge
        for i in 0..<halfWidth {
            let idx = getIndex(nFrames + i, count: nFrames, mode: mode)
            padded[halfWidth + nFrames + i] = row[idx]
        }

        return padded
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
