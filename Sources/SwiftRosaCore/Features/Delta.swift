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

    /// Compute delta features (derivative estimate).
    ///
    /// Matches librosa.feature.delta which uses scipy.signal.savgol_filter with deriv=order.
    /// For order=1, computes first derivative; for order=2, computes second derivative directly
    /// (not iteratively applying first derivative).
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width for delta computation (default: 9, must be odd).
    ///   - order: Order of the difference operator (default: 1). Order=2 computes 2nd derivative.
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

        if order == 1 {
            return computeFirstOrder(features, width: width, mode: mode)
        } else if order == 2 {
            return computeSecondOrder(features, width: width, mode: mode)
        } else {
            // For order > 2, fall back to iterative application (not exact but reasonable)
            var result = features
            for _ in 0..<order {
                result = computeFirstOrder(result, width: width, mode: mode)
            }
            return result
        }
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

    /// Optimized first-order delta using Savitzky-Golay filter.
    ///
    /// This implementation matches librosa's delta which uses scipy.signal.savgol_filter.
    /// For interior points, it applies the standard weighted difference formula.
    /// For edge points with mode=.interp, it fits a polynomial and evaluates its derivative.
    ///
    /// The delta formula is: delta[t] = sum(n * (x[t+n] - x[t-n])) / norm
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

        // Process each feature row
        for f in 0..<nFeatures {
            let row = features[f]
            var deltaRow = [Float](repeating: 0, count: nFrames)

            if mode == .interp {
                // Use polynomial interpolation at edges (matches librosa/scipy)
                computeDeltaWithInterp(row: row, deltaRow: &deltaRow, halfWidth: halfWidth, weights: weights)
            } else {
                // Use padding-based approach for other modes
                computeDeltaWithPadding(row: row, deltaRow: &deltaRow, halfWidth: halfWidth, weights: weights, mode: mode)
            }

            deltaFeatures.append(deltaRow)
        }

        return deltaFeatures
    }

    /// Compute delta with polynomial interpolation at edges (mode=.interp).
    /// Matches scipy.signal.savgol_filter with mode='interp', polyorder=1.
    ///
    /// Key insight: librosa.feature.delta uses polyorder=order (default 1), not polyorder=2.
    /// With polyorder=1, scipy fits a LINE to the edge window points.
    /// A line has CONSTANT derivative (its slope), so all edge positions get the same value.
    private static func computeDeltaWithInterp(
        row: [Float],
        deltaRow: inout [Float],
        halfWidth: Int,
        weights: [Float]
    ) {
        let nFrames = row.count
        let width = halfWidth * 2 + 1

        // For interior points, use standard formula
        for t in halfWidth..<(nFrames - halfWidth) {
            var delta: Float = 0
            for n in 1...halfWidth {
                delta += weights[n - 1] * (row[t + n] - row[t - n])
            }
            deltaRow[t] = delta
        }

        // For edge points, fit a LINEAR polynomial (polyorder=1) to the window
        // A linear fit has constant derivative (slope), so all edge positions get the same value
        // This matches librosa's behavior with polyorder=order=1

        // Left edge: fit line to first 'width' points and use slope for all left edge positions
        let leftFitPoints = min(width, nFrames)
        if leftFitPoints >= 2 {
            // Fit linear polynomial y = c0 + c1*x
            let leftCoeffs = fitPolynomial(Array(row.prefix(leftFitPoints)), order: 1)
            // The derivative of a line is just c1 (the slope) - constant everywhere
            let leftSlope = leftCoeffs.count > 1 ? leftCoeffs[1] : 0
            for t in 0..<halfWidth {
                deltaRow[t] = leftSlope
            }
        }

        // Right edge: fit line to last 'width' points and use slope for all right edge positions
        let rightFitPoints = min(width, nFrames)
        if rightFitPoints >= 2 {
            let rightData = Array(row.suffix(rightFitPoints))
            let rightCoeffs = fitPolynomial(rightData, order: 1)
            // The derivative of a line is just c1 (the slope) - constant everywhere
            let rightSlope = rightCoeffs.count > 1 ? rightCoeffs[1] : 0
            for t in (nFrames - halfWidth)..<nFrames {
                deltaRow[t] = rightSlope
            }
        }
    }

    /// Compute second-order derivative (acceleration).
    /// Uses Savitzky-Golay filter with deriv=2 to match librosa.
    private static func computeSecondOrder(
        _ features: [[Float]],
        width: Int,
        mode: DeltaEdgeMode
    ) -> [[Float]] {
        let nFeatures = features.count
        let nFrames = features[0].count
        let halfWidth = width / 2

        // Pre-compute second derivative coefficients
        // For polyorder=2, deriv=2, these match scipy.signal.savgol_coeffs
        let coeffs = computeSecondDerivCoeffs(width: width)

        var deltaFeatures = [[Float]]()
        deltaFeatures.reserveCapacity(nFeatures)

        for f in 0..<nFeatures {
            let row = features[f]
            var deltaRow = [Float](repeating: 0, count: nFrames)

            if mode == .interp {
                computeSecondDerivWithInterp(row: row, deltaRow: &deltaRow, halfWidth: halfWidth, coeffs: coeffs)
            } else {
                computeSecondDerivWithPadding(row: row, deltaRow: &deltaRow, halfWidth: halfWidth, coeffs: coeffs, mode: mode)
            }

            deltaFeatures.append(deltaRow)
        }

        return deltaFeatures
    }

    /// Compute Savitzky-Golay second derivative coefficients.
    /// These match scipy.signal.savgol_coeffs(width, polyorder=2, deriv=2).
    ///
    /// The derivation uses the Vandermonde matrix approach:
    /// 1. Build V = [1, x, x^2] for x in [-m, ..., m]
    /// 2. Compute H = (V^T V)^-1 V^T (the "hat" matrix)
    /// 3. 2nd derivative coefficients = 2 * H[2, :] (third row times 2)
    private static func computeSecondDerivCoeffs(width: Int) -> [Float] {
        let m = width / 2
        var coeffs = [Float](repeating: 0, count: width)

        // For V = [1, x, x^2], V^T V is a 3x3 matrix:
        // [n,      sum_k,    sum_k^2   ]
        // [sum_k,  sum_k^2,  sum_k^3   ]
        // [sum_k^2, sum_k^3, sum_k^4   ]
        // where sums are for k = -m to +m

        // Due to symmetry: sum_k = sum_k^3 = 0
        // So V^T V simplifies to:
        // [n,        0,        sum_k^2]
        // [0,        sum_k^2,  0      ]
        // [sum_k^2,  0,        sum_k^4]

        let n = Float(width)
        let mf = Float(m)

        // sum of k^2 for k=-m..m is 2 * sum(k^2 for k=1..m) = 2 * m(m+1)(2m+1)/6 = m(m+1)(2m+1)/3
        let sumK2 = mf * (mf + 1) * (2 * mf + 1) / 3
        // sum of k^4 for k=-m..m is 2 * sum(k^4 for k=1..m) = m(m+1)(2m+1)(3m^2+3m-1)/15
        let sumK4 = mf * (mf + 1) * (2 * mf + 1) * (3 * mf * mf + 3 * mf - 1) / 15

        // Invert the 3x3 block-diagonal-like matrix analytically
        // For the quadratic coefficient (3rd row of (V^T V)^-1):
        // It's the inverse of the 2x2 submatrix [[n, sumK2], [sumK2, sumK4]]
        let det = n * sumK4 - sumK2 * sumK2

        // The third row of the inverse is [sumK2/det, 0, n/det] (approximately)
        // Actually for the full inverse:
        // [[A, 0, B], [0, C, 0], [B, 0, D]]
        // where det = AD - B^2, A = n, D = sumK4, B = sumK2

        // Third row of inverse: [-sumK2/det, 0, n/det]

        // H = (V^T V)^-1 * V^T
        // Third row of H, times V^T's columns gives the coefficients

        // For each position k:
        // H[2, :] corresponds to the coefficient of x^2 in the quadratic fit
        // V^T column k is [1, k, k^2]
        // H[2, :] * V^T[:, k] = invRow3 . [1, k, k^2]

        // invRow3 = [-sumK2/det, 0, n/det]
        // coeff[k] = -sumK2/det * 1 + 0 * k + n/det * k^2
        //          = (n * k^2 - sumK2) / det

        // 2nd derivative = 2 * coeff[k]
        for i in 0..<width {
            let k = Float(i - m)
            let coeff = (n * k * k - sumK2) / det
            coeffs[i] = 2 * coeff
        }

        return coeffs
    }

    /// Compute second derivative with polynomial interpolation at edges.
    private static func computeSecondDerivWithInterp(
        row: [Float],
        deltaRow: inout [Float],
        halfWidth: Int,
        coeffs: [Float]
    ) {
        let nFrames = row.count
        let width = halfWidth * 2 + 1

        // For interior points, apply the savgol coefficients
        for t in halfWidth..<(nFrames - halfWidth) {
            var delta: Float = 0
            for k in 0..<width {
                delta += coeffs[k] * row[t - halfWidth + k]
            }
            deltaRow[t] = delta
        }

        // For edge points, fit a quadratic and evaluate 2nd derivative (which is constant)
        // 2nd derivative of ax^2 + bx + c is 2a

        // Left edge: fit quadratic to first 'width' points
        let leftFitPoints = min(width, nFrames)
        if leftFitPoints >= 3 {
            let leftCoeffs = fitPolynomial(Array(row.prefix(leftFitPoints)), order: 2)
            // Second derivative of quadratic is 2*a (where a is the x^2 coefficient)
            let leftSecondDeriv = leftCoeffs.count > 2 ? 2 * leftCoeffs[2] : 0
            for t in 0..<halfWidth {
                deltaRow[t] = leftSecondDeriv
            }
        }

        // Right edge: fit quadratic to last 'width' points
        let rightFitPoints = min(width, nFrames)
        if rightFitPoints >= 3 {
            let rightData = Array(row.suffix(rightFitPoints))
            let rightCoeffs = fitPolynomial(rightData, order: 2)
            let rightSecondDeriv = rightCoeffs.count > 2 ? 2 * rightCoeffs[2] : 0
            for t in (nFrames - halfWidth)..<nFrames {
                deltaRow[t] = rightSecondDeriv
            }
        }
    }

    /// Compute second derivative with padding for non-interp modes.
    private static func computeSecondDerivWithPadding(
        row: [Float],
        deltaRow: inout [Float],
        halfWidth: Int,
        coeffs: [Float],
        mode: DeltaEdgeMode
    ) {
        let nFrames = row.count
        let width = halfWidth * 2 + 1

        // Pad the row
        let paddedLen = halfWidth + nFrames + halfWidth
        var padded = [Float](repeating: 0, count: paddedLen)

        // Copy center data
        for i in 0..<nFrames {
            padded[halfWidth + i] = row[i]
        }

        // Pad edges
        for i in 0..<halfWidth {
            let leftIdx = getIndex(-halfWidth + i, count: nFrames, mode: mode)
            padded[i] = row[leftIdx]
            let rightIdx = getIndex(nFrames + i, count: nFrames, mode: mode)
            padded[halfWidth + nFrames + i] = row[rightIdx]
        }

        // Compute second derivative for all frames
        for t in 0..<nFrames {
            var delta: Float = 0
            for k in 0..<width {
                delta += coeffs[k] * padded[t + k]
            }
            deltaRow[t] = delta
        }
    }

    /// Compute delta with padding for non-interp modes.
    private static func computeDeltaWithPadding(
        row: [Float],
        deltaRow: inout [Float],
        halfWidth: Int,
        weights: [Float],
        mode: DeltaEdgeMode
    ) {
        let nFrames = row.count

        // Pad the row
        let paddedLen = halfWidth + nFrames + halfWidth
        var padded = [Float](repeating: 0, count: paddedLen)

        // Copy center data
        for i in 0..<nFrames {
            padded[halfWidth + i] = row[i]
        }

        // Pad edges
        for i in 0..<halfWidth {
            let leftIdx = getIndex(-halfWidth + i, count: nFrames, mode: mode)
            padded[i] = row[leftIdx]
            let rightIdx = getIndex(nFrames + i, count: nFrames, mode: mode)
            padded[halfWidth + nFrames + i] = row[rightIdx]
        }

        // Compute delta for all frames
        for t in 0..<nFrames {
            var delta: Float = 0
            for n in 1...halfWidth {
                let leftVal = padded[halfWidth + t - n]
                let rightVal = padded[halfWidth + t + n]
                delta += weights[n - 1] * (rightVal - leftVal)
            }
            deltaRow[t] = delta
        }
    }

    /// Fit a polynomial of given order to data points using least squares.
    /// Returns coefficients [c0, c1, c2, ...] where y = c0 + c1*x + c2*x^2 + ...
    private static func fitPolynomial(_ data: [Float], order: Int) -> [Float] {
        let n = data.count
        guard n > order else { return [Float](repeating: 0, count: order + 1) }

        // Build Vandermonde matrix and solve normal equations
        // X^T X c = X^T y
        let nCoeffs = order + 1

        // Build X^T X (symmetric, so only compute upper triangle)
        var XTX = [[Float]](repeating: [Float](repeating: 0, count: nCoeffs), count: nCoeffs)
        for i in 0..<nCoeffs {
            for j in i..<nCoeffs {
                var sum: Float = 0
                for k in 0..<n {
                    let x = Float(k)
                    sum += pow(x, Float(i)) * pow(x, Float(j))
                }
                XTX[i][j] = sum
                XTX[j][i] = sum
            }
        }

        // Build X^T y
        var XTy = [Float](repeating: 0, count: nCoeffs)
        for i in 0..<nCoeffs {
            var sum: Float = 0
            for k in 0..<n {
                let x = Float(k)
                sum += pow(x, Float(i)) * data[k]
            }
            XTy[i] = sum
        }

        // Solve using Gaussian elimination with partial pivoting
        return solveLinearSystem(XTX, XTy)
    }

    /// Evaluate the derivative of a polynomial at point x.
    /// coeffs = [c0, c1, c2, ...] → derivative = c1 + 2*c2*x + 3*c3*x^2 + ...
    private static func evaluatePolynomialDerivative(_ coeffs: [Float], at x: Float) -> Float {
        guard coeffs.count > 1 else { return 0 }

        var result: Float = 0
        var xPower: Float = 1
        for i in 1..<coeffs.count {
            result += Float(i) * coeffs[i] * xPower
            xPower *= x
        }
        return result
    }

    /// Solve linear system Ax = b using Gaussian elimination.
    private static func solveLinearSystem(_ A: [[Float]], _ b: [Float]) -> [Float] {
        let n = b.count
        guard n > 0 else { return [] }

        // Create augmented matrix
        var aug = [[Float]]()
        for i in 0..<n {
            var row = A[i]
            row.append(b[i])
            aug.append(row)
        }

        // Forward elimination with partial pivoting
        for col in 0..<n {
            // Find pivot
            var maxRow = col
            var maxVal = abs(aug[col][col])
            for row in (col + 1)..<n {
                if abs(aug[row][col]) > maxVal {
                    maxVal = abs(aug[row][col])
                    maxRow = row
                }
            }

            // Swap rows
            if maxRow != col {
                let temp = aug[col]
                aug[col] = aug[maxRow]
                aug[maxRow] = temp
            }

            // Check for singularity
            guard abs(aug[col][col]) > 1e-10 else {
                return [Float](repeating: 0, count: n)
            }

            // Eliminate column
            for row in (col + 1)..<n {
                let factor = aug[row][col] / aug[col][col]
                for j in col..<(n + 1) {
                    aug[row][j] -= factor * aug[col][j]
                }
            }
        }

        // Back substitution
        var x = [Float](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = aug[i][n]
            for j in (i + 1)..<n {
                sum -= aug[i][j] * x[j]
            }
            x[i] = sum / aug[i][i]
        }

        return x
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
