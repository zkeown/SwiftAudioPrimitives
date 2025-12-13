import Accelerate
import Foundation

/// Distance metric for recurrence computation.
public enum RQAMetric: Sendable {
    /// Euclidean distance.
    case euclidean
    /// Cosine distance (1 - cosine similarity).
    case cosine
    /// Chebyshev (max) distance.
    case chebyshev
}

/// Configuration for Recurrence Quantification Analysis.
public struct RQAConfig: Sendable {
    /// Distance metric.
    public let metric: RQAMetric

    /// Maximum recurrence size for sparse representation.
    /// Set to nil for full recurrence matrix.
    public let maxSize: Int?

    /// Number of nearest neighbors (k) for sparse RQA.
    public let k: Int?

    /// Width for bandwidth constraint (only recurrences within this diagonal band).
    /// Set to nil for no bandwidth constraint.
    public let bandwidth: Int?

    /// Whether to make the recurrence matrix symmetric.
    public let symmetric: Bool

    /// Whether to exclude self-recurrence (main diagonal).
    public let excludeSelf: Bool

    /// Create RQA configuration.
    ///
    /// - Parameters:
    ///   - metric: Distance metric (default: euclidean).
    ///   - maxSize: Max recurrence size for sparse output (default: nil for full).
    ///   - k: Number of nearest neighbors (default: nil).
    ///   - bandwidth: Diagonal bandwidth constraint (default: nil).
    ///   - symmetric: Make matrix symmetric (default: true).
    ///   - excludeSelf: Exclude self-recurrence (default: false).
    public init(
        metric: RQAMetric = .euclidean,
        maxSize: Int? = nil,
        k: Int? = nil,
        bandwidth: Int? = nil,
        symmetric: Bool = true,
        excludeSelf: Bool = false
    ) {
        self.metric = metric
        self.maxSize = maxSize
        self.k = k
        self.bandwidth = bandwidth
        self.symmetric = symmetric
        self.excludeSelf = excludeSelf
    }
}

/// Result of Recurrence Quantification Analysis.
public struct RQAResult: Sendable {
    /// Recurrence matrix (nFrames x nFrames).
    /// Values are distances; use threshold to convert to binary recurrence plot.
    public let recurrenceMatrix: [[Float]]

    /// Sparse representation as (i, j, distance) tuples.
    public let sparse: [(Int, Int, Float)]?

    /// Recurrence rate (percentage of recurrent points).
    public let recurrenceRate: Float

    /// Determinism (percentage of recurrent points forming diagonal lines).
    public let determinism: Float

    /// Average diagonal line length.
    public let averageDiagonalLength: Float

    /// Laminarity (percentage of recurrent points forming vertical lines).
    public let laminarity: Float

    /// Average vertical line length.
    public let averageVerticalLength: Float
}

/// Recurrence Quantification Analysis.
///
/// RQA constructs a recurrence matrix showing when states in a time series
/// revisit previous states. This is useful for detecting patterns, periodicity,
/// and structure in audio and other signals.
///
/// ## Example Usage
///
/// ```swift
/// let rqa = RQA()
///
/// // Compute recurrence matrix from features
/// let features: [[Float]] = mfccCoefficients  // (nFeatures, nFrames)
/// let result = rqa.compute(features)
///
/// // Access recurrence matrix
/// let recMatrix = result.recurrenceMatrix  // (nFrames, nFrames)
///
/// // Get RQA statistics
/// print("Recurrence rate: \(result.recurrenceRate)")
/// print("Determinism: \(result.determinism)")
/// ```
///
/// - Note: Matches `librosa.segment.recurrence_matrix` and RQA analysis behavior.
public struct RQA: Sendable {
    /// Configuration.
    public let config: RQAConfig

    /// Create an RQA processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: RQAConfig = RQAConfig()) {
        self.config = config
    }

    /// Compute recurrence matrix and RQA statistics.
    ///
    /// - Parameter features: Feature matrix (nFeatures, nFrames).
    /// - Returns: RQA result with recurrence matrix and statistics.
    public func compute(_ features: [[Float]]) -> RQAResult {
        guard !features.isEmpty && !features[0].isEmpty else {
            return RQAResult(
                recurrenceMatrix: [],
                sparse: nil,
                recurrenceRate: 0,
                determinism: 0,
                averageDiagonalLength: 0,
                laminarity: 0,
                averageVerticalLength: 0
            )
        }

        let nFrames = features[0].count

        // Transpose to (nFrames, nFeatures) for efficient row access
        let featuresT = transposeFeatures(features)

        // Compute distance matrix
        var distMatrix = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFrames)

        for i in 0..<nFrames {
            featuresT[i].withUnsafeBufferPointer { iPtr in
                for j in (config.symmetric ? i : 0)..<nFrames {
                    // Check bandwidth constraint
                    if let bw = config.bandwidth, abs(i - j) > bw {
                        distMatrix[i][j] = Float.infinity
                        if config.symmetric {
                            distMatrix[j][i] = Float.infinity
                        }
                        continue
                    }

                    // Check self-exclusion
                    if config.excludeSelf && i == j {
                        distMatrix[i][j] = Float.infinity
                        continue
                    }

                    featuresT[j].withUnsafeBufferPointer { jPtr in
                        let dist = computeDistance(
                            xPtr: iPtr.baseAddress!,
                            yPtr: jPtr.baseAddress!,
                            nFeatures: features.count
                        )
                        distMatrix[i][j] = dist
                        if config.symmetric && i != j {
                            distMatrix[j][i] = dist
                        }
                    }
                }
            }
        }

        // Apply k-nearest neighbors if specified
        if let k = config.k {
            distMatrix = applyKNN(distMatrix, k: k)
        }

        // Compute sparse representation if maxSize specified
        var sparse: [(Int, Int, Float)]?
        if let maxSize = config.maxSize {
            sparse = computeSparse(distMatrix, maxSize: maxSize)
        }

        // Compute RQA statistics
        let stats = computeStatistics(distMatrix)

        return RQAResult(
            recurrenceMatrix: distMatrix,
            sparse: sparse,
            recurrenceRate: stats.recurrenceRate,
            determinism: stats.determinism,
            averageDiagonalLength: stats.avgDiagLength,
            laminarity: stats.laminarity,
            averageVerticalLength: stats.avgVertLength
        )
    }

    /// Compute only the recurrence matrix (without statistics).
    ///
    /// - Parameter features: Feature matrix (nFeatures, nFrames).
    /// - Returns: Recurrence matrix (nFrames x nFrames).
    public func recurrenceMatrix(_ features: [[Float]]) -> [[Float]] {
        return compute(features).recurrenceMatrix
    }

    /// Convert distance matrix to binary recurrence plot using threshold.
    ///
    /// - Parameters:
    ///   - distMatrix: Distance matrix.
    ///   - threshold: Distance threshold for recurrence.
    /// - Returns: Binary recurrence matrix (1.0 = recurrent, 0.0 = not).
    public static func binaryRecurrence(_ distMatrix: [[Float]], threshold: Float) -> [[Float]] {
        return distMatrix.map { row in
            row.map { $0 <= threshold ? 1.0 : 0.0 }
        }
    }

    // MARK: - Private Implementation

    /// Transpose feature matrix from (nFeatures, nFrames) to (nFrames, nFeatures).
    private func transposeFeatures(_ matrix: [[Float]]) -> [[Float]] {
        guard !matrix.isEmpty && !matrix[0].isEmpty else { return [] }
        let nFeatures = matrix.count
        let nFrames = matrix[0].count

        var result = [[Float]](repeating: [Float](repeating: 0, count: nFeatures), count: nFrames)
        for f in 0..<nFeatures {
            for t in 0..<nFrames {
                result[t][f] = matrix[f][t]
            }
        }
        return result
    }

    /// Compute distance between two feature vectors.
    private func computeDistance(
        xPtr: UnsafePointer<Float>,
        yPtr: UnsafePointer<Float>,
        nFeatures: Int
    ) -> Float {
        switch config.metric {
        case .euclidean:
            var distSq: Float = 0
            vDSP_distancesq(xPtr, 1, yPtr, 1, &distSq, vDSP_Length(nFeatures))
            return sqrt(distSq)

        case .cosine:
            var dotProduct: Float = 0
            var xNormSq: Float = 0
            var yNormSq: Float = 0

            vDSP_dotpr(xPtr, 1, yPtr, 1, &dotProduct, vDSP_Length(nFeatures))
            vDSP_dotpr(xPtr, 1, xPtr, 1, &xNormSq, vDSP_Length(nFeatures))
            vDSP_dotpr(yPtr, 1, yPtr, 1, &yNormSq, vDSP_Length(nFeatures))

            let denom = sqrt(xNormSq) * sqrt(yNormSq)
            if denom > 0 {
                return 1.0 - dotProduct / denom
            }
            return 1.0

        case .chebyshev:
            var maxDiff: Float = 0
            for f in 0..<nFeatures {
                let diff = abs(xPtr[f] - yPtr[f])
                if diff > maxDiff {
                    maxDiff = diff
                }
            }
            return maxDiff
        }
    }

    /// Apply k-nearest neighbors to distance matrix.
    private func applyKNN(_ distMatrix: [[Float]], k: Int) -> [[Float]] {
        let n = distMatrix.count
        var result = [[Float]](repeating: [Float](repeating: Float.infinity, count: n), count: n)

        for i in 0..<n {
            // Get sorted indices by distance
            let sortedIndices = distMatrix[i].enumerated()
                .sorted { $0.element < $1.element }
                .prefix(k)
                .map { $0.offset }

            // Keep only k nearest neighbors
            for j in sortedIndices {
                result[i][j] = distMatrix[i][j]
                if config.symmetric {
                    result[j][i] = distMatrix[j][i]
                }
            }
        }

        return result
    }

    /// Compute sparse representation.
    private func computeSparse(_ distMatrix: [[Float]], maxSize: Int) -> [(Int, Int, Float)] {
        var entries: [(Int, Int, Float)] = []
        let n = distMatrix.count

        for i in 0..<n {
            for j in 0..<n {
                let dist = distMatrix[i][j]
                if dist.isFinite {
                    entries.append((i, j, dist))
                }
            }
        }

        // Sort by distance and take top maxSize
        entries.sort { $0.2 < $1.2 }
        return Array(entries.prefix(maxSize))
    }

    /// Compute RQA statistics.
    private func computeStatistics(_ distMatrix: [[Float]]) -> (
        recurrenceRate: Float,
        determinism: Float,
        avgDiagLength: Float,
        laminarity: Float,
        avgVertLength: Float
    ) {
        let n = distMatrix.count
        guard n > 0 else {
            return (0, 0, 0, 0, 0)
        }

        // Compute adaptive threshold (median distance)
        var allDistances: [Float] = []
        for i in 0..<n {
            for j in 0..<n {
                if distMatrix[i][j].isFinite {
                    allDistances.append(distMatrix[i][j])
                }
            }
        }

        guard !allDistances.isEmpty else {
            return (0, 0, 0, 0, 0)
        }

        allDistances.sort()
        let medianThreshold = allDistances[allDistances.count / 2]

        // Convert to binary using median threshold
        let binaryMatrix = Self.binaryRecurrence(distMatrix, threshold: medianThreshold)

        // Count recurrent points
        var recurrentCount: Float = 0
        let totalPairs = Float(n * n)

        for i in 0..<n {
            for j in 0..<n {
                recurrentCount += binaryMatrix[i][j]
            }
        }

        let recurrenceRate = recurrentCount / totalPairs

        // Compute diagonal line statistics (determinism)
        var diagLengths: [Int] = []

        // Check all diagonals (excluding main if self-excluded)
        let startDiag = config.excludeSelf ? 1 : 0
        for d in startDiag..<n {
            // Upper diagonal
            var length = 0
            for k in 0..<(n - d) {
                if binaryMatrix[k][k + d] == 1.0 {
                    length += 1
                } else if length > 0 {
                    diagLengths.append(length)
                    length = 0
                }
            }
            if length > 0 {
                diagLengths.append(length)
            }

            // Lower diagonal (skip d=0 to avoid double counting)
            if d > 0 {
                length = 0
                for k in 0..<(n - d) {
                    if binaryMatrix[k + d][k] == 1.0 {
                        length += 1
                    } else if length > 0 {
                        diagLengths.append(length)
                        length = 0
                    }
                }
                if length > 0 {
                    diagLengths.append(length)
                }
            }
        }

        // Determinism: percentage of recurrent points in diagonal structures
        let diagPointsInLines = diagLengths.filter { $0 >= 2 }.reduce(0, +)
        let determinism = recurrentCount > 0 ? Float(diagPointsInLines) / recurrentCount : 0

        // Average diagonal line length
        let linesLongerThan2 = diagLengths.filter { $0 >= 2 }
        let avgDiagLength = linesLongerThan2.isEmpty ? 0 : Float(linesLongerThan2.reduce(0, +)) / Float(linesLongerThan2.count)

        // Compute vertical line statistics (laminarity)
        var vertLengths: [Int] = []
        for j in 0..<n {
            var length = 0
            for i in 0..<n {
                if binaryMatrix[i][j] == 1.0 {
                    length += 1
                } else if length > 0 {
                    vertLengths.append(length)
                    length = 0
                }
            }
            if length > 0 {
                vertLengths.append(length)
            }
        }

        // Laminarity: percentage of recurrent points in vertical structures
        let vertPointsInLines = vertLengths.filter { $0 >= 2 }.reduce(0, +)
        let laminarity = recurrentCount > 0 ? Float(vertPointsInLines) / recurrentCount : 0

        // Average vertical line length
        let vertLinesLongerThan2 = vertLengths.filter { $0 >= 2 }
        let avgVertLength = vertLinesLongerThan2.isEmpty ? 0 : Float(vertLinesLongerThan2.reduce(0, +)) / Float(vertLinesLongerThan2.count)

        return (recurrenceRate, determinism, avgDiagLength, laminarity, avgVertLength)
    }
}

// MARK: - Presets

extension RQA {
    /// Standard configuration for feature-based recurrence analysis.
    public static var standard: RQA {
        RQA(config: RQAConfig(
            metric: .euclidean,
            maxSize: nil,
            k: nil,
            bandwidth: nil,
            symmetric: true,
            excludeSelf: true
        ))
    }

    /// Configuration for sparse recurrence with limited neighbors.
    public static func sparse(k: Int = 10) -> RQA {
        RQA(config: RQAConfig(
            metric: .euclidean,
            maxSize: nil,
            k: k,
            bandwidth: nil,
            symmetric: true,
            excludeSelf: true
        ))
    }

    /// Configuration with bandwidth constraint (local recurrence only).
    public static func local(bandwidth: Int) -> RQA {
        RQA(config: RQAConfig(
            metric: .euclidean,
            maxSize: nil,
            k: nil,
            bandwidth: bandwidth,
            symmetric: true,
            excludeSelf: true
        ))
    }

    /// Configuration optimized for music structure analysis.
    public static var music: RQA {
        RQA(config: RQAConfig(
            metric: .cosine,
            maxSize: nil,
            k: nil,
            bandwidth: nil,
            symmetric: true,
            excludeSelf: true
        ))
    }
}
