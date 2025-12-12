import Accelerate
import Foundation

/// Distance metric for DTW.
public enum DTWDistanceMetric: Sendable {
    /// Euclidean distance.
    case euclidean
    /// Manhattan (L1) distance.
    case manhattan
    /// Cosine distance (1 - cosine similarity).
    case cosine
}

/// Window constraint for DTW.
public enum DTWWindow: Sendable {
    /// No constraint (full matrix).
    case none
    /// Sakoe-Chiba band.
    case sakoeChiba
    /// Itakura parallelogram.
    case itakura
}

/// Configuration for DTW.
public struct DTWConfig: Sendable {
    /// Distance metric.
    public let metric: DTWDistanceMetric

    /// Window constraint type.
    public let windowType: DTWWindow

    /// Window size (for Sakoe-Chiba) or slope constraint (for Itakura).
    public let windowParam: Int?

    /// Create DTW configuration.
    ///
    /// - Parameters:
    ///   - metric: Distance metric (default: euclidean).
    ///   - windowType: Window constraint (default: none).
    ///   - windowParam: Window size or slope (default: nil).
    public init(
        metric: DTWDistanceMetric = .euclidean,
        windowType: DTWWindow = .none,
        windowParam: Int? = nil
    ) {
        self.metric = metric
        self.windowType = windowType
        self.windowParam = windowParam
    }
}

/// DTW alignment result.
public struct DTWResult: Sendable {
    /// Total alignment distance.
    public let distance: Float

    /// Normalized distance (distance / path length).
    public let normalizedDistance: Float

    /// Alignment path as (i, j) pairs.
    public let path: [(Int, Int)]

    /// Optional cost matrix (for visualization).
    public let costMatrix: [[Float]]?
}

/// Subsequence DTW result.
public struct SubsequenceDTWResult: Sendable {
    /// Distance of best alignment.
    public let distance: Float

    /// Start index in reference where query best aligns.
    public let startIndex: Int

    /// End index in reference.
    public let endIndex: Int

    /// Alignment path.
    public let path: [(Int, Int)]
}

/// Dynamic Time Warping.
///
/// DTW finds the optimal alignment between two sequences that may vary in
/// speed or timing. Commonly used for speech recognition, music analysis,
/// and time series classification.
///
/// ## Example Usage
///
/// ```swift
/// let dtw = DTW()
///
/// // Compute alignment
/// let result = dtw.align(sequence1, sequence2)
/// print("Distance: \(result.distance)")
/// print("Path: \(result.path)")
///
/// // Just compute distance (more efficient)
/// let distance = dtw.distance(sequence1, sequence2)
///
/// // Find query in longer reference
/// let subResult = dtw.subsequenceAlign(query: short, reference: long)
/// ```
public struct DTW: Sendable {
    /// Configuration.
    public let config: DTWConfig

    /// Create a DTW processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: DTWConfig = DTWConfig()) {
        self.config = config
    }

    /// Compute DTW distance and alignment path.
    ///
    /// - Parameters:
    ///   - x: First sequence (nFeatures, nFramesX).
    ///   - y: Second sequence (nFeatures, nFramesY).
    ///   - includeCostMatrix: Include cost matrix in result (default: false).
    /// - Returns: DTW result with distance and path.
    public func align(
        _ x: [[Float]],
        _ y: [[Float]],
        includeCostMatrix: Bool = false
    ) -> DTWResult {
        guard !x.isEmpty && !y.isEmpty && !x[0].isEmpty && !y[0].isEmpty else {
            return DTWResult(distance: Float.infinity, normalizedDistance: Float.infinity, path: [], costMatrix: nil)
        }

        let nX = x[0].count
        let nY = y[0].count

        // Compute distance matrix
        let distMatrix = computeDistanceMatrix(x, y)

        // Compute accumulated cost matrix
        var cost = [[Float]](repeating: [Float](repeating: Float.infinity, count: nY), count: nX)

        // Initialize first cell
        cost[0][0] = distMatrix[0][0]

        // Initialize first row
        for j in 1..<nY {
            if isInWindow(0, j, nX, nY) {
                cost[0][j] = cost[0][j - 1] + distMatrix[0][j]
            }
        }

        // Initialize first column
        for i in 1..<nX {
            if isInWindow(i, 0, nX, nY) {
                cost[i][0] = cost[i - 1][0] + distMatrix[i][0]
            }
        }

        // Fill cost matrix
        for i in 1..<nX {
            for j in 1..<nY {
                if isInWindow(i, j, nX, nY) {
                    let minPrev = min(cost[i - 1][j], min(cost[i][j - 1], cost[i - 1][j - 1]))
                    cost[i][j] = distMatrix[i][j] + minPrev
                }
            }
        }

        // Backtrack to find path
        let path = backtrack(cost)

        let totalDistance = cost[nX - 1][nY - 1]
        let normalizedDistance = totalDistance / Float(path.count)

        return DTWResult(
            distance: totalDistance,
            normalizedDistance: normalizedDistance,
            path: path,
            costMatrix: includeCostMatrix ? cost : nil
        )
    }

    /// Compute DTW distance only (more memory efficient).
    ///
    /// - Parameters:
    ///   - x: First sequence.
    ///   - y: Second sequence.
    /// - Returns: DTW distance.
    public func distance(
        _ x: [[Float]],
        _ y: [[Float]]
    ) -> Float {
        guard !x.isEmpty && !y.isEmpty && !x[0].isEmpty && !y[0].isEmpty else {
            return Float.infinity
        }

        let nX = x[0].count
        let nY = y[0].count

        // Use only two rows for memory efficiency
        var prevRow = [Float](repeating: Float.infinity, count: nY)
        var currRow = [Float](repeating: Float.infinity, count: nY)

        // Initialize first row
        prevRow[0] = pointDistance(x, y, 0, 0)
        for j in 1..<nY {
            if isInWindow(0, j, nX, nY) {
                prevRow[j] = prevRow[j - 1] + pointDistance(x, y, 0, j)
            }
        }

        // Fill remaining rows
        for i in 1..<nX {
            currRow[0] = isInWindow(i, 0, nX, nY) ? prevRow[0] + pointDistance(x, y, i, 0) : Float.infinity

            for j in 1..<nY {
                if isInWindow(i, j, nX, nY) {
                    let minPrev = min(prevRow[j], min(currRow[j - 1], prevRow[j - 1]))
                    currRow[j] = pointDistance(x, y, i, j) + minPrev
                } else {
                    currRow[j] = Float.infinity
                }
            }

            swap(&prevRow, &currRow)
        }

        return prevRow[nY - 1]
    }

    /// Subsequence DTW: find best alignment of query within reference.
    ///
    /// - Parameters:
    ///   - query: Query sequence (shorter).
    ///   - reference: Reference sequence (longer).
    /// - Returns: Subsequence DTW result.
    public func subsequenceAlign(
        query: [[Float]],
        reference: [[Float]]
    ) -> SubsequenceDTWResult {
        guard !query.isEmpty && !reference.isEmpty && !query[0].isEmpty && !reference[0].isEmpty else {
            return SubsequenceDTWResult(distance: Float.infinity, startIndex: 0, endIndex: 0, path: [])
        }

        let nQ = query[0].count
        let nR = reference[0].count

        // Compute distance matrix
        var cost = [[Float]](repeating: [Float](repeating: Float.infinity, count: nR), count: nQ)

        // Subsequence DTW: allow starting anywhere in reference (first row initialized differently)
        for j in 0..<nR {
            cost[0][j] = pointDistance(query, reference, 0, j)
        }

        // Fill cost matrix (standard DTW within query dimension)
        for i in 1..<nQ {
            for j in 1..<nR {
                let minPrev = min(cost[i - 1][j], min(cost[i][j - 1], cost[i - 1][j - 1]))
                cost[i][j] = pointDistance(query, reference, i, j) + minPrev
            }
        }

        // Find minimum in last row (where query ends)
        var bestEnd = 0
        var bestDistance = cost[nQ - 1][0]
        for j in 1..<nR {
            if cost[nQ - 1][j] < bestDistance {
                bestDistance = cost[nQ - 1][j]
                bestEnd = j
            }
        }

        // Backtrack to find path and start
        var path = [(Int, Int)]()
        var i = nQ - 1
        var j = bestEnd

        while i > 0 {
            path.append((i, j))

            let diagonal = j > 0 ? cost[i - 1][j - 1] : Float.infinity
            let up = cost[i - 1][j]
            let left = j > 0 ? cost[i][j - 1] : Float.infinity

            let minCost = min(diagonal, min(up, left))

            if minCost == diagonal {
                i -= 1
                j -= 1
            } else if minCost == up {
                i -= 1
            } else {
                j -= 1
            }
        }

        path.append((0, j))
        let startIndex = j

        path.reverse()

        return SubsequenceDTWResult(
            distance: bestDistance,
            startIndex: startIndex,
            endIndex: bestEnd,
            path: path
        )
    }

    // MARK: - Private Implementation

    private func computeDistanceMatrix(_ x: [[Float]], _ y: [[Float]]) -> [[Float]] {
        let nX = x[0].count
        let nY = y[0].count

        var dist = [[Float]](repeating: [Float](repeating: 0, count: nY), count: nX)

        for i in 0..<nX {
            for j in 0..<nY {
                dist[i][j] = pointDistance(x, y, i, j)
            }
        }

        return dist
    }

    private func pointDistance(_ x: [[Float]], _ y: [[Float]], _ i: Int, _ j: Int) -> Float {
        let nFeatures = x.count

        switch config.metric {
        case .euclidean:
            var sum: Float = 0
            for f in 0..<nFeatures {
                let diff = x[f][i] - y[f][j]
                sum += diff * diff
            }
            return sqrt(sum)

        case .manhattan:
            var sum: Float = 0
            for f in 0..<nFeatures {
                sum += abs(x[f][i] - y[f][j])
            }
            return sum

        case .cosine:
            var dotProduct: Float = 0
            var normX: Float = 0
            var normY: Float = 0

            for f in 0..<nFeatures {
                dotProduct += x[f][i] * y[f][j]
                normX += x[f][i] * x[f][i]
                normY += y[f][j] * y[f][j]
            }

            let denom = sqrt(normX) * sqrt(normY)
            if denom > 0 {
                return 1.0 - dotProduct / denom
            }
            return 1.0
        }
    }

    private func isInWindow(_ i: Int, _ j: Int, _ nX: Int, _ nY: Int) -> Bool {
        switch config.windowType {
        case .none:
            return true

        case .sakoeChiba:
            let windowSize = config.windowParam ?? max(nX, nY)
            let expectedJ = Int(Float(j) * Float(nX) / Float(nY))
            return abs(i - expectedJ) <= windowSize

        case .itakura:
            let slope = Float(config.windowParam ?? 2)
            let ratio = Float(nY) / Float(nX)

            // Itakura parallelogram constraints
            let minJ = Float(i) * ratio / slope
            let maxJ = Float(i) * ratio * slope

            return Float(j) >= minJ && Float(j) <= maxJ
        }
    }

    private func backtrack(_ cost: [[Float]]) -> [(Int, Int)] {
        let nX = cost.count
        let nY = cost[0].count

        var path = [(Int, Int)]()
        var i = nX - 1
        var j = nY - 1

        while i > 0 || j > 0 {
            path.append((i, j))

            if i == 0 {
                j -= 1
            } else if j == 0 {
                i -= 1
            } else {
                let diagonal = cost[i - 1][j - 1]
                let up = cost[i - 1][j]
                let left = cost[i][j - 1]

                let minCost = min(diagonal, min(up, left))

                if minCost == diagonal {
                    i -= 1
                    j -= 1
                } else if minCost == up {
                    i -= 1
                } else {
                    j -= 1
                }
            }
        }

        path.append((0, 0))
        path.reverse()

        return path
    }
}

// MARK: - Presets

extension DTW {
    /// Standard configuration with Euclidean distance.
    public static var standard: DTW {
        DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .none,
            windowParam: nil
        ))
    }

    /// Configuration with Sakoe-Chiba band constraint.
    public static func withSakoeChibaBand(windowSize: Int) -> DTW {
        DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .sakoeChiba,
            windowParam: windowSize
        ))
    }

    /// Configuration optimized for speech comparison.
    public static var speech: DTW {
        DTW(config: DTWConfig(
            metric: .cosine,
            windowType: .sakoeChiba,
            windowParam: 50
        ))
    }
}
