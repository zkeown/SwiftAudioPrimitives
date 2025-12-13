import XCTest
@testable import SwiftRosaEffects

final class RQATests: XCTestCase {

    // MARK: - Basic Recurrence Matrix Tests

    func testRecurrenceMatrixBasic() {
        let rqa = RQA()

        // Simple feature sequence (2 features, 5 frames)
        let features: [[Float]] = [
            [1, 2, 3, 2, 1],  // Feature 1
            [0, 1, 1, 1, 0]   // Feature 2
        ]

        let result = rqa.compute(features)

        // Matrix should be square (nFrames x nFrames)
        XCTAssertEqual(result.recurrenceMatrix.count, 5)
        XCTAssertEqual(result.recurrenceMatrix[0].count, 5)
    }

    func testRecurrenceMatrixSymmetric() {
        let rqa = RQA(config: RQAConfig(symmetric: true))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ]

        let result = rqa.compute(features)

        // Matrix should be symmetric
        let n = result.recurrenceMatrix.count
        for i in 0..<n {
            for j in 0..<n {
                XCTAssertEqual(result.recurrenceMatrix[i][j],
                              result.recurrenceMatrix[j][i],
                              accuracy: 1e-6,
                              "Matrix should be symmetric at (\(i), \(j))")
            }
        }
    }

    func testRecurrenceMatrixDiagonal() {
        let rqa = RQA(config: RQAConfig(excludeSelf: false))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5]
        ]

        let result = rqa.compute(features)

        // Diagonal should be zero (distance to self)
        for i in 0..<result.recurrenceMatrix.count {
            XCTAssertEqual(result.recurrenceMatrix[i][i], 0, accuracy: 1e-6,
                "Diagonal should be zero (self-distance)")
        }
    }

    func testRecurrenceMatrixExcludeSelf() {
        let rqa = RQA(config: RQAConfig(excludeSelf: true))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5]
        ]

        let result = rqa.compute(features)

        // Diagonal should be infinity when excluding self
        for i in 0..<result.recurrenceMatrix.count {
            XCTAssertEqual(result.recurrenceMatrix[i][i], Float.infinity,
                "Diagonal should be infinity when excluding self")
        }
    }

    // MARK: - Distance Metric Tests

    func testEuclideanMetric() {
        let rqa = RQA(config: RQAConfig(metric: .euclidean, excludeSelf: false))

        // Two frames with known Euclidean distance
        let features: [[Float]] = [
            [0, 3],  // Frame 0: (0,0), Frame 1: (3,4)
            [0, 4]
        ]

        let result = rqa.compute(features)

        // Distance from frame 0 to frame 1 should be sqrt(3^2 + 4^2) = 5
        XCTAssertEqual(result.recurrenceMatrix[0][1], 5, accuracy: 0.1)
    }

    func testCosineMetric() {
        let rqa = RQA(config: RQAConfig(metric: .cosine, excludeSelf: false))

        // Two frames with same direction should have zero cosine distance
        let features: [[Float]] = [
            [1, 2],  // Frame 0: (1,1), Frame 1: (2,2) - same direction
            [1, 2]
        ]

        let result = rqa.compute(features)

        // Same direction vectors should have ~0 cosine distance
        XCTAssertEqual(result.recurrenceMatrix[0][1], 0, accuracy: 0.1)
    }

    func testChebyshevMetric() {
        let rqa = RQA(config: RQAConfig(metric: .chebyshev, excludeSelf: false))

        // Chebyshev = max absolute difference
        let features: [[Float]] = [
            [0, 5],  // Frame 0: (0,0), Frame 1: (5,3)
            [0, 3]
        ]

        let result = rqa.compute(features)

        // Chebyshev distance = max(|5-0|, |3-0|) = 5
        XCTAssertEqual(result.recurrenceMatrix[0][1], 5, accuracy: 0.1)
    }

    // MARK: - Bandwidth Constraint Tests

    func testBandwidthConstraint() {
        let bandwidth = 2
        let rqa = RQA(config: RQAConfig(bandwidth: bandwidth))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]

        let result = rqa.compute(features)

        // Points outside bandwidth should be infinity
        let n = result.recurrenceMatrix.count
        for i in 0..<n {
            for j in 0..<n {
                if abs(i - j) > bandwidth {
                    XCTAssertEqual(result.recurrenceMatrix[i][j], Float.infinity,
                        "Point (\(i), \(j)) should be infinity outside bandwidth")
                }
            }
        }
    }

    // MARK: - K-Nearest Neighbors Tests

    func testKNNConstraint() {
        let k = 3
        let rqa = RQA(config: RQAConfig(k: k, symmetric: false, excludeSelf: true))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]

        let result = rqa.compute(features)

        // Each row should have at most k finite values (with symmetric=false)
        for row in result.recurrenceMatrix {
            let finiteCount = row.filter { $0.isFinite }.count
            XCTAssertLessThanOrEqual(finiteCount, k,
                "Row should have at most \(k) finite values")
        }
    }

    // MARK: - Sparse Representation Tests

    func testSparseRepresentation() {
        let maxSize = 10
        let rqa = RQA(config: RQAConfig(maxSize: maxSize, excludeSelf: true))

        let features: [[Float]] = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]

        let result = rqa.compute(features)

        XCTAssertNotNil(result.sparse)
        if let sparse = result.sparse {
            XCTAssertLessThanOrEqual(sparse.count, maxSize)

            // Sparse entries should be sorted by distance
            for i in 0..<(sparse.count - 1) {
                XCTAssertLessThanOrEqual(sparse[i].2, sparse[i + 1].2)
            }
        }
    }

    // MARK: - RQA Statistics Tests

    func testRecurrenceRate() {
        let rqa = RQA()

        // Periodic signal should have high recurrence rate
        let features: [[Float]] = [
            [1, 2, 3, 1, 2, 3, 1, 2, 3]
        ]

        let result = rqa.compute(features)

        // Recurrence rate should be between 0 and 1
        XCTAssertGreaterThanOrEqual(result.recurrenceRate, 0)
        XCTAssertLessThanOrEqual(result.recurrenceRate, 1)
    }

    func testDeterminism() {
        let rqa = RQA()

        let features: [[Float]] = [
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        ]

        let result = rqa.compute(features)

        // Determinism should be between 0 and 1
        XCTAssertGreaterThanOrEqual(result.determinism, 0)
        XCTAssertLessThanOrEqual(result.determinism, 1)
    }

    func testLaminarity() {
        let rqa = RQA()

        // Signal with some constant regions
        let features: [[Float]] = [
            [1, 1, 1, 2, 2, 2, 3, 3, 3]
        ]

        let result = rqa.compute(features)

        // Laminarity should be between 0 and 1
        XCTAssertGreaterThanOrEqual(result.laminarity, 0)
        XCTAssertLessThanOrEqual(result.laminarity, 1)
    }

    // MARK: - Binary Recurrence Tests

    func testBinaryRecurrence() {
        let distMatrix: [[Float]] = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ]

        let binary = RQA.binaryRecurrence(distMatrix, threshold: 1.5)

        // Values <= 1.5 should be 1, others 0
        XCTAssertEqual(binary[0][0], 1)  // 0 <= 1.5
        XCTAssertEqual(binary[0][1], 1)  // 1 <= 1.5
        XCTAssertEqual(binary[0][2], 0)  // 2 > 1.5
    }

    // MARK: - Presets Tests

    func testStandardPreset() {
        let rqa = RQA.standard

        let features: [[Float]] = [[1, 2, 3, 4, 5]]
        let result = rqa.compute(features)

        XCTAssertGreaterThan(result.recurrenceMatrix.count, 0)
    }

    func testSparsePreset() {
        let rqa = RQA.sparse(k: 3)

        let features: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]
        let result = rqa.compute(features)

        // Sparse preset applies KNN - verify matrix has structure
        // Due to symmetric=true in the preset, rows may have more than k values
        // but the total number of unique finite pairs should be limited
        var totalFinite = 0
        for row in result.recurrenceMatrix {
            totalFinite += row.filter { $0.isFinite }.count
        }
        // Should have significantly fewer entries than a full matrix
        let fullMatrixEntries = features[0].count * features[0].count
        XCTAssertLessThan(totalFinite, fullMatrixEntries)
    }

    func testLocalPreset() {
        let bandwidth = 2
        let rqa = RQA.local(bandwidth: bandwidth)

        let features: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]
        let result = rqa.compute(features)

        // Points outside bandwidth should be infinity
        let n = result.recurrenceMatrix.count
        for i in 0..<n {
            for j in 0..<n {
                if abs(i - j) > bandwidth {
                    XCTAssertEqual(result.recurrenceMatrix[i][j], Float.infinity)
                }
            }
        }
    }

    func testMusicPreset() {
        let rqa = RQA.music

        let features: [[Float]] = [[1, 2, 3, 4, 5]]
        let result = rqa.compute(features)

        XCTAssertGreaterThan(result.recurrenceMatrix.count, 0)
    }

    // MARK: - Edge Cases

    func testEmptyFeatures() {
        let rqa = RQA()

        let features: [[Float]] = []
        let result = rqa.compute(features)

        XCTAssertEqual(result.recurrenceMatrix.count, 0)
        XCTAssertEqual(result.recurrenceRate, 0)
    }

    func testSingleFrame() {
        let rqa = RQA()

        let features: [[Float]] = [[1]]
        let result = rqa.compute(features)

        XCTAssertEqual(result.recurrenceMatrix.count, 1)
        XCTAssertEqual(result.recurrenceMatrix[0].count, 1)
    }

    func testRecurrenceMatrixOnlyMethod() {
        let rqa = RQA()

        let features: [[Float]] = [[1, 2, 3, 4, 5]]

        // Method that only returns recurrence matrix
        let matrix = rqa.recurrenceMatrix(features)

        XCTAssertEqual(matrix.count, 5)
        XCTAssertEqual(matrix[0].count, 5)
    }

    func testMultiDimensionalFeatures() {
        let rqa = RQA()

        // 3 features, 5 frames (like 3 MFCCs over 5 frames)
        let features: [[Float]] = [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1]
        ]

        let result = rqa.compute(features)

        XCTAssertEqual(result.recurrenceMatrix.count, 5)
        XCTAssertGreaterThanOrEqual(result.recurrenceRate, 0)
    }
}
