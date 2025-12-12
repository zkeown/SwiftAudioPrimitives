import XCTest
@testable import SwiftAudioPrimitives

final class DTWTests: XCTestCase {

    // MARK: - Basic DTW Tests

    func testIdenticalSequences() {
        let dtw = DTW()

        // Two identical sequences should have zero distance
        let seq: [[Float]] = [[1, 2, 3, 4, 5]]

        let result = dtw.align(seq, seq)

        XCTAssertEqual(result.distance, 0, accuracy: 1e-6,
            "Identical sequences should have zero distance")

        // Path should be diagonal
        XCTAssertEqual(result.path.count, 5)
        for (i, j) in result.path {
            XCTAssertEqual(i, j, "Path should be diagonal for identical sequences")
        }
    }

    func testDifferentLengthSequences() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3, 4, 5]]
        let seq2: [[Float]] = [[1, 2, 2, 3, 3, 4, 5]]  // Longer with repetitions

        let result = dtw.align(seq1, seq2)

        // Should complete without error
        XCTAssertGreaterThan(result.path.count, 0)
        XCTAssertFalse(result.distance.isInfinite)
    }

    func testDistanceOnly() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3, 4, 5]]
        let seq2: [[Float]] = [[1.1, 2.1, 3.1, 4.1, 5.1]]

        let distance = dtw.distance(seq1, seq2)

        // Should be small but non-zero
        XCTAssertGreaterThan(distance, 0)
        XCTAssertLessThan(distance, 1)
    }

    func testNormalizedDistance() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3, 4, 5]]
        let seq2: [[Float]] = [[1.5, 2.5, 3.5, 4.5, 5.5]]

        let result = dtw.align(seq1, seq2)

        // Normalized distance should be total / path length
        let expectedNormalized = result.distance / Float(result.path.count)
        XCTAssertEqual(result.normalizedDistance, expectedNormalized, accuracy: 1e-6)
    }

    // MARK: - Multi-dimensional Sequences

    func testMultiDimensionalSequences() {
        let dtw = DTW()

        // 2D features (like 2 MFCC coefficients over 5 frames)
        let seq1: [[Float]] = [
            [1, 2, 3, 4, 5],  // Feature 1
            [5, 4, 3, 2, 1]   // Feature 2
        ]
        let seq2: [[Float]] = [
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [5.1, 4.1, 3.1, 2.1, 1.1]
        ]

        let result = dtw.align(seq1, seq2)

        XCTAssertGreaterThan(result.path.count, 0)
        XCTAssertFalse(result.distance.isNaN)
    }

    // MARK: - Distance Metrics

    func testEuclideanDistance() {
        let dtw = DTW(config: DTWConfig(metric: .euclidean))

        // 2 features, 1 frame each - like comparing two 2D points
        let seq1: [[Float]] = [[0], [0]]  // Point (0, 0)
        let seq2: [[Float]] = [[3], [4]]  // Point (3, 4)

        let distance = dtw.distance(seq1, seq2)

        // Euclidean distance from (0,0) to (3,4) = 5
        XCTAssertEqual(distance, 5, accuracy: 0.1)
    }

    func testManhattanDistance() {
        let dtw = DTW(config: DTWConfig(metric: .manhattan))

        // 2 features, 1 frame each
        let seq1: [[Float]] = [[0], [0]]  // Point (0, 0)
        let seq2: [[Float]] = [[3], [4]]  // Point (3, 4)

        let distance = dtw.distance(seq1, seq2)

        // Manhattan distance = |3-0| + |4-0| = 7
        XCTAssertEqual(distance, 7, accuracy: 0.1)
    }

    func testCosineDistance() {
        let dtw = DTW(config: DTWConfig(metric: .cosine))

        // 2 features, 1 frame each - same direction vectors should have zero cosine distance
        let seq1: [[Float]] = [[1], [2]]  // Vector (1, 2)
        let seq2: [[Float]] = [[2], [4]]  // Vector (2, 4) - same direction as seq1

        let distance = dtw.distance(seq1, seq2)

        XCTAssertLessThan(distance, 0.1, "Same-direction vectors should have low cosine distance")
    }

    // MARK: - Window Constraints

    func testSakoeChibaBand() {
        let dtwConstrained = DTW.withSakoeChibaBand(windowSize: 2)
        let dtwFull = DTW.standard

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        let resultConstrained = dtwConstrained.align(seq1, seq2)
        let resultFull = dtwFull.align(seq1, seq2)

        // Both should complete
        XCTAssertGreaterThan(resultConstrained.path.count, 0)
        XCTAssertGreaterThan(resultFull.path.count, 0)

        // For identical sequences, results should be similar
        XCTAssertEqual(resultConstrained.distance, resultFull.distance, accuracy: 0.1)
    }

    func testSakoeChibaPathConstraint() {
        // Test that Sakoe-Chiba actually constrains the path
        let windowSize = 2
        let dtw = DTW.withSakoeChibaBand(windowSize: windowSize)

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]

        let result = dtw.align(seq1, seq2, includeCostMatrix: true)

        // Verify path stays within window constraint
        let nX = seq1[0].count
        let nY = seq2[0].count
        for (i, j) in result.path {
            let expectedJ = Int(Float(i) * Float(nY) / Float(nX))
            let deviation = abs(j - expectedJ)
            XCTAssertLessThanOrEqual(deviation, windowSize,
                "Path point (\(i), \(j)) deviates \(deviation) from diagonal, expected <= \(windowSize)")
        }
    }

    func testSakoeChibaWithDifferentLengths() {
        // Test Sakoe-Chiba with sequences of different lengths
        let windowSize = 3
        let dtw = DTW.withSakoeChibaBand(windowSize: windowSize)

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  // 10 elements
        let seq2: [[Float]] = [[1, 2, 3, 4, 5]]  // 5 elements

        let result = dtw.align(seq1, seq2)

        // Should still find valid alignment
        XCTAssertGreaterThan(result.path.count, 0)
        XCTAssertFalse(result.distance.isInfinite, "Should find valid path with window constraint")

        // Path endpoints
        XCTAssertEqual(result.path.first?.0, 0)
        XCTAssertEqual(result.path.first?.1, 0)
        XCTAssertEqual(result.path.last?.0, 9)
        XCTAssertEqual(result.path.last?.1, 4)
    }

    func testItakuraParallelogram() {
        let dtw = DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .itakura,
            windowParam: 2  // slope constraint
        ))

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]

        let result = dtw.align(seq1, seq2, includeCostMatrix: true)

        // For identical equal-length sequences, path should be diagonal
        XCTAssertGreaterThan(result.path.count, 0)
        XCTAssertEqual(result.distance, 0, accuracy: 1e-6)

        // Verify path is diagonal
        for (i, j) in result.path {
            XCTAssertEqual(i, j, "Diagonal path expected for identical sequences")
        }
    }

    func testItakuraSlopeConstraint() {
        // Test that Itakura constraint limits slope deviation
        let slope = 2
        let dtw = DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .itakura,
            windowParam: slope
        ))

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        let result = dtw.align(seq1, seq2, includeCostMatrix: true)

        // Verify path satisfies slope constraints
        let nX = seq1[0].count
        let nY = seq2[0].count
        let expectedSlope = Float(nY) / Float(nX)

        for (i, j) in result.path {
            // Check slope from origin (skip origin itself)
            if i > 0 {
                let slopeFromStart = Float(j) / Float(i)
                let normalizedSlope = slopeFromStart / expectedSlope

                XCTAssertGreaterThanOrEqual(normalizedSlope, 1.0 / Float(slope) - 0.01,
                    "Slope from origin at (\(i), \(j)) = \(normalizedSlope) should be >= \(1.0 / Float(slope))")
                XCTAssertLessThanOrEqual(normalizedSlope, Float(slope) + 0.01,
                    "Slope from origin at (\(i), \(j)) = \(normalizedSlope) should be <= \(slope)")
            }
        }
    }

    func testItakuraWithDifferentLengths() {
        // Test Itakura with sequences of different lengths
        let dtw = DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .itakura,
            windowParam: 2
        ))

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]  // 8 elements
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6]]  // 6 elements

        let result = dtw.align(seq1, seq2)

        // Should find valid alignment
        XCTAssertGreaterThan(result.path.count, 0)
        XCTAssertFalse(result.distance.isInfinite, "Should find valid path with Itakura constraint")

        // Verify endpoints
        XCTAssertEqual(result.path.first?.0, 0)
        XCTAssertEqual(result.path.first?.1, 0)
        XCTAssertEqual(result.path.last?.0, 7)
        XCTAssertEqual(result.path.last?.1, 5)
    }

    func testWindowConstraintReducesComputation() {
        // Verify that window constraints produce valid but potentially different results
        let dtwFull = DTW.standard
        let dtwSakoe = DTW.withSakoeChibaBand(windowSize: 2)
        let dtwItakura = DTW(config: DTWConfig(
            metric: .euclidean,
            windowType: .itakura,
            windowParam: 2
        ))

        // Sequences that require warping
        let seq1: [[Float]] = [[1, 2, 2, 3, 3, 3, 4, 5]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 4, 4, 4, 5]]

        let resultFull = dtwFull.align(seq1, seq2)
        let resultSakoe = dtwSakoe.align(seq1, seq2)
        let resultItakura = dtwItakura.align(seq1, seq2)

        // All should produce valid results
        XCTAssertFalse(resultFull.distance.isInfinite)
        XCTAssertFalse(resultSakoe.distance.isInfinite)
        XCTAssertFalse(resultItakura.distance.isInfinite)

        // Constrained results should be >= unconstrained (can't do better with constraints)
        XCTAssertGreaterThanOrEqual(resultSakoe.distance, resultFull.distance - 0.01,
            "Sakoe-Chiba distance should be >= unconstrained")
        XCTAssertGreaterThanOrEqual(resultItakura.distance, resultFull.distance - 0.01,
            "Itakura distance should be >= unconstrained")
    }

    // MARK: - Subsequence DTW

    func testSubsequenceAlignment() {
        let dtw = DTW()

        // Query is a subsequence of reference
        let query: [[Float]] = [[3, 4, 5]]
        let reference: [[Float]] = [[1, 2, 3, 4, 5, 6, 7]]

        let result = dtw.subsequenceAlign(query: query, reference: reference)

        // Should find alignment starting around index 2
        XCTAssertEqual(result.startIndex, 2, "Should find subsequence starting at index 2")
        XCTAssertEqual(result.endIndex, 4, "Should find subsequence ending at index 4")
        XCTAssertLessThan(result.distance, 1, "Distance should be small for matching subsequence")
    }

    func testSubsequenceAlignmentWithOffset() {
        let dtw = DTW()

        let query: [[Float]] = [[10, 11, 12]]
        let reference: [[Float]] = [[0, 0, 0, 10, 11, 12, 0, 0]]

        let result = dtw.subsequenceAlign(query: query, reference: reference)

        // Should find alignment at indices 3-5
        XCTAssertGreaterThanOrEqual(result.startIndex, 2)
        XCTAssertLessThanOrEqual(result.endIndex, 6)
    }

    // MARK: - Cost Matrix

    func testCostMatrixIncluded() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3]]
        let seq2: [[Float]] = [[1, 2, 3, 4]]

        let result = dtw.align(seq1, seq2, includeCostMatrix: true)

        XCTAssertNotNil(result.costMatrix)

        if let costMatrix = result.costMatrix {
            XCTAssertEqual(costMatrix.count, 3, "Cost matrix should have 3 rows")
            XCTAssertEqual(costMatrix[0].count, 4, "Cost matrix should have 4 columns")
        }
    }

    func testCostMatrixExcluded() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3]]
        let seq2: [[Float]] = [[1, 2, 3, 4]]

        let result = dtw.align(seq1, seq2, includeCostMatrix: false)

        XCTAssertNil(result.costMatrix)
    }

    // MARK: - Presets

    func testPresets() {
        let standard = DTW.standard
        let withBand = DTW.withSakoeChibaBand(windowSize: 10)
        let speech = DTW.speech

        let seq1: [[Float]] = [[1, 2, 3, 4, 5]]
        let seq2: [[Float]] = [[1.1, 2.1, 3.1, 4.1, 5.1]]

        let _ = standard.distance(seq1, seq2)
        let _ = withBand.distance(seq1, seq2)
        let _ = speech.distance(seq1, seq2)

        XCTAssert(true)
    }

    // MARK: - Edge Cases

    func testEmptySequences() {
        let dtw = DTW()

        let result = dtw.align([[]], [[]])

        XCTAssertEqual(result.distance, Float.infinity)
        XCTAssertEqual(result.path.count, 0)
    }

    func testSingleElementSequences() {
        let dtw = DTW()

        let seq1: [[Float]] = [[5]]
        let seq2: [[Float]] = [[8]]

        let result = dtw.align(seq1, seq2)

        // Distance should be |5-8| = 3
        XCTAssertEqual(result.distance, 3, accuracy: 0.1)
        XCTAssertEqual(result.path.count, 1)
    }

    func testPathMonotonicity() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 3, 5, 7, 9]]
        let seq2: [[Float]] = [[2, 4, 6, 8]]

        let result = dtw.align(seq1, seq2)

        // Path should be monotonically non-decreasing
        for k in 1..<result.path.count {
            let (prevI, prevJ) = result.path[k-1]
            let (currI, currJ) = result.path[k]

            XCTAssertGreaterThanOrEqual(currI, prevI, "Path i-index should be non-decreasing")
            XCTAssertGreaterThanOrEqual(currJ, prevJ, "Path j-index should be non-decreasing")
        }
    }

    func testPathEndpoints() {
        let dtw = DTW()

        let seq1: [[Float]] = [[1, 2, 3, 4, 5]]
        let seq2: [[Float]] = [[1, 2, 3, 4]]

        let result = dtw.align(seq1, seq2)

        // Path should start at (0, 0)
        XCTAssertEqual(result.path.first?.0, 0)
        XCTAssertEqual(result.path.first?.1, 0)

        // Path should end at (n-1, m-1)
        XCTAssertEqual(result.path.last?.0, 4)
        XCTAssertEqual(result.path.last?.1, 3)
    }
}
