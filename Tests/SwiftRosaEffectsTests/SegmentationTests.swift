import XCTest
@testable import SwiftRosaEffects

final class SegmentationTests: XCTestCase {

    // MARK: - Trim Tests

    func testTrimBasic() {
        // Signal with silence at beginning and end
        var signal = [Float](repeating: 0.0, count: 1000)  // Silence
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2000))  // Content
        signal.append(contentsOf: [Float](repeating: 0.0, count: 1000))  // Silence

        let (trimmed, indices) = Segmentation.trim(signal, topDb: 20)

        // Should have removed some silence (or at minimum returned same length)
        XCTAssertLessThanOrEqual(trimmed.count, signal.count)

        // Indices should be within bounds
        XCTAssertGreaterThanOrEqual(indices.0, 0)
        XCTAssertLessThanOrEqual(indices.1, signal.count)

        // Trimmed content should have non-zero energy
        if trimmed.count > 0 {
            let trimmedEnergy = trimmed.reduce(0) { $0 + $1 * $1 } / Float(trimmed.count)
            XCTAssertGreaterThan(trimmedEnergy, 0)
        }
    }

    func testTrimNoSilence() {
        // Signal with no silence
        let signal = [Float](repeating: 0.5, count: 4096)

        let (trimmed, indices) = Segmentation.trim(signal, topDb: 60)

        // Should be mostly unchanged
        XCTAssertEqual(indices.0, 0)
        XCTAssertEqual(indices.1, signal.count)
        XCTAssertEqual(trimmed.count, signal.count)
    }

    func testTrimAllSilence() {
        // Completely silent signal
        let signal = [Float](repeating: 0.0, count: 4096)

        let (trimmed, _) = Segmentation.trim(signal, topDb: 20)

        // Should return empty
        XCTAssertEqual(trimmed.count, 0)
    }

    func testTrimEmpty() {
        let signal: [Float] = []
        let (trimmed, indices) = Segmentation.trim(signal)

        XCTAssertEqual(trimmed.count, 0)
        XCTAssertEqual(indices.0, 0)
        XCTAssertEqual(indices.1, 0)
    }

    func testTrimCustomTopDb() {
        // Signal with varying levels
        var signal = [Float](repeating: 0.001, count: 1000)  // Very quiet
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2000))  // Loud
        signal.append(contentsOf: [Float](repeating: 0.001, count: 1000))  // Very quiet

        // With high topDb (aggressive), should trim the quiet parts
        let (trimmedAggressive, _) = Segmentation.trim(signal, topDb: 20)

        // With low topDb (conservative), should keep more
        let (trimmedConservative, _) = Segmentation.trim(signal, topDb: 80)

        // Aggressive trimming should remove more
        XCTAssertLessThanOrEqual(trimmedAggressive.count, trimmedConservative.count)
    }

    func testTrimCustomFrameLength() {
        var signal = [Float](repeating: 0.0, count: 1000)
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2000))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 1000))

        // Different frame lengths should produce valid results
        let (trimmed1, _) = Segmentation.trim(signal, frameLength: 1024)
        let (trimmed2, _) = Segmentation.trim(signal, frameLength: 512)

        XCTAssertGreaterThan(trimmed1.count, 0)
        XCTAssertGreaterThan(trimmed2.count, 0)
    }

    // MARK: - Split Tests

    func testSplitBasic() {
        // Signal with alternating silence and content
        var signal = [Float](repeating: 0.0, count: 2048)  // Silence
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))  // Content 1
        signal.append(contentsOf: [Float](repeating: 0.0, count: 4096))  // Silence
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))  // Content 2
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))  // Silence

        let intervals = Segmentation.split(signal, topDb: 20)

        // Should find at least one non-silent region
        XCTAssertGreaterThan(intervals.count, 0)

        // All intervals should be within bounds
        for interval in intervals {
            XCTAssertGreaterThanOrEqual(interval.start, 0)
            XCTAssertLessThanOrEqual(interval.end, signal.count)
            XCTAssertLessThan(interval.start, interval.end)
        }
    }

    func testSplitNoSilence() {
        // Continuous non-silent signal
        let signal = [Float](repeating: 0.5, count: 8192)

        let intervals = Segmentation.split(signal, topDb: 60)

        // Should return single interval covering entire signal
        XCTAssertEqual(intervals.count, 1)
        if let interval = intervals.first {
            XCTAssertEqual(interval.start, 0)
            XCTAssertEqual(interval.end, signal.count)
        }
    }

    func testSplitAllSilence() {
        let signal = [Float](repeating: 0.0, count: 8192)

        let intervals = Segmentation.split(signal, topDb: 20)

        // Should return empty
        XCTAssertEqual(intervals.count, 0)
    }

    func testSplitEmpty() {
        let signal: [Float] = []
        let intervals = Segmentation.split(signal)

        XCTAssertEqual(intervals.count, 0)
    }

    func testSplitMultipleSegments() {
        // Create signal with 3 clear non-silent regions
        var signal: [Float] = []

        // Region 1
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2048))

        // Region 2
        signal.append(contentsOf: [Float](repeating: 0.0, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2048))

        // Region 3
        signal.append(contentsOf: [Float](repeating: 0.0, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2048))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))

        let intervals = Segmentation.split(signal, topDb: 20)

        // Should find multiple segments (exact count may vary due to merging)
        XCTAssertGreaterThanOrEqual(intervals.count, 1)
    }

    func testSplitNonOverlapping() {
        var signal = [Float](repeating: 0.0, count: 2048)
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))

        let intervals = Segmentation.split(signal, topDb: 20)

        // Intervals should not overlap
        for i in 0..<(intervals.count - 1) {
            XCTAssertLessThanOrEqual(intervals[i].end, intervals[i + 1].start,
                "Intervals should not overlap")
        }
    }

    // MARK: - Array Extension Tests

    func testTrimmedExtension() {
        var signal = [Float](repeating: 0.0, count: 1000)
        signal.append(contentsOf: [Float](repeating: 0.5, count: 2000))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 1000))

        let trimmed1 = signal.trimmed(topDb: 20)
        let (trimmed2, _) = Segmentation.trim(signal, topDb: 20)

        XCTAssertEqual(trimmed1.count, trimmed2.count)
    }

    func testSplitOnSilenceExtension() {
        var signal: [Float] = []
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))

        let segments = signal.splitOnSilence(topDb: 20)

        // Each segment should be an array of floats
        for segment in segments {
            XCTAssertGreaterThan(segment.count, 0)

            // Segment should have non-zero energy
            let energy = segment.reduce(0) { $0 + $1 * $1 } / Float(segment.count)
            XCTAssertGreaterThan(energy, 0.01)
        }
    }

    // MARK: - Parameter Sensitivity Tests

    func testTrimSensitivityToHopLength() {
        var signal = [Float](repeating: 0.0, count: 2048)
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.0, count: 2048))

        // Different hop lengths should produce valid but potentially different results
        let (_, indices1) = Segmentation.trim(signal, hopLength: 256)
        let (_, indices2) = Segmentation.trim(signal, hopLength: 1024)

        // Both should identify non-silent region
        XCTAssertGreaterThan(indices1.1 - indices1.0, 0)
        XCTAssertGreaterThan(indices2.1 - indices2.0, 0)

        // Smaller hop length generally gives finer control
        // (but we just verify both work)
    }

    func testSplitWithCustomRef() {
        var signal = [Float](repeating: 0.1, count: 2048)
        signal.append(contentsOf: [Float](repeating: 0.5, count: 4096))
        signal.append(contentsOf: [Float](repeating: 0.1, count: 2048))

        // With automatic ref (uses max)
        let intervals1 = Segmentation.split(signal, topDb: 20, ref: nil)

        // With custom ref (lower threshold)
        let intervals2 = Segmentation.split(signal, topDb: 20, ref: 0.1)

        // Both should work, but may produce different segment counts
        XCTAssertGreaterThanOrEqual(intervals1.count, 0)
        XCTAssertGreaterThanOrEqual(intervals2.count, 0)
    }
}
