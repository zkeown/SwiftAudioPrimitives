import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Delta feature validation tests against librosa reference.
///
/// Validates Delta.compute() output against librosa.feature.delta() with:
/// - Width 3, 5, 9 (first-order delta)
/// - Order 2 (delta-delta/acceleration)
///
/// Uses Savitzky-Golay polynomial interpolation at boundaries to match librosa.
///
/// Reference: librosa 0.11.0
final class DeltaValidationTests: XCTestCase {

    // MARK: - Reference Data

    private static var referenceData: [String: Any]?

    override class func setUp() {
        super.setUp()
        loadReferenceData()
    }

    private static func loadReferenceData() {
        let possiblePaths = [
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceData/librosa_reference.json")
                .path,
            FileManager.default.currentDirectoryPath + "/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json"
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path),
               let data = FileManager.default.contents(atPath: path),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                referenceData = json
                return
            }
        }
    }

    private func getReference() throws -> [String: Any] {
        guard let ref = Self.referenceData else {
            throw XCTSkip("Reference data not found. Run scripts/generate_reference_data.py first.")
        }
        return ref
    }

    // MARK: - Helper Methods

    /// Transpose 2D array from [rows, cols] to [cols, rows]
    private func transpose(_ array: [[Float]]) -> [[Float]] {
        guard !array.isEmpty, !array[0].isEmpty else { return [] }
        let rows = array.count
        let cols = array[0].count
        var result = [[Float]](repeating: [Float](repeating: 0, count: rows), count: cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[j][i] = array[i][j]
            }
        }
        return result
    }

    /// Compute relative error between actual and expected arrays using hybrid tolerance.
    /// Uses the formula: abs(actual - expected) <= atol + rtol * abs(expected)
    /// This matches numpy's allclose behavior and handles near-zero values properly.
    ///
    /// - Parameters:
    ///   - rtol: Relative tolerance (default 1e-3 = 0.1%)
    ///   - atol: Absolute tolerance (default 1e-5)
    ///   - edgeSkip: Number of edge frames to skip on each side
    private func computeRelativeError(_ actual: [[Float]], _ expected: [[Float]], tolerance: Double, atol: Double = 1e-5, edgeSkip: Int = 0) -> (maxError: Double, meanError: Double, passRate: Double) {
        var maxError: Double = 0
        var sumError: Double = 0
        var passCount = 0
        var totalCount = 0

        let rows = min(actual.count, expected.count)
        for i in 0..<rows {
            let cols = min(actual[i].count, expected[i].count)
            // Skip edge frames on each side
            let startCol = edgeSkip
            let endCol = cols - edgeSkip
            guard startCol < endCol else { continue }

            for j in startCol..<endCol {
                let exp = Double(expected[i][j])
                let act = Double(actual[i][j])
                let absError = abs(act - exp)

                // Skip truly zero values
                if abs(exp) < 1e-10 && abs(act) < 1e-10 {
                    continue
                }

                // Use hybrid tolerance: pass if abs_error <= atol + rtol * abs(expected)
                let hybridThreshold = atol + tolerance * abs(exp)
                let passes = absError <= hybridThreshold

                // For statistics, use relative error where meaningful
                var relError: Double = 0
                if abs(exp) > atol {
                    relError = absError / abs(exp)
                } else {
                    // For small expected values, use absolute error scaled by atol
                    relError = absError / atol
                }

                maxError = max(maxError, relError)
                sumError += relError
                totalCount += 1

                if passes {
                    passCount += 1
                }
            }
        }

        let meanError = totalCount > 0 ? sumError / Double(totalCount) : 0
        let passRate = totalCount > 0 ? Double(passCount) / Double(totalCount) : 1.0
        return (maxError, meanError, passRate)
    }

    // MARK: - Delta Tests - Sine 440Hz

    /// Test delta width=3 for 440Hz sine
    /// Uses MFCC reference values directly to isolate Delta validation from MFCC implementation
    func testDelta_Sine440Hz_Width3() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let sineRef = deltaRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["delta_width_3"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccSineRef = mfccRef["sine_440hz"] as? [String: Any],
              let mfccFlat = mfccSineRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC sine_440hz reference data not found")
        }

        // Reference is [nFrames, nMFCC], need [nMFCC, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Use MFCC reference values directly (isolates Delta testing from MFCC impl)
        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        // Compute delta with width=3
        let actual = Delta.compute(mfccInput, width: 3, order: 1)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "Delta row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta)

        print("Delta Sine440Hz Width3 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta, "Delta mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta)")
    }

    /// Test delta width=5 for 440Hz sine
    func testDelta_Sine440Hz_Width5() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let sineRef = deltaRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["delta_width_5"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccSineRef = mfccRef["sine_440hz"] as? [String: Any],
              let mfccFlat = mfccSineRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC sine_440hz reference data not found")
        }

        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        let actual = Delta.compute(mfccInput, width: 5, order: 1)

        XCTAssertEqual(actual.count, expected.count, "Delta row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta)

        print("Delta Sine440Hz Width5 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta, "Delta mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta)")
    }

    /// Test delta width=9 for 440Hz sine
    func testDelta_Sine440Hz_Width9() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let sineRef = deltaRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["delta_width_9"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccSineRef = mfccRef["sine_440hz"] as? [String: Any],
              let mfccFlat = mfccSineRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC sine_440hz reference data not found")
        }

        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        let actual = Delta.compute(mfccInput, width: 9, order: 1)

        XCTAssertEqual(actual.count, expected.count, "Delta row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta)

        print("Delta Sine440Hz Width9 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta, "Delta mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta)")
    }

    /// Test delta-delta (order=2) for 440Hz sine
    func testDelta2_Sine440Hz_Width5() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let sineRef = deltaRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["delta2_width_5"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccSineRef = mfccRef["sine_440hz"] as? [String: Any],
              let mfccFlat = mfccSineRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC sine_440hz reference data not found")
        }

        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        // Order 2 = delta-delta
        let actual = Delta.compute(mfccInput, width: 5, order: 2)

        XCTAssertEqual(actual.count, expected.count, "Delta2 row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta2 column count mismatch")
        }

        // Compare with tolerance (use delta2 tolerance for 2nd derivative)
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta2)

        print("Delta2 Sine440Hz Width5 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta2 pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta2, "Delta2 mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta2)")
    }

    // MARK: - Delta Tests - Multi-Tone

    /// Test delta width=5 for multi-tone signal
    func testDelta_MultiTone_Width5() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let multiRef = deltaRef["multi_tone"] as? [String: Any],
              let expectedFlat = multiRef["delta_width_5"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccMultiRef = mfccRef["multi_tone"] as? [String: Any],
              let mfccFlat = mfccMultiRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC multi_tone reference data not found")
        }

        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        let actual = Delta.compute(mfccInput, width: 5, order: 1)

        XCTAssertEqual(actual.count, expected.count, "Delta row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta)

        print("Delta MultiTone Width5 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta, "Delta mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta)")
    }

    /// Test delta-delta for multi-tone signal
    func testDelta2_MultiTone_Width5() async throws {
        let ref = try getReference()
        guard let deltaRef = ref["delta"] as? [String: Any],
              let multiRef = deltaRef["multi_tone"] as? [String: Any],
              let expectedFlat = multiRef["delta2_width_5"] as? [[Double]],
              let mfccRef = ref["mfcc"] as? [String: Any],
              let mfccMultiRef = mfccRef["multi_tone"] as? [String: Any],
              let mfccFlat = mfccMultiRef["mfcc"] as? [[Double]] else {
            throw XCTSkip("Delta/MFCC multi_tone reference data not found")
        }

        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        let mfccTimeFeature = mfccFlat.map { $0.map { Float($0) } }
        let mfccInput = transpose(mfccTimeFeature)

        let actual = Delta.compute(mfccInput, width: 5, order: 2)

        XCTAssertEqual(actual.count, expected.count, "Delta2 row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Delta2 column count mismatch")
        }

        // Compare with tolerance (use delta2 tolerance for 2nd derivative)
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.delta2)

        print("Delta2 MultiTone Width5 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Delta2 pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.delta2, "Delta2 mean error \(meanError) exceeds tolerance \(ValidationTolerances.delta2)")
    }
}
