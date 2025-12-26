import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Tonnetz validation tests against librosa reference.
///
/// Validates Tonnetz output against librosa.feature.tonnetz() for:
/// - 440Hz sine wave
/// - Multi-tone harmonic signal
///
/// Reference: librosa 0.11.0
final class TonnetzValidationTests: XCTestCase {

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
    private func computeRelativeError(_ actual: [[Float]], _ expected: [[Float]], tolerance: Double, atol: Double = 1e-5) -> (maxError: Double, meanError: Double, passRate: Double) {
        var maxError: Double = 0
        var sumError: Double = 0
        var passCount = 0
        var totalCount = 0

        let rows = min(actual.count, expected.count)
        for i in 0..<rows {
            let cols = min(actual[i].count, expected[i].count)
            for j in 0..<cols {
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

    // MARK: - Tonnetz Tests

    /// Test Tonnetz transformation using reference chroma_cqt as input.
    /// This isolates the Tonnetz transformation from CQT/chroma computation differences.
    func testTonnetz_Sine440Hz() async throws {
        let ref = try getReference()
        guard let tonnetzRef = ref["tonnetz"] as? [String: Any],
              let sineRef = tonnetzRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["tonnetz"] as? [[Double]],
              let chromaFlat = sineRef["chroma_cqt"] as? [[Double]],
              let hopLength = sineRef["hop_length"] as? Int else {
            throw XCTSkip("Tonnetz sine_440hz reference data not found")
        }

        // Reference is [nFrames, 6], need [6, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Use reference chroma_cqt directly (isolates Tonnetz from CQT/chroma issues)
        let chromaTimeFeature = chromaFlat.map { $0.map { Float($0) } }
        let chromaInput = transpose(chromaTimeFeature)

        // Compute Tonnetz from reference chroma
        let tonnetz = Tonnetz(config: TonnetzConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            nChroma: 12
        ))
        let actual = tonnetz.transform(chroma: chromaInput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "Tonnetz row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Tonnetz column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.tonnetz)

        print("Tonnetz Sine440Hz - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Tonnetz pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.tonnetz, "Tonnetz mean error \(meanError) exceeds tolerance \(ValidationTolerances.tonnetz)")
    }

    /// Test Tonnetz transformation for multi-tone signal using reference chroma_cqt.
    func testTonnetz_MultiTone() async throws {
        let ref = try getReference()
        guard let tonnetzRef = ref["tonnetz"] as? [String: Any],
              let multiRef = tonnetzRef["multi_tone"] as? [String: Any],
              let expectedFlat = multiRef["tonnetz"] as? [[Double]],
              let chromaFlat = multiRef["chroma_cqt"] as? [[Double]],
              let hopLength = multiRef["hop_length"] as? Int else {
            throw XCTSkip("Tonnetz multi_tone reference data not found")
        }

        // Reference is [nFrames, 6], need [6, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Use reference chroma_cqt directly
        let chromaTimeFeature = chromaFlat.map { $0.map { Float($0) } }
        let chromaInput = transpose(chromaTimeFeature)

        // Compute Tonnetz from reference chroma
        let tonnetz = Tonnetz(config: TonnetzConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            nChroma: 12
        ))
        let actual = tonnetz.transform(chroma: chromaInput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "Tonnetz row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "Tonnetz column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.tonnetz)

        print("Tonnetz MultiTone - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "Tonnetz pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.tonnetz, "Tonnetz mean error \(meanError) exceeds tolerance \(ValidationTolerances.tonnetz)")
    }
}
