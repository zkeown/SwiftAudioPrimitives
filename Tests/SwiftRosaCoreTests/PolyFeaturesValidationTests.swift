import XCTest
@testable import SwiftRosaCore

/// Polynomial features validation tests against librosa reference.
///
/// Validates PolyFeatures output against librosa.feature.poly_features() with:
/// - Order 1 (linear fit)
/// - Order 2 (quadratic fit)
///
/// Reference: librosa 0.11.0
final class PolyFeaturesValidationTests: XCTestCase {

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

    /// Compute relative error between actual and expected arrays
    private func computeRelativeError(_ actual: [[Float]], _ expected: [[Float]], tolerance: Double) -> (maxError: Double, meanError: Double, passRate: Double) {
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

                // Skip near-zero values where relative error is meaningless
                if abs(exp) < 1e-6 {
                    continue
                }

                let relError = abs(act - exp) / max(abs(exp), 1e-10)
                maxError = max(maxError, relError)
                sumError += relError
                totalCount += 1

                if relError < tolerance {
                    passCount += 1
                }
            }
        }

        let meanError = totalCount > 0 ? sumError / Double(totalCount) : 0
        let passRate = totalCount > 0 ? Double(passCount) / Double(totalCount) : 1.0
        return (maxError, meanError, passRate)
    }

    // MARK: - Poly Features Tests

    /// Test polynomial features order 1 (linear) for multi-tone signal
    func testPolyFeatures_Order1() async throws {
        let ref = try getReference()
        guard let polyRef = ref["poly_features"] as? [String: Any],
              let order1Ref = polyRef["order_1"] as? [String: Any],
              let expectedFlat = order1Ref["poly_features"] as? [[Double]],
              let order = order1Ref["order"] as? Int,
              let nFFT = order1Ref["n_fft"] as? Int,
              let hopLength = order1Ref["hop_length"] as? Int else {
            throw XCTSkip("Poly features order_1 reference data not found")
        }

        // Reference is [nFrames, order+1], need [order+1, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Generate test signal (multi-tone, matching reference data generation)
        let signal = TestSignalGenerator.multiTone()

        // Compute STFT magnitude
        let stftConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            winLength: nFFT,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: stftConfig)
        let magnitude = await stft.magnitude(signal)

        // Compute polynomial features
        let poly = PolyFeatures(config: PolyFeaturesConfig(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            order: order
        ))
        let actual = poly.transform(magnitudeSpec: magnitude)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PolyFeatures row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PolyFeatures column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.polyFeatures)

        print("PolyFeatures Order1 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PolyFeatures pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.polyFeatures, "PolyFeatures mean error \(meanError) exceeds tolerance \(ValidationTolerances.polyFeatures)")
    }

    /// Test polynomial features order 2 (quadratic) for multi-tone signal
    func testPolyFeatures_Order2() async throws {
        let ref = try getReference()
        guard let polyRef = ref["poly_features"] as? [String: Any],
              let order2Ref = polyRef["order_2"] as? [String: Any],
              let expectedFlat = order2Ref["poly_features"] as? [[Double]],
              let order = order2Ref["order"] as? Int,
              let nFFT = order2Ref["n_fft"] as? Int,
              let hopLength = order2Ref["hop_length"] as? Int else {
            throw XCTSkip("Poly features order_2 reference data not found")
        }

        // Reference is [nFrames, order+1], need [order+1, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Generate test signal (multi-tone, matching reference data generation)
        let signal = TestSignalGenerator.multiTone()

        // Compute STFT magnitude
        let stftConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            winLength: nFFT,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: stftConfig)
        let magnitude = await stft.magnitude(signal)

        // Compute polynomial features
        let poly = PolyFeatures(config: PolyFeaturesConfig(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            order: order
        ))
        let actual = poly.transform(magnitudeSpec: magnitude)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PolyFeatures row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PolyFeatures column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.polyFeatures)

        print("PolyFeatures Order2 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PolyFeatures pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.polyFeatures, "PolyFeatures mean error \(meanError) exceeds tolerance \(ValidationTolerances.polyFeatures)")
    }
}
