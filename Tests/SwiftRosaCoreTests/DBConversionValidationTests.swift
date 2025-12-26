import XCTest
@testable import SwiftRosaCore

/// dB conversion validation tests against librosa reference.
///
/// Validates:
/// - power_to_db: 10 * log10(S / ref)
/// - amplitude_to_db: 20 * log10(S / ref)
/// - db_to_power: ref * 10^(S_db / 10)
/// - db_to_amplitude: ref * 10^(S_db / 20)
///
/// Reference: librosa 0.11.0
final class DBConversionValidationTests: XCTestCase {

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

    // MARK: - Power to dB Tests

    /// Test power_to_db against librosa reference
    func testPowerToDb() throws {
        let ref = try getReference()
        guard let dbRef = ref["db_conversion"] as? [String: Any],
              let powerValues = dbRef["power_values"] as? [Double],
              let expectedDb = dbRef["power_to_db"] as? [Double],
              let refValue = dbRef["ref"] as? Double else {
            throw XCTSkip("dB conversion reference data not found")
        }

        // Convert to 2D array (single row) for Convert API
        let input: [[Float]] = [powerValues.map { Float($0) }]

        // Use topDb=80 to match librosa default behavior (reference data uses defaults)
        // Note: librosa uses amin=1e-10 by default
        let result = Convert.powerToDb(input, ref: Float(refValue), amin: 1e-10, topDb: 80.0)

        XCTAssertEqual(result.count, 1, "Should have one row")
        XCTAssertEqual(result[0].count, expectedDb.count, "Column count should match")

        // Compare each value
        for i in 0..<expectedDb.count {
            let expected = Float(expectedDb[i])
            let actual = result[0][i]
            let absError = abs(expected - actual)

            // dB values can be large negative numbers, use absolute error for those
            // and relative error for reasonable values
            if abs(expected) > 1 {
                let relError = Double(absError / abs(expected))
                XCTAssertLessThan(relError, ValidationTolerances.dbConversion,
                    "powerToDb[\(i)]: expected \(expected), got \(actual), relError \(relError)")
            } else {
                XCTAssertLessThan(Double(absError), 0.01,
                    "powerToDb[\(i)]: expected \(expected), got \(actual), absError \(absError)")
            }
        }
    }

    /// Test amplitude_to_db against librosa reference
    func testAmplitudeToDb() throws {
        let ref = try getReference()
        guard let dbRef = ref["db_conversion"] as? [String: Any],
              let amplitudeValues = dbRef["amplitude_values"] as? [Double],
              let expectedDb = dbRef["amplitude_to_db"] as? [Double],
              let refValue = dbRef["ref"] as? Double else {
            throw XCTSkip("dB conversion reference data not found")
        }

        // Convert to 2D array
        let input: [[Float]] = [amplitudeValues.map { Float($0) }]

        // Use amin=1e-5 and topDb=80 (librosa defaults for amplitude)
        let result = Convert.amplitudeToDb(input, ref: Float(refValue), amin: 1e-5, topDb: 80.0)

        XCTAssertEqual(result.count, 1, "Should have one row")
        XCTAssertEqual(result[0].count, expectedDb.count, "Column count should match")

        // Compare each value
        for i in 0..<expectedDb.count {
            let expected = Float(expectedDb[i])
            let actual = result[0][i]
            let absError = abs(expected - actual)

            if abs(expected) > 1 {
                let relError = Double(absError / abs(expected))
                XCTAssertLessThan(relError, ValidationTolerances.dbConversion,
                    "amplitudeToDb[\(i)]: expected \(expected), got \(actual), relError \(relError)")
            } else {
                XCTAssertLessThan(Double(absError), 0.01,
                    "amplitudeToDb[\(i)]: expected \(expected), got \(actual), absError \(absError)")
            }
        }
    }

    // MARK: - dB to Power/Amplitude Tests

    /// Test db_to_power against librosa reference
    func testDbToPower() throws {
        let ref = try getReference()
        guard let dbRef = ref["db_conversion"] as? [String: Any],
              let powerToDbValues = dbRef["power_to_db"] as? [Double],
              let expectedPower = dbRef["db_to_power"] as? [Double],
              let refValue = dbRef["ref"] as? Double else {
            throw XCTSkip("dB conversion reference data not found")
        }

        // Convert to 2D array
        let input: [[Float]] = [powerToDbValues.map { Float($0) }]

        let result = Convert.dbToPower(input, ref: Float(refValue))

        XCTAssertEqual(result.count, 1, "Should have one row")
        XCTAssertEqual(result[0].count, expectedPower.count, "Column count should match")

        // Compare each value
        for i in 0..<expectedPower.count {
            let expected = Float(expectedPower[i])
            let actual = result[0][i]

            // For very small values, use absolute error
            if expected < 1e-5 {
                let absError = abs(expected - actual)
                XCTAssertLessThan(Double(absError), 1e-8,
                    "dbToPower[\(i)]: expected \(expected), got \(actual)")
            } else {
                let relError = Double(abs(expected - actual) / expected)
                XCTAssertLessThan(relError, ValidationTolerances.dbConversion,
                    "dbToPower[\(i)]: expected \(expected), got \(actual), relError \(relError)")
            }
        }
    }

    /// Test db_to_amplitude against librosa reference
    func testDbToAmplitude() throws {
        let ref = try getReference()
        guard let dbRef = ref["db_conversion"] as? [String: Any],
              let amplitudeToDbValues = dbRef["amplitude_to_db"] as? [Double],
              let expectedAmplitude = dbRef["db_to_amplitude"] as? [Double],
              let refValue = dbRef["ref"] as? Double else {
            throw XCTSkip("dB conversion reference data not found")
        }

        // Convert to 2D array
        let input: [[Float]] = [amplitudeToDbValues.map { Float($0) }]

        let result = Convert.dbToAmplitude(input, ref: Float(refValue))

        XCTAssertEqual(result.count, 1, "Should have one row")
        XCTAssertEqual(result[0].count, expectedAmplitude.count, "Column count should match")

        // Compare each value
        for i in 0..<expectedAmplitude.count {
            let expected = Float(expectedAmplitude[i])
            let actual = result[0][i]

            // For very small values, use absolute error
            if expected < 1e-4 {
                let absError = abs(expected - actual)
                XCTAssertLessThan(Double(absError), 1e-6,
                    "dbToAmplitude[\(i)]: expected \(expected), got \(actual)")
            } else {
                let relError = Double(abs(expected - actual) / expected)
                XCTAssertLessThan(relError, ValidationTolerances.dbConversion,
                    "dbToAmplitude[\(i)]: expected \(expected), got \(actual), relError \(relError)")
            }
        }
    }

    // MARK: - Round-Trip Tests

    /// Test power -> dB -> power round-trip
    func testPowerRoundTrip() {
        let testValues: [Float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        let input: [[Float]] = [testValues]

        // Forward: power -> dB
        let db = Convert.powerToDb(input, ref: 1.0, amin: 1e-10, topDb: nil)

        // Inverse: dB -> power
        let recovered = Convert.dbToPower(db, ref: 1.0)

        XCTAssertEqual(recovered.count, 1)
        XCTAssertEqual(recovered[0].count, testValues.count)

        for i in 0..<testValues.count {
            let original = testValues[i]
            let roundTrip = recovered[0][i]
            let relError = Double(abs(original - roundTrip) / original)

            XCTAssertLessThan(relError, 1e-5,
                "Round-trip error at \(original): got \(roundTrip), relError \(relError)")
        }
    }

    /// Test amplitude -> dB -> amplitude round-trip
    func testAmplitudeRoundTrip() {
        let testValues: [Float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        let input: [[Float]] = [testValues]

        // Forward: amplitude -> dB
        let db = Convert.amplitudeToDb(input, ref: 1.0, amin: 1e-10, topDb: nil)

        // Inverse: dB -> amplitude
        let recovered = Convert.dbToAmplitude(db, ref: 1.0)

        XCTAssertEqual(recovered.count, 1)
        XCTAssertEqual(recovered[0].count, testValues.count)

        for i in 0..<testValues.count {
            let original = testValues[i]
            let roundTrip = recovered[0][i]
            let relError = Double(abs(original - roundTrip) / original)

            XCTAssertLessThan(relError, 1e-5,
                "Round-trip error at \(original): got \(roundTrip), relError \(relError)")
        }
    }

    // MARK: - Edge Cases

    /// Test with empty input
    func testEmptyInput() {
        let empty: [[Float]] = []

        XCTAssertTrue(Convert.powerToDb(empty).isEmpty)
        XCTAssertTrue(Convert.amplitudeToDb(empty).isEmpty)
        XCTAssertTrue(Convert.dbToPower(empty).isEmpty)
        XCTAssertTrue(Convert.dbToAmplitude(empty).isEmpty)
    }

    /// Test clamping behavior for very small values
    func testAminClamping() {
        // Values below amin should be clamped
        let tinyValues: [[Float]] = [[1e-15, 1e-12, 1e-10, 1e-8]]

        // With amin=1e-10, first three values should give same dB
        let dbResult = Convert.powerToDb(tinyValues, ref: 1.0, amin: 1e-10, topDb: nil)

        // 10 * log10(1e-10) = -100
        let expectedClamped: Float = -100.0

        XCTAssertEqual(dbResult[0][0], expectedClamped, accuracy: 0.01,
            "Value 1e-15 should be clamped to amin")
        XCTAssertEqual(dbResult[0][1], expectedClamped, accuracy: 0.01,
            "Value 1e-12 should be clamped to amin")
        XCTAssertEqual(dbResult[0][2], expectedClamped, accuracy: 0.01,
            "Value 1e-10 equals amin")

        // 1e-8 is above amin, should be -80 dB
        XCTAssertEqual(dbResult[0][3], -80.0, accuracy: 0.01,
            "Value 1e-8 should be -80 dB")
    }

    /// Test topDb threshold behavior
    func testTopDbThreshold() {
        // Wide range of values
        let values: [[Float]] = [[1e-10, 1e-6, 1e-3, 1.0]]

        // With topDb=40, values more than 40dB below max should be clamped
        let dbResult = Convert.powerToDb(values, ref: 1.0, amin: 1e-10, topDb: 40.0)

        // Max should be 0 dB (for value 1.0)
        // Threshold should be -40 dB
        let maxDb = dbResult[0].max()!
        XCTAssertEqual(maxDb, 0.0, accuracy: 0.01, "Max should be 0 dB")

        let minDb = dbResult[0].min()!
        XCTAssertEqual(minDb, -40.0, accuracy: 0.01, "Min should be clamped to -40 dB")
    }
}
