import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Emphasis (preemphasis/deemphasis) validation tests against librosa reference.
///
/// Validates:
/// - Preemphasis filter output
/// - Deemphasis filter output
/// - Round-trip (preemphasis -> deemphasis) reconstruction
///
/// Reference: librosa 0.11.0
final class EmphasisValidationTests: XCTestCase {

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
                .deletingLastPathComponent()
                .appendingPathComponent("SwiftRosaCoreTests/ReferenceData/librosa_reference.json")
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

    /// Load a test signal from reference data
    private func loadTestSignal(_ name: String) throws -> [Float] {
        let ref = try getReference()
        guard let testSignals = ref["test_signals"] as? [String: Any],
              let signalData = testSignals[name] as? [String: Any],
              let data = signalData["data"] as? [Double] else {
            throw XCTSkip("Test signal '\(name)' not found in reference data")
        }
        return data.map { Float($0) }
    }

    // MARK: - Helper Methods

    /// Compute max absolute error between two arrays
    private func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// Compute mean absolute error between two arrays
    private func meanAbsError(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.infinity }
        return zip(a, b).map { abs($0 - $1) }.reduce(0, +) / Float(a.count)
    }

    // MARK: - Preemphasis Tests

    /// Test preemphasis with coef=0.97 on 440Hz sine
    func testPreemphasis_Sine440Hz_Coef097() async throws {
        let ref = try getReference()
        guard let emphasisRef = ref["emphasis"] as? [String: Any],
              let sineRef = emphasisRef["sine_440hz"] as? [String: Any],
              let coef097 = sineRef["coef_0.97"] as? [String: Any],
              let expectedPreemph = coef097["preemphasis"] as? [Double] else {
            throw XCTSkip("Emphasis sine_440hz coef_0.97 preemphasis reference not found")
        }

        // Load exact test signal from reference data
        let signal = try loadTestSignal("sine_440hz")

        // Apply preemphasis
        let preemph = Emphasis.preemphasis(signal, coef: 0.97)

        // Compare with reference
        let expected = expectedPreemph.map { Float($0) }
        let maxErr = maxAbsError(preemph, expected)
        let meanErr = meanAbsError(preemph, expected)

        print("Preemphasis sine_440hz coef=0.97:")
        print("  Length: \(preemph.count), Expected: \(expected.count)")
        print("  Max error: \(String(format: "%.6f", maxErr))")
        print("  Mean error: \(String(format: "%.6f", meanErr))")

        // Tight tolerance for this simple filter
        XCTAssertLessThan(maxErr, 1e-5, "Preemphasis max error too high")
        XCTAssertLessThan(meanErr, 1e-6, "Preemphasis mean error too high")
    }

    /// Test preemphasis with coef=0.95 on multi-tone
    func testPreemphasis_MultiTone_Coef095() async throws {
        let ref = try getReference()
        guard let emphasisRef = ref["emphasis"] as? [String: Any],
              let multiRef = emphasisRef["multi_tone"] as? [String: Any],
              let coef095 = multiRef["coef_0.95"] as? [String: Any],
              let expectedPreemph = coef095["preemphasis"] as? [Double] else {
            throw XCTSkip("Emphasis multi_tone coef_0.95 preemphasis reference not found")
        }

        // Load exact test signal from reference data
        let signal = try loadTestSignal("multi_tone")

        // Apply preemphasis
        let preemph = Emphasis.preemphasis(signal, coef: 0.95)

        // Compare with reference
        let expected = expectedPreemph.map { Float($0) }
        let maxErr = maxAbsError(preemph, expected)
        let meanErr = meanAbsError(preemph, expected)

        print("Preemphasis multi_tone coef=0.95:")
        print("  Length: \(preemph.count), Expected: \(expected.count)")
        print("  Max error: \(String(format: "%.6f", maxErr))")
        print("  Mean error: \(String(format: "%.6f", meanErr))")

        XCTAssertLessThan(maxErr, 1e-5, "Preemphasis max error too high")
        XCTAssertLessThan(meanErr, 1e-6, "Preemphasis mean error too high")
    }

    // MARK: - Deemphasis Tests

    /// Test deemphasis with coef=0.97 on 440Hz sine
    func testDeemphasis_Sine440Hz_Coef097() async throws {
        let ref = try getReference()
        guard let emphasisRef = ref["emphasis"] as? [String: Any],
              let sineRef = emphasisRef["sine_440hz"] as? [String: Any],
              let coef097 = sineRef["coef_0.97"] as? [String: Any],
              let expectedDeemph = coef097["deemphasis"] as? [Double] else {
            throw XCTSkip("Emphasis sine_440hz coef_0.97 deemphasis reference not found")
        }

        // Load exact test signal from reference data
        let signal = try loadTestSignal("sine_440hz")

        // Apply deemphasis
        let deemph = Emphasis.deemphasis(signal, coef: 0.97)

        // Compare with reference
        let expected = expectedDeemph.map { Float($0) }
        let maxErr = maxAbsError(deemph, expected)
        let meanErr = meanAbsError(deemph, expected)

        print("Deemphasis sine_440hz coef=0.97:")
        print("  Length: \(deemph.count), Expected: \(expected.count)")
        print("  Max error: \(String(format: "%.6f", maxErr))")
        print("  Mean error: \(String(format: "%.6f", meanErr))")

        // Tight tolerance for this simple filter
        XCTAssertLessThan(maxErr, 1e-5, "Deemphasis max error too high")
        XCTAssertLessThan(meanErr, 1e-6, "Deemphasis mean error too high")
    }

    // MARK: - Round-trip Tests

    /// Test that preemphasis -> deemphasis recovers original signal
    func testRoundtrip_Sine440Hz() async throws {
        let ref = try getReference()
        guard let emphasisRef = ref["emphasis"] as? [String: Any],
              let sineRef = emphasisRef["sine_440hz"] as? [String: Any],
              let coef097 = sineRef["coef_0.97"] as? [String: Any],
              let expectedRoundtripError = coef097["roundtrip_error"] as? Double else {
            throw XCTSkip("Emphasis sine_440hz coef_0.97 roundtrip reference not found")
        }

        // Load exact test signal from reference data
        let signal = try loadTestSignal("sine_440hz")

        // Apply preemphasis then deemphasis
        let preemph = Emphasis.preemphasis(signal, coef: 0.97)
        let roundtrip = Emphasis.deemphasis(preemph, coef: 0.97)

        // Check reconstruction error
        let maxErr = maxAbsError(roundtrip, signal)

        print("Roundtrip sine_440hz coef=0.97:")
        print("  Max reconstruction error: \(String(format: "%.6f", maxErr))")
        print("  Expected (librosa): \(String(format: "%.6f", expectedRoundtripError))")

        // Round-trip should be near-perfect
        XCTAssertLessThan(maxErr, 1e-5, "Round-trip reconstruction error too high")
    }

    /// Test round-trip on multi-tone
    func testRoundtrip_MultiTone() async throws {
        // Load exact test signal from reference data
        let signal = try loadTestSignal("multi_tone")

        // Apply preemphasis then deemphasis with different coefficients
        for coef: Float in [0.95, 0.97] {
            let preemph = Emphasis.preemphasis(signal, coef: coef)
            let roundtrip = Emphasis.deemphasis(preemph, coef: coef)

            let maxErr = maxAbsError(roundtrip, signal)
            print("Roundtrip multi_tone coef=\(coef):")
            print("  Max reconstruction error: \(String(format: "%.6f", maxErr))")

            XCTAssertLessThan(maxErr, 1e-5, "Round-trip reconstruction error too high for coef=\(coef)")
        }
    }

    // MARK: - Energy Tests

    /// Test that preemphasis boosts high frequencies (energy changes correctly)
    func testPreemphasis_EnergyChange() async throws {
        let ref = try getReference()
        guard let emphasisRef = ref["emphasis"] as? [String: Any],
              let sineRef = emphasisRef["sine_440hz"] as? [String: Any],
              let coef097 = sineRef["coef_0.97"] as? [String: Any],
              let expectedPreemphEnergy = coef097["preemphasis_energy"] as? Double else {
            throw XCTSkip("Emphasis sine_440hz coef_0.97 energy reference not found")
        }

        // Load exact test signal from reference data
        let signal = try loadTestSignal("sine_440hz")

        // Apply preemphasis
        let preemph = Emphasis.preemphasis(signal, coef: 0.97)

        // Compute energies
        let preemphEnergy = preemph.reduce(0) { $0 + $1 * $1 }

        print("Preemphasis energy (sine_440hz, coef=0.97):")
        print("  Swift energy: \(String(format: "%.2f", preemphEnergy))")
        print("  Expected (librosa): \(String(format: "%.2f", expectedPreemphEnergy))")

        // Energy should match within 1%
        let energyRatio = Double(preemphEnergy) / expectedPreemphEnergy
        XCTAssertGreaterThan(energyRatio, 0.99, "Preemphasis energy too low")
        XCTAssertLessThan(energyRatio, 1.01, "Preemphasis energy too high")
    }
}
