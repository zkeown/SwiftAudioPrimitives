import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Time stretch validation tests against librosa reference.
///
/// Validates PhaseVocoder.timeStretch() output against librosa.effects.time_stretch() for:
/// - Various stretch rates (0.5, 0.8, 1.25, 2.0)
/// - Output length, energy ratio, and sample correlation
///
/// Reference: librosa 0.11.0
final class TimeStretchValidationTests: XCTestCase {

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

    /// Compute total energy of a signal (sum of squares)
    private func totalEnergy(_ signal: [Float]) -> Float {
        signal.reduce(0) { $0 + $1 * $1 }
    }

    /// Compute correlation between two arrays
    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let minLen = min(a.count, b.count)
        guard minLen > 0 else { return 0 }

        let aSlice = Array(a.prefix(minLen))
        let bSlice = Array(b.prefix(minLen))

        let n = Float(minLen)
        let meanA = aSlice.reduce(0, +) / n
        let meanB = bSlice.reduce(0, +) / n

        var sumAB: Float = 0
        var sumA2: Float = 0
        var sumB2: Float = 0

        for i in 0..<minLen {
            let da = aSlice[i] - meanA
            let db = bSlice[i] - meanB
            sumAB += da * db
            sumA2 += da * da
            sumB2 += db * db
        }

        let denom = sqrt(sumA2 * sumB2)
        return denom > 0 ? sumAB / denom : 0
    }

    // MARK: - Time Stretch Tests (Sine 440Hz)

    /// Test time stretch with rate=0.5 (slow down by 2x)
    func testTimeStretch_Sine440Hz_Rate0_5() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let sineRef = tsRef["sine_440hz"] as? [String: Any],
              let rateRef = sineRef["rate_0.5"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double,
              let expectedFirst500 = rateRef["first_500_samples"] as? [Double] else {
            throw XCTSkip("time_stretch sine_440hz rate_0.5 reference not found")
        }

        // Load test signal
        let signal = try loadTestSignal("sine_440hz")

        // Apply time stretch
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 0.5)

        // Check length
        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch sine_440hz rate=0.5:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        // Check energy ratio
        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio))")

        // Energy ratio should match within 20%
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")

        // Check first 500 samples correlation
        let expected = expectedFirst500.map { Float($0) }
        let actual = Array(stretched.prefix(500))
        let corr = correlation(actual, expected)
        print("  First 500 samples correlation: \(String(format: "%.4f", corr))")
        XCTAssertGreaterThan(corr, 0.9, "First 500 samples correlation too low")
    }

    /// Test time stretch with rate=0.8 (slow down by 25%)
    func testTimeStretch_Sine440Hz_Rate0_8() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let sineRef = tsRef["sine_440hz"] as? [String: Any],
              let rateRef = sineRef["rate_0.8"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double else {
            throw XCTSkip("time_stretch sine_440hz rate_0.8 reference not found")
        }

        let signal = try loadTestSignal("sine_440hz")
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 0.8)

        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch sine_440hz rate=0.8:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }

    /// Test time stretch with rate=1.25 (speed up by 25%)
    func testTimeStretch_Sine440Hz_Rate1_25() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let sineRef = tsRef["sine_440hz"] as? [String: Any],
              let rateRef = sineRef["rate_1.25"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double else {
            throw XCTSkip("time_stretch sine_440hz rate_1.25 reference not found")
        }

        let signal = try loadTestSignal("sine_440hz")
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 1.25)

        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch sine_440hz rate=1.25:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }

    /// Test time stretch with rate=2.0 (speed up by 2x)
    func testTimeStretch_Sine440Hz_Rate2_0() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let sineRef = tsRef["sine_440hz"] as? [String: Any],
              let rateRef = sineRef["rate_2.0"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double else {
            throw XCTSkip("time_stretch sine_440hz rate_2.0 reference not found")
        }

        let signal = try loadTestSignal("sine_440hz")
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 2.0)

        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch sine_440hz rate=2.0:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }

    // MARK: - Time Stretch Tests (Multi-tone)

    /// Test time stretch with rate=0.5 on multi-tone signal
    func testTimeStretch_MultiTone_Rate0_5() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let multiRef = tsRef["multi_tone"] as? [String: Any],
              let rateRef = multiRef["rate_0.5"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double else {
            throw XCTSkip("time_stretch multi_tone rate_0.5 reference not found")
        }

        let signal = try loadTestSignal("multi_tone")
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 0.5)

        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch multi_tone rate=0.5:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }

    /// Test time stretch with rate=2.0 on multi-tone signal
    func testTimeStretch_MultiTone_Rate2_0() async throws {
        let ref = try getReference()
        guard let tsRef = ref["time_stretch"] as? [String: Any],
              let multiRef = tsRef["multi_tone"] as? [String: Any],
              let rateRef = multiRef["rate_2.0"] as? [String: Any],
              let expectedLength = rateRef["stretched_length"] as? Int,
              let expectedEnergyRatio = rateRef["energy_ratio"] as? Double else {
            throw XCTSkip("time_stretch multi_tone rate_2.0 reference not found")
        }

        let signal = try loadTestSignal("multi_tone")
        let vocoder = PhaseVocoder()
        let stretched = await vocoder.timeStretch(signal, rate: 2.0)

        let lengthRatio = Float(stretched.count) / Float(expectedLength)
        print("TimeStretch multi_tone rate=2.0:")
        print("  Length: \(stretched.count), Expected: \(expectedLength), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output length ratio too small")
        XCTAssertLessThan(lengthRatio, 1.05, "Output length ratio too large")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(stretched)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }
}
