import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Pitch shift validation tests against librosa reference.
///
/// Validates PhaseVocoder.pitchShift() output against librosa.effects.pitch_shift() for:
/// - Various semitone shifts (2, 4, -3)
/// - Output length and energy ratio
///
/// Reference: librosa 0.11.0
final class PitchShiftValidationTests: XCTestCase {

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

    // MARK: - Pitch Shift Tests (Sine 440Hz)

    /// Test pitch shift up by 2 semitones
    func testPitchShift_Sine440Hz_Steps2() async throws {
        let ref = try getReference()
        guard let psRef = ref["pitch_shift"] as? [String: Any],
              let sineRef = psRef["sine_440hz"] as? [String: Any],
              let stepsRef = sineRef["steps_2"] as? [String: Any],
              let expectedLength = stepsRef["shifted_length"] as? Int,
              let expectedEnergyRatio = stepsRef["energy_ratio"] as? Double else {
            throw XCTSkip("pitch_shift sine_440hz steps_2 reference not found")
        }

        // Load test signal
        let signal = try loadTestSignal("sine_440hz")

        // Apply pitch shift
        let vocoder = PhaseVocoder()
        let shifted = await vocoder.pitchShift(signal, semitones: 2, sampleRate: 22050)

        // Check length (should match original since pitch shift preserves duration)
        let lengthRatio = Float(shifted.count) / Float(signal.count)
        print("PitchShift sine_440hz steps=2:")
        print("  Length: \(shifted.count), Original: \(signal.count), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output should be approximately same length as input")
        XCTAssertLessThan(lengthRatio, 1.05, "Output should be approximately same length as input")

        // Check energy ratio
        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(shifted)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")

        // Energy ratio should match within 20%
        XCTAssertLessThan(energyDiff, 0.2, "Energy ratio differs by more than 20%")
    }

    /// Test pitch shift up by 4 semitones
    func testPitchShift_Sine440Hz_Steps4() async throws {
        let ref = try getReference()
        guard let psRef = ref["pitch_shift"] as? [String: Any],
              let sineRef = psRef["sine_440hz"] as? [String: Any],
              let stepsRef = sineRef["steps_4"] as? [String: Any],
              let expectedEnergyRatio = stepsRef["energy_ratio"] as? Double else {
            throw XCTSkip("pitch_shift sine_440hz steps_4 reference not found")
        }

        let signal = try loadTestSignal("sine_440hz")
        let vocoder = PhaseVocoder()
        let shifted = await vocoder.pitchShift(signal, semitones: 4, sampleRate: 22050)

        let lengthRatio = Float(shifted.count) / Float(signal.count)
        print("PitchShift sine_440hz steps=4:")
        print("  Length: \(shifted.count), Original: \(signal.count), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output should be approximately same length as input")
        XCTAssertLessThan(lengthRatio, 1.05, "Output should be approximately same length as input")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(shifted)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.3, "Energy ratio differs by more than 30%")
    }

    /// Test pitch shift down by 3 semitones
    func testPitchShift_Sine440Hz_StepsMinus3() async throws {
        let ref = try getReference()
        guard let psRef = ref["pitch_shift"] as? [String: Any],
              let sineRef = psRef["sine_440hz"] as? [String: Any],
              let stepsRef = sineRef["steps_-3"] as? [String: Any],
              let expectedEnergyRatio = stepsRef["energy_ratio"] as? Double else {
            throw XCTSkip("pitch_shift sine_440hz steps_-3 reference not found")
        }

        let signal = try loadTestSignal("sine_440hz")
        let vocoder = PhaseVocoder()
        let shifted = await vocoder.pitchShift(signal, semitones: -3, sampleRate: 22050)

        let lengthRatio = Float(shifted.count) / Float(signal.count)
        print("PitchShift sine_440hz steps=-3:")
        print("  Length: \(shifted.count), Original: \(signal.count), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output should be approximately same length as input")
        XCTAssertLessThan(lengthRatio, 1.05, "Output should be approximately same length as input")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(shifted)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.3, "Energy ratio differs by more than 30%")
    }

    // MARK: - Pitch Shift Tests (Multi-tone)

    /// Test pitch shift up by 2 semitones on multi-tone
    func testPitchShift_MultiTone_Steps2() async throws {
        let ref = try getReference()
        guard let psRef = ref["pitch_shift"] as? [String: Any],
              let multiRef = psRef["multi_tone"] as? [String: Any],
              let stepsRef = multiRef["steps_2"] as? [String: Any],
              let expectedEnergyRatio = stepsRef["energy_ratio"] as? Double else {
            throw XCTSkip("pitch_shift multi_tone steps_2 reference not found")
        }

        let signal = try loadTestSignal("multi_tone")
        let vocoder = PhaseVocoder()
        let shifted = await vocoder.pitchShift(signal, semitones: 2, sampleRate: 22050)

        let lengthRatio = Float(shifted.count) / Float(signal.count)
        print("PitchShift multi_tone steps=2:")
        print("  Length: \(shifted.count), Original: \(signal.count), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output should be approximately same length as input")
        XCTAssertLessThan(lengthRatio, 1.05, "Output should be approximately same length as input")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(shifted)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.3, "Energy ratio differs by more than 30%")
    }

    /// Test pitch shift down by 3 semitones on multi-tone
    func testPitchShift_MultiTone_StepsMinus3() async throws {
        let ref = try getReference()
        guard let psRef = ref["pitch_shift"] as? [String: Any],
              let multiRef = psRef["multi_tone"] as? [String: Any],
              let stepsRef = multiRef["steps_-3"] as? [String: Any],
              let expectedEnergyRatio = stepsRef["energy_ratio"] as? Double else {
            throw XCTSkip("pitch_shift multi_tone steps_-3 reference not found")
        }

        let signal = try loadTestSignal("multi_tone")
        let vocoder = PhaseVocoder()
        let shifted = await vocoder.pitchShift(signal, semitones: -3, sampleRate: 22050)

        let lengthRatio = Float(shifted.count) / Float(signal.count)
        print("PitchShift multi_tone steps=-3:")
        print("  Length: \(shifted.count), Original: \(signal.count), Ratio: \(String(format: "%.3f", lengthRatio))")
        XCTAssertGreaterThan(lengthRatio, 0.95, "Output should be approximately same length as input")
        XCTAssertLessThan(lengthRatio, 1.05, "Output should be approximately same length as input")

        let inputEnergy = totalEnergy(signal)
        let outputEnergy = totalEnergy(shifted)
        let actualEnergyRatio = Double(outputEnergy / inputEnergy)
        let energyDiff = abs(actualEnergyRatio - expectedEnergyRatio) / expectedEnergyRatio
        print("  Energy ratio: \(String(format: "%.4f", actualEnergyRatio)), Expected: \(String(format: "%.4f", expectedEnergyRatio)), Diff: \(String(format: "%.2f%%", energyDiff * 100))")
        XCTAssertLessThan(energyDiff, 0.3, "Energy ratio differs by more than 30%")
    }
}
