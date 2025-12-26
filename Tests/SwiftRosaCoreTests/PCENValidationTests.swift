import XCTest
@testable import SwiftRosaCore

/// PCEN (Per-Channel Energy Normalization) validation tests against librosa reference.
///
/// Validates PCEN output against librosa.pcen() with:
/// - Default parameters (gain=0.98, bias=2.0, power=0.5, time_constant=0.4, eps=1e-6)
/// - Custom parameters (gain=0.8, bias=10, power=0.25, time_constant=0.06, eps=1e-6)
///
/// Reference: librosa 0.11.0
final class PCENValidationTests: XCTestCase {

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

    // MARK: - PCEN Tests with Default Parameters

    /// Test PCEN for 440Hz sine with default librosa parameters
    func testPCEN_Sine440Hz_DefaultParams() async throws {
        let ref = try getReference()
        guard let pcenRef = ref["pcen"] as? [String: Any],
              let sineRef = pcenRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["pcen_default"] as? [[Double]],
              let defaultParams = sineRef["default_params"] as? [String: Any],
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int,
              let nMels = sineRef["n_mels"] as? Int else {
            throw XCTSkip("PCEN sine_440hz reference data not found")
        }

        // Extract default parameters from reference data
        guard let gain = defaultParams["gain"] as? Double,
              let bias = defaultParams["bias"] as? Double,
              let power = defaultParams["power"] as? Double,
              let timeConstant = defaultParams["time_constant"] as? Double,
              let eps = defaultParams["eps"] as? Double else {
            throw XCTSkip("PCEN default parameters not found")
        }

        // Convert reference data to Float and transpose from [nFrames, nMels] to [nMels, nFrames]
        let expectedTimeFreq = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFreq)

        // Generate test signal
        let signal = TestSignalGenerator.sine(frequency: 440)

        // Compute mel spectrogram
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let melOutput = await melSpec.transform(signal)

        // Apply PCEN with default parameters from reference data
        let pcen = PCEN(config: PCENConfig(
            gain: Float(gain),
            bias: Float(bias),
            power: Float(power),
            timeConstant: Float(timeConstant),
            sampleRate: 22050,
            hopLength: hopLength,
            eps: Float(eps)
        ))
        let actual = pcen.transform(melOutput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PCEN row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PCEN column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.pcen)

        // Report statistics
        print("PCEN Sine440Hz Default - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PCEN pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.pcen, "PCEN mean error \(meanError) exceeds tolerance \(ValidationTolerances.pcen)")
    }

    /// Test PCEN for 440Hz sine with custom parameters
    func testPCEN_Sine440Hz_CustomParams() async throws {
        let ref = try getReference()
        guard let pcenRef = ref["pcen"] as? [String: Any],
              let sineRef = pcenRef["sine_440hz"] as? [String: Any],
              let expectedFlat = sineRef["pcen_custom"] as? [[Double]],
              let customParams = sineRef["custom_params"] as? [String: Any],
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int,
              let nMels = sineRef["n_mels"] as? Int else {
            throw XCTSkip("PCEN sine_440hz custom reference data not found")
        }

        // Extract custom parameters
        guard let gain = customParams["gain"] as? Double,
              let bias = customParams["bias"] as? Double,
              let power = customParams["power"] as? Double,
              let timeConstant = customParams["time_constant"] as? Double,
              let eps = customParams["eps"] as? Double else {
            throw XCTSkip("PCEN custom parameters not found")
        }

        // Convert and transpose reference
        let expectedTimeFreq = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFreq)

        // Generate test signal
        let signal = TestSignalGenerator.sine(frequency: 440)

        // Compute mel spectrogram
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let melOutput = await melSpec.transform(signal)

        // Apply PCEN with custom parameters
        let pcen = PCEN(config: PCENConfig(
            gain: Float(gain),
            bias: Float(bias),
            power: Float(power),
            timeConstant: Float(timeConstant),
            sampleRate: 22050,
            hopLength: hopLength,
            eps: Float(eps)
        ))
        let actual = pcen.transform(melOutput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PCEN row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PCEN column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.pcen)

        print("PCEN Sine440Hz Custom - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PCEN pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.pcen, "PCEN mean error \(meanError) exceeds tolerance \(ValidationTolerances.pcen)")
    }

    /// Test PCEN for multi-tone signal with default parameters
    func testPCEN_MultiTone_DefaultParams() async throws {
        let ref = try getReference()
        guard let pcenRef = ref["pcen"] as? [String: Any],
              let multiRef = pcenRef["multi_tone"] as? [String: Any],
              let expectedFlat = multiRef["pcen_default"] as? [[Double]],
              let defaultParams = multiRef["default_params"] as? [String: Any],
              let nFFT = multiRef["n_fft"] as? Int,
              let hopLength = multiRef["hop_length"] as? Int,
              let nMels = multiRef["n_mels"] as? Int else {
            throw XCTSkip("PCEN multi_tone reference data not found")
        }

        // Extract default parameters from reference data
        guard let gain = defaultParams["gain"] as? Double,
              let bias = defaultParams["bias"] as? Double,
              let power = defaultParams["power"] as? Double,
              let timeConstant = defaultParams["time_constant"] as? Double,
              let eps = defaultParams["eps"] as? Double else {
            throw XCTSkip("PCEN default parameters not found")
        }

        // Convert and transpose reference
        let expectedTimeFreq = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFreq)

        // Generate test signal
        let signal = TestSignalGenerator.multiTone()

        // Compute mel spectrogram
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let melOutput = await melSpec.transform(signal)

        // Apply PCEN with default parameters from reference data
        let pcen = PCEN(config: PCENConfig(
            gain: Float(gain),
            bias: Float(bias),
            power: Float(power),
            timeConstant: Float(timeConstant),
            sampleRate: 22050,
            hopLength: hopLength,
            eps: Float(eps)
        ))
        let actual = pcen.transform(melOutput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PCEN row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PCEN column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.pcen)

        print("PCEN MultiTone Default - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PCEN pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.pcen, "PCEN mean error \(meanError) exceeds tolerance \(ValidationTolerances.pcen)")
    }

    /// Test PCEN for chirp signal with default parameters
    func testPCEN_Chirp_DefaultParams() async throws {
        let ref = try getReference()
        guard let pcenRef = ref["pcen"] as? [String: Any],
              let chirpRef = pcenRef["chirp"] as? [String: Any],
              let expectedFlat = chirpRef["pcen_default"] as? [[Double]],
              let defaultParams = chirpRef["default_params"] as? [String: Any],
              let nFFT = chirpRef["n_fft"] as? Int,
              let hopLength = chirpRef["hop_length"] as? Int,
              let nMels = chirpRef["n_mels"] as? Int else {
            throw XCTSkip("PCEN chirp reference data not found")
        }

        // Extract default parameters from reference data
        guard let gain = defaultParams["gain"] as? Double,
              let bias = defaultParams["bias"] as? Double,
              let power = defaultParams["power"] as? Double,
              let timeConstant = defaultParams["time_constant"] as? Double,
              let eps = defaultParams["eps"] as? Double else {
            throw XCTSkip("PCEN default parameters not found")
        }

        // Convert and transpose reference
        let expectedTimeFreq = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFreq)

        // Generate test signal
        let signal = TestSignalGenerator.chirp()

        // Compute mel spectrogram
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let melOutput = await melSpec.transform(signal)

        // Apply PCEN with default parameters from reference data
        let pcen = PCEN(config: PCENConfig(
            gain: Float(gain),
            bias: Float(bias),
            power: Float(power),
            timeConstant: Float(timeConstant),
            sampleRate: 22050,
            hopLength: hopLength,
            eps: Float(eps)
        ))
        let actual = pcen.transform(melOutput)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "PCEN row count mismatch")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "PCEN column count mismatch")
        }

        // Compare with tolerance
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: ValidationTolerances.pcen)

        print("PCEN Chirp Default - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "PCEN pass rate \(passRate) below 95%")
        XCTAssertLessThan(meanError, ValidationTolerances.pcen, "PCEN mean error \(meanError) exceeds tolerance \(ValidationTolerances.pcen)")
    }

    // MARK: - Streaming State Management

    /// Test PCEN streaming with state preservation
    func testPCEN_StreamingStateManagement() async throws {
        // Generate a longer signal
        let fullSignal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 2.0)

        // Split into two chunks
        let midpoint = fullSignal.count / 2
        let chunk1 = Array(fullSignal[..<midpoint])
        let chunk2 = Array(fullSignal[midpoint...])

        // Compute mel spectrogram for each chunk
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )

        let mel1 = await melSpec.transform(chunk1)
        let mel2 = await melSpec.transform(chunk2)

        // Apply PCEN in streaming mode
        let pcen = PCEN(config: PCENConfig(
            gain: 0.98,
            bias: 2.0,
            power: 0.5,
            timeConstant: 0.025,
            sampleRate: 22050,
            hopLength: 512
        ))

        // Process first chunk and get state
        let (output1, finalM) = pcen.transformWithState(mel1)

        // Process second chunk with preserved state
        let output2 = pcen.transform(mel2, initialM: finalM)

        // Verify outputs are valid
        XCTAssertFalse(output1.isEmpty, "First chunk output should not be empty")
        XCTAssertFalse(output2.isEmpty, "Second chunk output should not be empty")
        XCTAssertEqual(finalM.count, 128, "Final state should have 128 mel bands")

        // Verify state values are reasonable (positive, finite)
        for m in finalM {
            XCTAssertTrue(m.isFinite, "State value should be finite")
            XCTAssertGreaterThanOrEqual(m, 0, "State value should be non-negative")
        }

        // Also compute non-streaming result for comparison
        let fullMel = await melSpec.transform(fullSignal)
        let fullOutput = pcen.transform(fullMel)

        // The streaming approach may have slight boundary effects, but outputs should be similar in shape
        XCTAssertEqual(output1.count, fullOutput.count, "Streaming output should have same number of mel bands")
    }

    // MARK: - Edge Cases

    /// Test PCEN with empty input
    func testPCEN_EmptyInput() {
        let pcen = PCEN.standard
        let result = pcen.transform([])

        XCTAssertTrue(result.isEmpty, "Empty input should produce empty output")
    }

    /// Test PCEN with single frame
    func testPCEN_SingleFrame() {
        let pcen = PCEN.standard

        // Single frame with 128 mel bands
        let singleFrame: [[Float]] = (0..<128).map { _ in [Float.random(in: 0..<1)] }
        let result = pcen.transform(singleFrame)

        XCTAssertEqual(result.count, 128, "Single frame should preserve mel band count")
        XCTAssertEqual(result[0].count, 1, "Single frame should preserve frame count")

        // All values should be finite
        for row in result {
            for value in row {
                XCTAssertTrue(value.isFinite, "PCEN output should be finite")
            }
        }
    }
}
