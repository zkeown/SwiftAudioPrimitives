import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

// MARK: - Test Signal Helpers

/// Generate a pure sine wave using Double precision internally.
private func generateSine(frequency: Float, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
    let nSamples = Int(Float(sampleRate) * duration)
    let freqD = Double(frequency)
    let srD = Double(sampleRate)
    return (0..<nSamples).map { i in
        let t = Double(i) / srD
        return Float(sin(2 * Double.pi * freqD * t))
    }
}

/// Generate a multi-tone signal (harmonic series) using Double precision internally.
private func generateMultiTone(fundamentalFrequency: Float = 440, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
    let nSamples = Int(Float(sampleRate) * duration)
    let f0 = Double(fundamentalFrequency)
    let srD = Double(sampleRate)
    return (0..<nSamples).map { i in
        let t = Double(i) / srD
        let val = 0.5 * sin(2 * Double.pi * f0 * t) +
                  0.3 * sin(2 * Double.pi * f0 * 2 * t) +
                  0.2 * sin(2 * Double.pi * f0 * 4 * t)
        return Float(val)
    }
}

/// VQT validation tests against librosa reference.
///
/// Validates VQT output against librosa.vqt() for:
/// - 440Hz sine wave with different gamma values
///
/// **KNOWN ISSUE**: VQT inherits CQT magnitude errors (~10-20%) due to
/// algorithmic differences between Swift's time-domain correlation and
/// librosa's FFT-based implementation. See Part 3 of validation plan.
/// These tests currently FAIL and document the issue for proper fix.
///
/// Reference: librosa 0.11.0
final class VQTValidationTests: XCTestCase {

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

    /// Compute relative error between actual and expected arrays.
    /// Only considers values above `threshold` to focus on meaningful signal content.
    private func computeRelativeError(_ actual: [[Float]], _ expected: [[Float]], tolerance: Double, threshold: Double = 0.1) -> (maxError: Double, meanError: Double, passRate: Double) {
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

                // Only consider values above threshold (focus on meaningful signal)
                if abs(exp) < threshold {
                    continue
                }

                let relError = abs(act - exp) / abs(exp)
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

    // MARK: - VQT Tests

    /// Test VQT with gamma=0 (equivalent to CQT) for 440Hz sine
    func testVQT_Sine440Hz_Gamma0() async throws {
        let ref = try getReference()
        guard let vqtRef = ref["vqt"] as? [String: Any],
              let sineRef = vqtRef["sine_440hz"] as? [String: Any],
              let results = sineRef["results"] as? [String: Any],
              let gamma0Ref = results["gamma_0"] as? [String: Any],
              let expectedFlat = gamma0Ref["magnitude"] as? [[Double]],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("VQT sine_440hz gamma_0 reference data not found")
        }

        // Reference is [nFrames, nBins], need [nBins, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Generate test signal
        let signal = generateSine(frequency: 440)

        // Compute VQT with gamma=0 (CQT mode), sparsity=0 for exact match
        let vqt = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave,
            gamma: 0,
            sparsity: 0  // Disable kernel pruning for exact librosa match
        ))

        let actual = await vqt.magnitude(signal)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "VQT row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "VQT column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance - focus on bins with significant energy (>1.0)
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: 0.05, threshold: 1.0)

        print("VQT Sine440Hz gamma=0 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "VQT pass rate \(passRate) below 95%")
        XCTAssertLessThan(maxError, 0.05, "VQT max error \(maxError) exceeds 5%")
        XCTAssertLessThan(meanError, 0.01, "VQT mean error \(meanError) exceeds 1%")
    }

    /// Test VQT with gamma=0.5 for 440Hz sine
    func testVQT_Sine440Hz_Gamma0_5() async throws {
        let ref = try getReference()
        guard let vqtRef = ref["vqt"] as? [String: Any],
              let sineRef = vqtRef["sine_440hz"] as? [String: Any],
              let results = sineRef["results"] as? [String: Any],
              let gammaRef = results["gamma_0.5"] as? [String: Any],
              let expectedFlat = gammaRef["magnitude"] as? [[Double]],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("VQT sine_440hz gamma_0.5 reference data not found")
        }

        // Reference is [nFrames, nBins], need [nBins, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Generate test signal
        let signal = generateSine(frequency: 440)

        // Compute VQT with gamma=0.5, sparsity=0 for exact match
        let vqt = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave,
            gamma: 0.5,
            sparsity: 0  // Disable kernel pruning for exact librosa match
        ))

        let actual = await vqt.magnitude(signal)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "VQT row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "VQT column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance - focus on bins with significant energy (>1.0)
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: 0.05, threshold: 1.0)

        print("VQT Sine440Hz gamma=0.5 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "VQT pass rate \(passRate) below 95%")
        XCTAssertLessThan(maxError, 0.05, "VQT max error \(maxError) exceeds 5%")
        XCTAssertLessThan(meanError, 0.01, "VQT mean error \(meanError) exceeds 1%")
    }

    /// Test VQT with gamma=1.0 for 440Hz sine
    func testVQT_Sine440Hz_Gamma1_0() async throws {
        let ref = try getReference()
        guard let vqtRef = ref["vqt"] as? [String: Any],
              let sineRef = vqtRef["sine_440hz"] as? [String: Any],
              let results = sineRef["results"] as? [String: Any],
              let gammaRef = results["gamma_1.0"] as? [String: Any],
              let expectedFlat = gammaRef["magnitude"] as? [[Double]],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("VQT sine_440hz gamma_1.0 reference data not found")
        }

        // Reference is [nFrames, nBins], need [nBins, nFrames]
        let expectedTimeFeature = expectedFlat.map { $0.map { Float($0) } }
        let expected = transpose(expectedTimeFeature)

        // Generate test signal
        let signal = generateSine(frequency: 440)

        // Compute VQT with gamma=1.0, sparsity=0 for exact match
        let vqt = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave,
            gamma: 1.0,
            sparsity: 0  // Disable kernel pruning for exact librosa match
        ))

        let actual = await vqt.magnitude(signal)

        // Verify shapes match
        XCTAssertEqual(actual.count, expected.count, "VQT row count mismatch: got \(actual.count), expected \(expected.count)")
        if !actual.isEmpty && !expected.isEmpty {
            XCTAssertEqual(actual[0].count, expected[0].count, "VQT column count mismatch: got \(actual[0].count), expected \(expected[0].count)")
        }

        // Compare with tolerance - focus on bins with significant energy (>1.0)
        let (maxError, meanError, passRate) = computeRelativeError(actual, expected, tolerance: 0.05, threshold: 1.0)

        print("VQT Sine440Hz gamma=1.0 - Max error: \(String(format: "%.4e", maxError)), Mean error: \(String(format: "%.4e", meanError)), Pass rate: \(String(format: "%.1f%%", passRate * 100))")

        XCTAssertGreaterThan(passRate, 0.95, "VQT pass rate \(passRate) below 95%")
        XCTAssertLessThan(maxError, 0.05, "VQT max error \(maxError) exceeds 5%")
        XCTAssertLessThan(meanError, 0.01, "VQT mean error \(meanError) exceeds 1%")
    }
}
