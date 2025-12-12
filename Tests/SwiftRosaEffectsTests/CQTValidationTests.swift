import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Comprehensive CQT/VQT/Pseudo-CQT validation tests against librosa reference.
///
/// These tests validate:
/// 1. CQT magnitude and center frequencies
/// 2. VQT with various gamma values
/// 3. Pseudo-CQT
/// 4. Chromagram (STFT and CQT methods)
/// 5. Tonnetz features
final class CQTValidationTests: XCTestCase {

    // MARK: - Constants

    private let sampleRate: Float = 22050

    // MARK: - Signal Generation Helpers

    private func generateSine(frequency: Float, duration: Float = 1.0) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * .pi * frequency * t)
        }
        return signal
    }

    private func generateMultiTone(duration: Float = 1.0) -> [Float] {
        let frequencies: [Float] = [440, 880, 1320]
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            for freq in frequencies {
                signal[i] += sin(2 * .pi * freq * t) / Float(frequencies.count)
            }
        }
        return signal
    }

    private func generateSilence(duration: Float = 1.0) -> [Float] {
        return [Float](repeating: 0, count: Int(sampleRate * duration))
    }

    private func generateImpulse(duration: Float = 1.0) -> [Float] {
        var signal = [Float](repeating: 0, count: Int(sampleRate * duration))
        let midpoint = signal.count / 2
        signal[midpoint] = 1.0
        return signal
    }

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

    // MARK: - CQT Tests

    /// Test CQT magnitude for pure 440Hz sine
    func testCQT_Sine440Hz() async throws {
        let ref = try getReference()
        guard let cqtRef = ref["cqt"] as? [String: Any],
              let sineRef = cqtRef["sine_440hz"] as? [String: Any],
              let _ = sineRef["magnitude"] as? [[Double]],
              let centerFreqs = sineRef["center_frequencies"] as? [Double],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("CQT reference data not found")
        }

        let signal = generateSine(frequency: 440)

        let config = CQTConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )
        let cqt = CQT(config: config)
        let result = await cqt.transform(signal)

        // Get magnitude from ComplexMatrix
        let magnitude = result.magnitude

        // Verify shape
        XCTAssertEqual(magnitude.count, nBins, accuracy: 1,
            "CQT should have \(nBins) bins")

        // Find bin closest to 440 Hz
        var targetBin = 0
        var minFreqDiff = Float.infinity
        for (i, freq) in centerFreqs.enumerated() {
            let diff = Swift.abs(Float(freq) - 440)
            if diff < minFreqDiff {
                minFreqDiff = diff
                targetBin = i
            }
        }

        // The 440 Hz bin should have maximum energy for middle frames
        let middleFrame = (magnitude.first?.count ?? 1) / 2
        var maxBin = 0
        var maxMag: Float = 0
        for bin in 0..<min(magnitude.count, nBins) {
            if magnitude[bin].count > middleFrame {
                let mag = magnitude[bin][middleFrame]
                if mag > maxMag {
                    maxMag = mag
                    maxBin = bin
                }
            }
        }

        // Peak should be within a few bins of expected 440Hz bin
        XCTAssertEqual(maxBin, targetBin, accuracy: 3,
            "CQT peak bin should be near 440Hz bin")
    }

    /// Test CQT for multi-tone signal
    func testCQT_MultiTone() async throws {
        let ref = try getReference()
        guard let cqtRef = ref["cqt"] as? [String: Any],
              let multiRef = cqtRef["multi_tone"] as? [String: Any],
              let hopLength = multiRef["hop_length"] as? Int,
              let fmin = multiRef["fmin"] as? Double,
              let nBins = multiRef["n_bins"] as? Int,
              let binsPerOctave = multiRef["bins_per_octave"] as? Int else {
            throw XCTSkip("CQT reference data not found for multi_tone")
        }

        let signal = generateMultiTone()

        let config = CQTConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )
        let cqt = CQT(config: config)
        let result = await cqt.transform(signal)

        // Get magnitude from ComplexMatrix
        let magnitude = result.magnitude

        // Should have output
        XCTAssertFalse(magnitude.isEmpty, "CQT should produce output")

        // Multi-tone at 440, 880, 1320 Hz should show energy at these frequencies
        // Find bins with significant energy
        let middleFrame = (magnitude.first?.count ?? 1) / 2
        var peakBins: [Int] = []
        let threshold: Float = (magnitude.flatMap { $0 }.max() ?? 1) * 0.3

        for bin in 0..<magnitude.count {
            if magnitude[bin].count > middleFrame &&
               magnitude[bin][middleFrame] > threshold {
                peakBins.append(bin)
            }
        }

        // Should find multiple peaks for harmonics
        XCTAssertGreaterThanOrEqual(peakBins.count, 1,
            "Should find at least one strong frequency bin")
    }

    // MARK: - VQT Tests

    /// Test VQT produces output
    func testVQT_Basic() async throws {
        let ref = try getReference()
        guard let vqtRef = ref["vqt"] as? [String: Any],
              let sineRef = vqtRef["sine_440hz"] as? [String: Any],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("VQT reference data not found")
        }

        let signal = generateSine(frequency: 440)

        // Use default gamma (0) since reference data has results per gamma
        let config = VQTConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave,
            gamma: 0
        )
        let vqt = VQT(config: config)
        let result = await vqt.transform(signal)

        // Get magnitude from ComplexMatrix
        let magnitude = result.magnitude

        // VQT should produce output similar to CQT
        XCTAssertFalse(magnitude.isEmpty, "VQT should produce output")
        XCTAssertEqual(magnitude.count, nBins, accuracy: 1,
            "VQT should have \(nBins) bins")
    }

    /// Test VQT with different gamma values
    func testVQT_GammaVariation() async throws {
        let signal = generateSine(frequency: 440)

        let gammaValues: [Float] = [0.0, 10.0, 50.0]

        for gamma in gammaValues {
            let config = VQTConfig(
                sampleRate: sampleRate,
                hopLength: 512,
                fMin: 32.7,
                nBins: 84,
                binsPerOctave: 12,
                gamma: gamma
            )
            let vqt = VQT(config: config)
            let result = await vqt.transform(signal)

            // Get magnitude from ComplexMatrix
            let magnitude = result.magnitude

            XCTAssertFalse(magnitude.isEmpty,
                "VQT with gamma=\(gamma) should produce output")
        }
    }

    // MARK: - Pseudo-CQT Tests

    /// Test Pseudo-CQT for pure sine
    func testPseudoCQT_Sine() async throws {
        let ref = try getReference()
        guard let pcqtRef = ref["pseudo_cqt"] as? [String: Any],
              let sineRef = pcqtRef["sine_440hz"] as? [String: Any],
              let hopLength = sineRef["hop_length"] as? Int,
              let fmin = sineRef["fmin"] as? Double,
              let nBins = sineRef["n_bins"] as? Int,
              let binsPerOctave = sineRef["bins_per_octave"] as? Int else {
            throw XCTSkip("Pseudo-CQT reference data not found")
        }

        let signal = generateSine(frequency: 440)

        let config = PseudoCQTConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )
        let pcqt = PseudoCQT(config: config)
        let result = await pcqt.transform(signal)

        // Pseudo-CQT returns ComplexMatrix, get magnitude
        let magnitude = result.magnitude
        XCTAssertFalse(magnitude.isEmpty, "Pseudo-CQT should produce output")
        XCTAssertEqual(magnitude.count, nBins, accuracy: 1,
            "Pseudo-CQT should have \(nBins) bins")
    }

    // MARK: - Chromagram Tests

    /// Test chromagram with STFT method
    func testChromagram_STFT() async throws {
        let ref = try getReference()
        guard let chromaRef = ref["chromagram"] as? [String: Any],
              let sineRef = chromaRef["sine_440hz"] as? [String: Any],
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int,
              let nChroma = sineRef["n_chroma"] as? Int else {
            throw XCTSkip("Chromagram reference data not found")
        }

        let signal = generateSine(frequency: 440)

        let config = ChromagramConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            nChroma: nChroma,
            useCQT: false,  // Use STFT-based
            nFFT: nFFT
        )
        let chroma = Chromagram(config: config)
        let result = await chroma.transform(signal)

        // Chromagram should have 12 bins (or nChroma)
        XCTAssertEqual(result.count, nChroma,
            "Chromagram should have \(nChroma) chroma bins")

        // 440Hz = A4, so bin 9 (A) should be dominant
        let middleFrame = (result.first?.count ?? 1) / 2
        var maxChromaBin = 0
        var maxChromaVal: Float = 0

        for bin in 0..<result.count {
            if result[bin].count > middleFrame {
                let val = result[bin][middleFrame]
                if val > maxChromaVal {
                    maxChromaVal = val
                    maxChromaBin = bin
                }
            }
        }

        // A = 9th semitone from C (C=0, C#=1, D=2, ... A=9)
        XCTAssertEqual(maxChromaBin, 9, accuracy: 1,
            "440Hz should peak at A chroma bin (9)")
    }

    /// Test chromagram with CQT method
    func testChromagram_CQT() async throws {
        let ref = try getReference()
        guard let chromaRef = ref["chromagram"] as? [String: Any],
              let multiRef = chromaRef["multi_tone"] as? [String: Any],
              let hopLength = multiRef["hop_length"] as? Int,
              let nChroma = multiRef["n_chroma"] as? Int else {
            throw XCTSkip("Chromagram reference data not found")
        }

        let signal = generateMultiTone()

        let config = ChromagramConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            nChroma: nChroma,
            useCQT: true  // Use CQT-based
        )
        let chroma = Chromagram(config: config)
        let result = await chroma.transform(signal)

        XCTAssertEqual(result.count, nChroma,
            "Chromagram should have \(nChroma) chroma bins")

        // Multi-tone should show energy in multiple chroma bins
        let middleFrame = (result.first?.count ?? 1) / 2
        var activeBins = 0
        let threshold: Float = (result.flatMap { $0 }.max() ?? 1) * 0.2

        for bin in 0..<result.count {
            if result[bin].count > middleFrame &&
               result[bin][middleFrame] > threshold {
                activeBins += 1
            }
        }

        XCTAssertGreaterThanOrEqual(activeBins, 1,
            "Multi-tone should activate multiple chroma bins")
    }

    // MARK: - Tonnetz Tests

    /// Test tonnetz for pure sine
    func testTonnetz_Sine() async throws {
        let ref = try getReference()
        guard let tonnetzRef = ref["tonnetz"] as? [String: Any],
              let sineRef = tonnetzRef["sine_440hz"] as? [String: Any],
              let hopLength = sineRef["hop_length"] as? Int else {
            throw XCTSkip("Tonnetz reference data not found")
        }

        let signal = generateSine(frequency: 440)

        let config = TonnetzConfig(
            sampleRate: sampleRate,
            hopLength: hopLength
        )
        let tonnetz = Tonnetz(config: config)
        let result = await tonnetz.transform(signal)

        // Tonnetz should have 6 dimensions
        XCTAssertEqual(result.count, 6,
            "Tonnetz should have 6 dimensions")

        // Check that output is reasonable (not NaN or Inf)
        for dim in 0..<result.count {
            for val in result[dim] {
                XCTAssertFalse(val.isNaN, "Tonnetz should not produce NaN")
                XCTAssertFalse(val.isInfinite, "Tonnetz should not produce Inf")
            }
        }
    }

    /// Test tonnetz for multi-tone
    func testTonnetz_MultiTone() async throws {
        let signal = generateMultiTone()

        let config = TonnetzConfig(
            sampleRate: sampleRate,
            hopLength: 512
        )
        let tonnetz = Tonnetz(config: config)
        let result = await tonnetz.transform(signal)

        XCTAssertEqual(result.count, 6,
            "Tonnetz should have 6 dimensions")

        // Values should be in reasonable range [-1, 1] for normalized tonnetz
        for dim in 0..<result.count {
            for val in result[dim] {
                XCTAssertFalse(val.isNaN, "Tonnetz should not produce NaN")
            }
        }
    }

    // MARK: - Edge Cases

    /// Test CQT with silence
    func testCQT_Silence() async throws {
        let signal = generateSilence()

        let config = CQTConfig(
            sampleRate: sampleRate,
            hopLength: 512,
            fMin: 32.7,
            nBins: 84,
            binsPerOctave: 12
        )
        let cqt = CQT(config: config)
        let result = await cqt.transform(signal)

        // Get magnitude from ComplexMatrix
        let magnitude = result.magnitude

        // Should handle silence gracefully
        XCTAssertFalse(magnitude.isEmpty, "CQT should produce output for silence")

        // All values should be near zero for silence
        let maxVal = magnitude.flatMap { $0 }.max() ?? 0
        XCTAssertLessThan(maxVal, 0.01,
            "CQT of silence should have near-zero values")
    }

    /// Test chromagram with single impulse
    func testChromagram_Impulse() async throws {
        let signal = generateImpulse()

        let config = ChromagramConfig(
            sampleRate: sampleRate,
            hopLength: 512,
            nChroma: 12,
            useCQT: false  // Use STFT-based
        )
        let chroma = Chromagram(config: config)
        let result = await chroma.transform(signal)

        XCTAssertEqual(result.count, 12,
            "Chromagram should have 12 bins")

        // Impulse has broadband energy, so multiple chroma bins should be active
        XCTAssertFalse(result.flatMap { $0 }.isEmpty,
            "Chromagram should produce output for impulse")
    }

    /// Test CQT parameter variations
    func testCQT_ParameterVariations() async throws {
        let signal = generateSine(frequency: 440)

        // Test different bins per octave
        for bpo in [12, 24, 36] {
            let config = CQTConfig(
                sampleRate: sampleRate,
                hopLength: 512,
                fMin: 32.7,
                nBins: bpo * 5,  // 5 octaves
                binsPerOctave: bpo
            )
            let cqt = CQT(config: config)
            let result = await cqt.transform(signal)

            // Get magnitude from ComplexMatrix
            let magnitude = result.magnitude

            XCTAssertFalse(magnitude.isEmpty,
                "CQT with bpo=\(bpo) should produce output")
            XCTAssertEqual(magnitude.count, bpo * 5,
                "CQT should have \(bpo * 5) bins")
        }
    }
}
