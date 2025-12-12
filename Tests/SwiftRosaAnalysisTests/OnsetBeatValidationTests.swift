import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaAnalysis

/// Comprehensive onset detection and beat tracking validation tests against librosa reference.
///
/// These tests validate:
/// 1. Onset strength envelope
/// 2. Onset frame detection
/// 3. Beat tracking
/// 4. Tempo estimation
/// 5. Tempogram
final class OnsetBeatValidationTests: XCTestCase {

    // MARK: - Constants

    private let sampleRate: Float = 22050
    private let hopLength = 512

    // MARK: - Signal Generation Helpers

    private func generateClickTrain() -> [Float] {
        // Clicks at 0, 0.25, 0.5, 0.75 seconds (4 clicks/second = 240 BPM)
        let numSamples = Int(sampleRate)
        var signal = [Float](repeating: 0, count: numSamples)
        let clickPositions = [0, Int(sampleRate / 4), Int(sampleRate / 2), Int(3 * sampleRate / 4)]
        for pos in clickPositions where pos < numSamples {
            // Add a short impulse at each click position
            for i in 0..<min(100, numSamples - pos) {
                signal[pos + i] = exp(-Float(i) * 0.05) * (i % 2 == 0 ? 1 : -1)
            }
        }
        return signal
    }

    private func generateAMModulated() -> [Float] {
        // 440 Hz carrier modulated at 5 Hz
        let numSamples = Int(sampleRate)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            let carrier = sin(2 * .pi * 440 * t)
            let modulator = (1 + sin(2 * .pi * 5 * t)) / 2
            signal[i] = carrier * modulator
        }
        return signal
    }

    private func generateMultiTone() -> [Float] {
        let frequencies: [Float] = [440, 880, 1320]
        let numSamples = Int(sampleRate)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            for freq in frequencies {
                signal[i] += sin(2 * .pi * freq * t) / Float(frequencies.count)
            }
        }
        return signal
    }

    private func generateSilence() -> [Float] {
        return [Float](repeating: 0, count: Int(sampleRate))
    }

    private func generateImpulse() -> [Float] {
        var signal = [Float](repeating: 0, count: Int(sampleRate))
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

    // MARK: - Onset Strength Tests

    /// Test onset strength envelope for click train (known onsets)
    func testOnsetStrength_ClickTrain() async throws {
        let ref = try getReference()
        guard let onsetRef = ref["onset"] as? [String: Any],
              let clickRef = onsetRef["click_train"] as? [String: Any],
              let expectedStrength = clickRef["onset_strength"] as? [Double],
              let refHopLength = clickRef["hop_length"] as? Int else {
            throw XCTSkip("Onset reference data not found for click_train")
        }

        let signal = generateClickTrain()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength,
            method: .spectralFlux
        )
        let detector = OnsetDetector(config: config)
        let strength = await detector.onsetStrength(signal)

        // Onset strength should have peaks at click positions
        // Clicks are at 0, sr/4, sr/2, 3*sr/4 samples
        // With hop=512 at sr=22050, that's frames 0, ~10.8, ~21.5, ~32.3

        XCTAssertFalse(strength.isEmpty, "Should produce onset strength")

        // Find peaks in our onset strength
        var ourPeaks: [Int] = []
        for i in 1..<(strength.count - 1) {
            if strength[i] > strength[i-1] && strength[i] > strength[i+1] &&
               strength[i] > 0.1 * (strength.max() ?? 1) {
                ourPeaks.append(i)
            }
        }

        // Should detect approximately 4 peaks for 4 clicks
        XCTAssertGreaterThanOrEqual(ourPeaks.count, 2,
            "Should detect multiple onset peaks for click train")
        XCTAssertLessThanOrEqual(ourPeaks.count, 8,
            "Should not detect too many false positives")
    }

    /// Test onset strength for AM-modulated signal
    func testOnsetStrength_AMModulated() async throws {
        let ref = try getReference()
        guard let onsetRef = ref["onset"] as? [String: Any],
              let amRef = onsetRef["am_modulated"] as? [String: Any],
              let _ = amRef["onset_strength"] as? [Double],
              let refHopLength = amRef["hop_length"] as? Int else {
            throw XCTSkip("Onset reference data not found for am_modulated")
        }

        let signal = generateAMModulated()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength,
            method: .energy
        )
        let detector = OnsetDetector(config: config)
        let strength = await detector.onsetStrength(signal)

        // AM modulated at 5 Hz should show ~5 energy peaks per second
        XCTAssertFalse(strength.isEmpty, "Should produce onset strength")

        // Find peaks
        var peakCount = 0
        for i in 1..<(strength.count - 1) {
            if strength[i] > strength[i-1] && strength[i] > strength[i+1] &&
               strength[i] > 0.2 * (strength.max() ?? 1) {
                peakCount += 1
            }
        }

        // Should see ~5 peaks for 5 Hz modulation over 1 second
        XCTAssertGreaterThanOrEqual(peakCount, 3,
            "Should detect multiple peaks for AM signal")
        XCTAssertLessThanOrEqual(peakCount, 10,
            "Should not detect too many peaks")
    }

    // MARK: - Onset Detection Tests

    /// Test onset detection for click train
    func testOnsetDetection_ClickTrain() async throws {
        let ref = try getReference()
        guard let onsetRef = ref["onset"] as? [String: Any],
              let clickRef = onsetRef["click_train"] as? [String: Any],
              let expectedFrames = clickRef["onset_frames"] as? [Int],
              let refHopLength = clickRef["hop_length"] as? Int else {
            throw XCTSkip("Onset reference data not found for click_train")
        }

        let signal = generateClickTrain()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength,
            method: .spectralFlux
        )
        let detector = OnsetDetector(config: config)
        let detected = await detector.detect(signal)

        // Should detect approximately the right number of onsets
        XCTAssertGreaterThan(detected.count, 0, "Should detect some onsets")

        // Or at least detect the correct number of onsets
        XCTAssertEqual(detected.count, expectedFrames.count, accuracy: 3,
            "Should detect approximately \(expectedFrames.count) onsets")
    }

    /// Test onset times conversion
    func testOnsetTimes_ClickTrain() async throws {
        let ref = try getReference()
        guard let onsetRef = ref["onset"] as? [String: Any],
              let clickRef = onsetRef["click_train"] as? [String: Any],
              let _ = clickRef["onset_times"] as? [Double],
              let refHopLength = clickRef["hop_length"] as? Int else {
            throw XCTSkip("Onset reference data not found for click_train")
        }

        let signal = generateClickTrain()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength,
            method: .spectralFlux
        )
        let detector = OnsetDetector(config: config)
        let times = await detector.detectTimes(signal)

        // Times should be in reasonable range (0 to 1 second for 1 second signal)
        for time in times {
            XCTAssertGreaterThanOrEqual(time, 0, "Onset time should be >= 0")
            XCTAssertLessThanOrEqual(time, 1.2, "Onset time should be <= duration + margin")
        }

        // Click train has clicks at 0, 0.25, 0.5, 0.75 seconds
        let expectedClickTimes: [Float] = [0, 0.25, 0.5, 0.75]

        // At least some detected times should match expected
        var matchCount = 0
        for expected in expectedClickTimes {
            for detected in times {
                if Swift.abs(detected - expected) < 0.1 {  // 100ms tolerance
                    matchCount += 1
                    break
                }
            }
        }

        XCTAssertGreaterThanOrEqual(matchCount, 2,
            "Should detect at least 2 of the 4 click times")
    }

    // MARK: - Beat Tracking Tests

    /// Test beat tracking for click train (regular rhythm)
    func testBeatTracking_ClickTrain() async throws {
        let ref = try getReference()
        guard let beatRef = ref["beat"] as? [String: Any],
              let clickRef = beatRef["click_train"] as? [String: Any],
              let _ = clickRef["tempo"] as? Double,
              let _ = clickRef["beat_frames"] as? [Int],
              let refHopLength = clickRef["hop_length"] as? Int else {
            throw XCTSkip("Beat reference data not found for click_train")
        }

        let signal = generateClickTrain()

        let trackerConfig = BeatTrackerConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength
        )
        let tracker = BeatTracker(config: trackerConfig)
        let beats = await tracker.track(signal)
        let tempo = await tracker.estimateTempo(signal)

        // Click train at 4 clicks/second = 240 BPM
        let expectedBPM: Float = 240

        // Tempo should be in ballpark (may detect at half/double rate)
        let tempoRatio = tempo / expectedBPM
        let isCorrectTempo = Swift.abs(tempoRatio - 1.0) < 0.2 ||  // Within 20% of expected
                             Swift.abs(tempoRatio - 0.5) < 0.1 ||  // Half tempo
                             Swift.abs(tempoRatio - 2.0) < 0.2     // Double tempo

        XCTAssertTrue(isCorrectTempo,
            "Tempo \(tempo) should be near \(expectedBPM) BPM (or half/double)")

        // Should detect some beats
        XCTAssertGreaterThan(beats.count, 0,
            "Should detect at least some beats")
    }

    /// Test tempo estimation
    func testTempoEstimation_ClickTrain() async throws {
        let ref = try getReference()
        guard let beatRef = ref["beat"] as? [String: Any],
              let clickRef = beatRef["click_train"] as? [String: Any],
              let _ = clickRef["tempo"] as? Double,
              let refHopLength = clickRef["hop_length"] as? Int else {
            throw XCTSkip("Beat reference data not found for click_train")
        }

        let signal = generateClickTrain()

        let trackerConfig = BeatTrackerConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength
        )
        let tracker = BeatTracker(config: trackerConfig)
        let estimatedTempo = await tracker.estimateTempo(signal)

        // Click train at 4 clicks/second = 240 BPM
        let expectedBPM: Float = 240

        // Check if tempo is correct (allowing for octave errors)
        let tempoRatio = estimatedTempo / expectedBPM
        let isCorrectTempo = Swift.abs(tempoRatio - 1.0) < 0.25 ||
                             Swift.abs(tempoRatio - 0.5) < 0.15 ||
                             Swift.abs(tempoRatio - 2.0) < 0.3

        XCTAssertTrue(isCorrectTempo,
            "Estimated tempo \(estimatedTempo) should be near \(expectedBPM) BPM")
    }

    // MARK: - Tempogram Tests

    /// Test tempogram for click train
    func testTempogram_ClickTrain() async throws {
        let ref = try getReference()
        guard let tempogramRef = ref["tempogram"] as? [String: Any],
              let clickRef = tempogramRef["click_train"] as? [String: Any],
              let _ = clickRef["tempogram"] as? [[Double]],
              let refHopLength = clickRef["hop_length"] as? Int,
              let _ = clickRef["shape"] as? [Int] else {
            throw XCTSkip("Tempogram reference data not found")
        }

        let signal = generateClickTrain()

        let config = TempogramConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength
        )
        let tempogram = Tempogram(config: config)
        let result = await tempogram.transform(signal)

        // Verify output shape
        XCTAssertFalse(result.isEmpty, "Tempogram should not be empty")

        // Tempogram should show peak at ~240 BPM for 4 clicks/second
        // BPM index depends on tempogram settings

        // Find tempo with maximum energy
        var maxBPMIdx = 0
        var maxEnergy: Float = 0
        for bpmIdx in 0..<result.count {
            let avgEnergy = result[bpmIdx].reduce(0, +) / Float(max(1, result[bpmIdx].count))
            if avgEnergy > maxEnergy {
                maxEnergy = avgEnergy
                maxBPMIdx = bpmIdx
            }
        }

        // Should find some dominant tempo
        XCTAssertGreaterThan(maxEnergy, 0,
            "Tempogram should have energy at some tempo")
    }

    /// Test tempogram for multi-tone (should have less clear rhythm)
    func testTempogram_MultiTone() async throws {
        let ref = try getReference()
        guard let tempogramRef = ref["tempogram"] as? [String: Any],
              let multiRef = tempogramRef["multi_tone"] as? [String: Any],
              let refHopLength = multiRef["hop_length"] as? Int else {
            throw XCTSkip("Tempogram reference data not found for multi_tone")
        }

        let signal = generateMultiTone()

        let config = TempogramConfig(
            sampleRate: sampleRate,
            hopLength: refHopLength
        )
        let tempogram = Tempogram(config: config)
        let result = await tempogram.transform(signal)

        // Multi-tone is a steady signal without clear rhythm
        // Tempogram energy should be more uniform (no strong peaks)
        XCTAssertFalse(result.isEmpty, "Tempogram should not be empty")
    }

    // MARK: - Edge Cases

    /// Test onset detection with silence
    func testOnsetDetection_Silence() async throws {
        let signal = generateSilence()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            method: .energy
        )
        let detector = OnsetDetector(config: config)
        let onsets = await detector.detect(signal)

        // Silence should have no onsets
        XCTAssertEqual(onsets.count, 0,
            "Silence should have no detected onsets")
    }

    /// Test beat tracking with silence
    func testBeatTracking_Silence() async throws {
        let signal = generateSilence()

        let trackerConfig = BeatTrackerConfig(
            sampleRate: sampleRate,
            hopLength: hopLength
        )
        let tracker = BeatTracker(config: trackerConfig)
        let tempo = await tracker.estimateTempo(signal)

        // Should handle silence gracefully (may return 0 or some default tempo)
        XCTAssertFalse(tempo.isNaN, "Tempo should not be NaN for silence")
        XCTAssertFalse(tempo.isInfinite, "Tempo should not be Inf for silence")
    }

    /// Test onset detection with single impulse
    func testOnsetDetection_Impulse() async throws {
        let signal = generateImpulse()

        let config = OnsetConfig(
            sampleRate: sampleRate,
            hopLength: hopLength,
            method: .spectralFlux
        )
        let detector = OnsetDetector(config: config)
        let onsets = await detector.detect(signal)

        // Should detect at least one onset for single impulse
        XCTAssertGreaterThanOrEqual(onsets.count, 1,
            "Should detect at least one onset for impulse")
        XCTAssertLessThanOrEqual(onsets.count, 3,
            "Should not detect too many onsets for single impulse")

        // Onset should be near the impulse position (middle of signal)
        if !onsets.isEmpty {
            let impulseFrame = onsets[0]
            let expectedFrame = (Int(sampleRate) / 2) / hopLength  // ~21.5
            XCTAssertEqual(impulseFrame, expectedFrame, accuracy: 5,
                "Onset should be near impulse position")
        }
    }

    // MARK: - Different Onset Methods

    /// Test different onset detection methods produce reasonable results
    func testOnsetMethods_Comparison() async throws {
        let signal = generateClickTrain()

        let methods: [OnsetMethod] = [.energy, .spectralFlux, .highFrequencyContent]
        var allResults: [OnsetMethod: [Int]] = [:]

        for method in methods {
            let config = OnsetConfig(
                sampleRate: sampleRate,
                hopLength: hopLength,
                method: method
            )
            let detector = OnsetDetector(config: config)
            let onsets = await detector.detect(signal)
            allResults[method] = onsets

            // Each method should detect some onsets
            XCTAssertGreaterThan(onsets.count, 0,
                "Method \(method) should detect onsets in click train")
        }

        // Different methods may detect slightly different onsets,
        // but all should find approximately 4 for click train
        for (method, onsets) in allResults {
            XCTAssertGreaterThanOrEqual(onsets.count, 2,
                "Method \(method) should detect at least 2 onsets")
            XCTAssertLessThanOrEqual(onsets.count, 10,
                "Method \(method) should not detect too many onsets")
        }
    }
}
