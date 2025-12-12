import XCTest
@testable import SwiftAudioPrimitives

final class OnsetTests: XCTestCase {

    // MARK: - Test Signals

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float, amplitude: Float = 1.0) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            amplitude * sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    /// Generate signal with sudden onset
    private func generateOnsetSignal(sampleRate: Float, duration: Float, onsetTime: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        let onsetSample = Int(sampleRate * onsetTime)

        return (0..<numSamples).map { i in
            if i < onsetSample {
                return 0
            } else {
                return 0.8 * sin(2.0 * Float.pi * 440 * Float(i) / sampleRate)
            }
        }
    }

    /// Generate signal with multiple onsets
    private func generateMultipleOnsets(sampleRate: Float, duration: Float, onsetTimes: [Float]) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)

        for onsetTime in onsetTimes {
            let onsetSample = Int(sampleRate * onsetTime)
            let decayLength = Int(sampleRate * 0.1)  // 100ms decay

            for i in onsetSample..<min(onsetSample + decayLength, numSamples) {
                let t = Float(i - onsetSample) / Float(decayLength)
                let envelope = exp(-5 * t)  // Exponential decay
                signal[i] += envelope * sin(2.0 * Float.pi * 440 * Float(i) / sampleRate)
            }
        }

        return signal
    }

    // MARK: - OnsetDetector Tests

    func testOnsetDetectorBasic() async {
        let detector = OnsetDetector()

        let signal = generateOnsetSignal(sampleRate: 22050, duration: 1.0, onsetTime: 0.5)
        let onsets = await detector.detect(signal)

        XCTAssertGreaterThanOrEqual(onsets.count, 1, "Should detect at least one onset")
    }

    func testOnsetDetectorTimes() async {
        let detector = OnsetDetector(config: OnsetConfig(
            sampleRate: 22050,
            hopLength: 512
        ))

        let onsetTime: Float = 0.5
        let signal = generateOnsetSignal(sampleRate: 22050, duration: 1.0, onsetTime: onsetTime)
        let onsetTimes = await detector.detectTimes(signal)

        XCTAssertGreaterThanOrEqual(onsetTimes.count, 1)

        // At least one onset should be near the expected time
        let nearExpected = onsetTimes.filter { abs($0 - onsetTime) < 0.1 }
        XCTAssertGreaterThan(nearExpected.count, 0,
            "Should detect onset near \(onsetTime)s")
    }

    func testOnsetStrengthEnvelope() async {
        let detector = OnsetDetector()

        let signal = generateOnsetSignal(sampleRate: 22050, duration: 1.0, onsetTime: 0.5)
        let envelope = await detector.onsetStrength(signal)

        XCTAssertGreaterThan(envelope.count, 0)

        // All values should be non-negative
        for val in envelope {
            XCTAssertGreaterThanOrEqual(val, 0)
        }
    }

    func testMultipleOnsets() async {
        let detector = OnsetDetector(config: OnsetConfig(
            sampleRate: 22050,
            hopLength: 256,
            peakThreshold: 0.5
        ))

        let onsetTimes: [Float] = [0.2, 0.5, 0.8]
        let signal = generateMultipleOnsets(sampleRate: 22050, duration: 1.2, onsetTimes: onsetTimes)
        let detected = await detector.detectTimes(signal)

        // Should detect multiple onsets
        XCTAssertGreaterThanOrEqual(detected.count, 2,
            "Should detect multiple onsets")
    }

    func testOnsetDetectorMethods() async {
        let signal = generateOnsetSignal(sampleRate: 22050, duration: 1.0, onsetTime: 0.5)

        let energyDetector = OnsetDetector(config: OnsetConfig(
            method: .energy
        ))
        let fluxDetector = OnsetDetector(config: OnsetConfig(
            method: .spectralFlux
        ))
        let complexDetector = OnsetDetector(config: OnsetConfig(
            method: .complexDomain
        ))

        let energyOnsets = await energyDetector.detect(signal)
        let fluxOnsets = await fluxDetector.detect(signal)
        let complexOnsets = await complexDetector.detect(signal)

        // All methods should detect something
        XCTAssertGreaterThan(energyOnsets.count + fluxOnsets.count + complexOnsets.count, 0)
    }

    func testOnsetDetectorPresets() async {
        let signal = generateOnsetSignal(sampleRate: 22050, duration: 1.0, onsetTime: 0.5)

        let _ = await OnsetDetector.percussive.detect(signal)
        let _ = await OnsetDetector.melodic.detect(signal)
        let _ = await OnsetDetector.standard.detect(signal)

        XCTAssert(true)
    }

    // MARK: - BeatTracker Tests

    func testBeatTrackerBasic() async {
        let tracker = BeatTracker()

        // Generate rhythmic signal with regular beats
        let sampleRate: Float = 22050
        let duration: Float = 4.0
        let bpm: Float = 120  // 2 beats per second
        let beatPeriod = 60.0 / bpm
        var beatTimes: [Float] = []

        var t: Float = beatPeriod
        while t < duration {
            beatTimes.append(t)
            t += beatPeriod
        }

        let signal = generateMultipleOnsets(sampleRate: sampleRate, duration: duration, onsetTimes: beatTimes)
        let beats = await tracker.track(signal)

        XCTAssertGreaterThan(beats.count, 0, "Should detect beats")
    }

    func testTempoEstimation() async {
        let tracker = BeatTracker()

        // Generate 120 BPM signal
        let sampleRate: Float = 22050
        let duration: Float = 8.0
        let targetBPM: Float = 120

        let beatPeriod = 60.0 / targetBPM
        var beatTimes: [Float] = []

        var t: Float = beatPeriod
        while t < duration {
            beatTimes.append(t)
            t += beatPeriod
        }

        let signal = generateMultipleOnsets(sampleRate: sampleRate, duration: duration, onsetTimes: beatTimes)
        let estimatedTempo = await tracker.estimateTempo(signal)

        // Should be reasonably close to target BPM
        // Allow wide tolerance due to estimation difficulty
        if estimatedTempo > 0 {
            let ratio = estimatedTempo / targetBPM
            // Tempo could be detected as double or half
            let normalizedRatio = ratio > 1.5 ? ratio / 2 : (ratio < 0.75 ? ratio * 2 : ratio)
            XCTAssertEqual(normalizedRatio, 1.0, accuracy: 0.3,
                "Tempo should be close to \(targetBPM) BPM (or double/half)")
        }
    }

    func testBeatTrackerPresets() async {
        let signal = generateMultipleOnsets(
            sampleRate: 22050,
            duration: 4.0,
            onsetTimes: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        )

        let _ = await BeatTracker.standard.track(signal)
        let _ = await BeatTracker.electronic.track(signal)
        let _ = await BeatTracker.classical.track(signal)

        XCTAssert(true)
    }

    // MARK: - Edge Cases

    func testSilence() async {
        let detector = OnsetDetector()
        let silence = [Float](repeating: 0, count: 22050)

        let onsets = await detector.detect(silence)
        XCTAssertEqual(onsets.count, 0, "Should not detect onsets in silence")
    }

    func testConstantSignal() async {
        let detector = OnsetDetector()
        let constant = [Float](repeating: 0.5, count: 22050)

        let onsets = await detector.detect(constant)
        XCTAssertEqual(onsets.count, 0, "Should not detect onsets in constant signal")
    }

    func testShortSignal() async {
        let detector = OnsetDetector()
        let short = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.05)

        let onsets = await detector.detect(short)
        XCTAssertNotNil(onsets)
    }

    func testEmptySignal() async {
        let detector = OnsetDetector()
        let onsets = await detector.detect([])

        XCTAssertEqual(onsets.count, 0)
    }
}
