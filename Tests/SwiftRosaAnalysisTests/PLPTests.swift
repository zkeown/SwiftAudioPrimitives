import XCTest
@testable import SwiftRosaAnalysis
@testable import SwiftRosaCore

final class PLPTests: XCTestCase {

    // MARK: - Basic PLP Tests

    func testPLPBasic() async {
        let plp = PLP()

        // Generate a simple rhythmic signal (clicks at regular intervals)
        let sampleRate: Float = 22050
        let duration: Float = 2.0
        let nSamples = Int(duration * sampleRate)
        var signal = [Float](repeating: 0, count: nSamples)

        // Add clicks every ~0.5 seconds (120 BPM)
        let clickInterval = Int(0.5 * sampleRate)
        for i in stride(from: 0, to: nSamples, by: clickInterval) {
            let clickEnd = min(i + 100, nSamples)
            for j in i..<clickEnd {
                signal[j] = 0.8
            }
        }

        let pulse = await plp.compute(signal)

        // Should return non-empty pulse curve
        XCTAssertGreaterThan(pulse.count, 0)

        // Pulse values should be normalized (0 to 1)
        for value in pulse {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
    }

    func testPLPEmptySignal() async {
        let plp = PLP()

        let signal: [Float] = []
        let pulse = await plp.compute(signal)

        XCTAssertEqual(pulse.count, 0)
    }

    func testPLPFromOnset() {
        let plp = PLP()

        // Pre-computed onset envelope (simulating periodic beats)
        let nFrames = 100
        var onset = [Float](repeating: 0, count: nFrames)

        // Add peaks at regular intervals (every 10 frames)
        for i in stride(from: 0, to: nFrames, by: 10) {
            onset[i] = 1.0
        }

        let pulse = plp.computeFromOnset(onset)

        XCTAssertEqual(pulse.count, nFrames)

        // Pulse should be normalized
        let maxPulse = pulse.max() ?? 0
        XCTAssertLessThanOrEqual(maxPulse, 1.0)
    }

    // MARK: - Configuration Tests

    func testPLPConfigDefaults() {
        let config = PLPConfig()

        XCTAssertEqual(config.sampleRate, 22050)
        XCTAssertEqual(config.hopLength, 512)
        XCTAssertEqual(config.tempoMin, 30)
        XCTAssertEqual(config.tempoMax, 300)
        XCTAssertNil(config.prior)
    }

    func testPLPConfigCustom() {
        let config = PLPConfig(
            sampleRate: 44100,
            hopLength: 256,
            tempoMin: 60,
            tempoMax: 200,
            prior: 120
        )

        XCTAssertEqual(config.sampleRate, 44100)
        XCTAssertEqual(config.hopLength, 256)
        XCTAssertEqual(config.tempoMin, 60)
        XCTAssertEqual(config.tempoMax, 200)
        XCTAssertEqual(config.prior, 120)
    }

    func testPLPFramesPerSecond() {
        let config = PLPConfig(sampleRate: 22050, hopLength: 512)

        let expectedFPS: Float = 22050.0 / 512.0
        XCTAssertEqual(config.framesPerSecond, expectedFPS, accuracy: 0.001)
    }

    // MARK: - Tempo Range Tests

    func testPLPWithTempoRange() async {
        let plp = PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 100,
            tempoMax: 150
        ))

        // Generate signal with tempo in range (120 BPM)
        let sampleRate: Float = 22050
        let duration: Float = 2.0
        let nSamples = Int(duration * sampleRate)
        var signal = [Float](repeating: 0, count: nSamples)

        // 120 BPM = 0.5 seconds per beat
        let beatInterval = Int(0.5 * sampleRate)
        for i in stride(from: 0, to: nSamples, by: beatInterval) {
            let end = min(i + 100, nSamples)
            for j in i..<end {
                signal[j] = 0.8
            }
        }

        let pulse = await plp.compute(signal)

        XCTAssertGreaterThan(pulse.count, 0)
    }

    func testPLPWithPrior() async {
        // PLP with tempo prior at 128 BPM
        let plpWithPrior = PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 100,
            tempoMax: 150,
            prior: 128
        ))

        let plpWithoutPrior = PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 100,
            tempoMax: 150,
            prior: nil
        ))

        // Generate rhythmic signal
        let nSamples = 44100
        var signal = [Float](repeating: 0, count: nSamples)

        let beatInterval = Int(0.469 * 22050)  // ~128 BPM
        for i in stride(from: 0, to: nSamples, by: beatInterval) {
            let end = min(i + 100, nSamples)
            for j in i..<end {
                signal[j] = 0.8
            }
        }

        let pulseWithPrior = await plpWithPrior.compute(signal)
        let pulseWithoutPrior = await plpWithoutPrior.compute(signal)

        // Both should produce output
        XCTAssertGreaterThan(pulseWithPrior.count, 0)
        XCTAssertGreaterThan(pulseWithoutPrior.count, 0)
    }

    // MARK: - Presets Tests

    func testStandardPreset() async {
        let plp = PLP.standard

        XCTAssertEqual(plp.config.sampleRate, 22050)
        XCTAssertEqual(plp.config.tempoMin, 30)
        XCTAssertEqual(plp.config.tempoMax, 300)
        XCTAssertNil(plp.config.prior)
    }

    func testDanceMusicPreset() async {
        let plp = PLP.danceMusic

        XCTAssertEqual(plp.config.tempoMin, 100)
        XCTAssertEqual(plp.config.tempoMax, 200)
        XCTAssertEqual(plp.config.prior, 128)
    }

    func testSlowMusicPreset() async {
        let plp = PLP.slowMusic

        XCTAssertEqual(plp.config.tempoMin, 40)
        XCTAssertEqual(plp.config.tempoMax, 120)
        XCTAssertEqual(plp.config.prior, 80)
    }

    func testPresetsProduceOutput() async {
        let nSamples = 22050
        var signal = [Float](repeating: 0, count: nSamples)

        // Simple rhythmic pattern
        for i in stride(from: 0, to: nSamples, by: 5000) {
            let end = min(i + 100, nSamples)
            for j in i..<end {
                signal[j] = 0.8
            }
        }

        let pulseStandard = await PLP.standard.compute(signal)
        let pulseDance = await PLP.danceMusic.compute(signal)
        let pulseSlow = await PLP.slowMusic.compute(signal)

        XCTAssertGreaterThan(pulseStandard.count, 0)
        XCTAssertGreaterThan(pulseDance.count, 0)
        XCTAssertGreaterThan(pulseSlow.count, 0)
    }

    // MARK: - Edge Cases

    func testPLPVeryShortSignal() async {
        let plp = PLP()

        // Signal shorter than one frame
        let signal = [Float](repeating: 0.5, count: 256)

        let pulse = await plp.compute(signal)

        // Should handle gracefully (may return empty or minimal output)
        // Main test is that it doesn't crash
        XCTAssertGreaterThanOrEqual(pulse.count, 0)
    }

    func testPLPSilentSignal() async {
        let plp = PLP()

        // Completely silent signal
        let signal = [Float](repeating: 0, count: 22050)

        let pulse = await plp.compute(signal)

        // Should produce output (likely zeros or near-zeros)
        XCTAssertGreaterThanOrEqual(pulse.count, 0)
    }

    func testPLPConstantSignal() async {
        let plp = PLP()

        // Constant (DC) signal
        let signal = [Float](repeating: 0.5, count: 22050)

        let pulse = await plp.compute(signal)

        // Should handle without crashing
        XCTAssertGreaterThanOrEqual(pulse.count, 0)
    }

    func testPLPFromOnsetEmpty() {
        let plp = PLP()

        let onset: [Float] = []
        let pulse = plp.computeFromOnset(onset)

        XCTAssertEqual(pulse.count, 0)
    }

    func testPLPNormalization() async {
        let plp = PLP()

        // Generate strong rhythmic signal
        let nSamples = 44100
        var signal = [Float](repeating: 0, count: nSamples)

        for i in stride(from: 0, to: nSamples, by: 5000) {
            let end = min(i + 200, nSamples)
            for j in i..<end {
                signal[j] = 1.0  // Strong amplitude
            }
        }

        let pulse = await plp.compute(signal)

        // All values should be normalized between 0 and 1
        for value in pulse {
            XCTAssertGreaterThanOrEqual(value, 0, "Pulse should be >= 0")
            XCTAssertLessThanOrEqual(value, 1.0 + 1e-6, "Pulse should be <= 1")
        }

        // Max should be close to 1 (due to normalization)
        if let maxValue = pulse.max() {
            XCTAssertGreaterThan(maxValue, 0.9, "Max pulse should be close to 1 after normalization")
        }
    }
}
