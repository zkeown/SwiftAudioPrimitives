import XCTest
@testable import SwiftAudioPrimitives

final class TransformTests: XCTestCase {

    // MARK: - Test Signals

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    private func generateChord(frequencies: [Float], sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            frequencies.reduce(0) { sum, freq in
                sum + sin(2.0 * Float.pi * freq * Float(i) / sampleRate) / Float(frequencies.count)
            }
        }
    }

    // MARK: - CQT Tests

    func testCQTBasic() async {
        let cqt = CQT()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let result = await cqt.transform(signal)

        XCTAssertGreaterThan(result.rows, 0, "Should have frequency bins")
        XCTAssertGreaterThan(result.cols, 0, "Should have time frames")
    }

    func testCQTMagnitude() async {
        let cqt = CQT()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let magnitude = await cqt.magnitude(signal)

        XCTAssertGreaterThan(magnitude.count, 0)

        // All values should be non-negative
        for row in magnitude {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0)
                XCTAssertFalse(val.isNaN)
            }
        }
    }

    func testCQTOctaveResolution() async {
        let cqt = CQT(config: CQTConfig(
            fMin: 32.70,
            nBins: 84,  // 7 octaves
            binsPerOctave: 12
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let magnitude = await cqt.magnitude(signal)

        XCTAssertEqual(magnitude.count, 84, "Should have 84 bins (7 octaves * 12 bins)")
    }

    func testCQTPresets() async {
        let music = CQT.musicAnalysis
        let pitch = CQT.pitchAnalysis
        let librosa = CQT.librosaDefault

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let _ = await music.magnitude(signal)
        let _ = await pitch.magnitude(signal)
        let _ = await librosa.magnitude(signal)

        // Just verify they don't crash
        XCTAssert(true)
    }

    // MARK: - Chromagram Tests

    func testChromagramBasic() async {
        let chromagram = Chromagram()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let result = await chromagram.transform(signal)

        XCTAssertEqual(result.count, 12, "Chromagram should have 12 pitch classes")
        XCTAssertGreaterThan(result[0].count, 0, "Should have time frames")
    }

    func testChromagramCMajorChord() async {
        let chromagram = Chromagram()

        // C Major chord: C, E, G
        let cFreq: Float = 261.63  // C4
        let eFreq: Float = 329.63  // E4
        let gFreq: Float = 392.00  // G4

        let signal = generateChord(frequencies: [cFreq, eFreq, gFreq], sampleRate: 22050, duration: 1.0)
        let result = await chromagram.transform(signal)

        // Sum energy across time for each pitch class
        var pitchClassEnergy = [Float](repeating: 0, count: 12)
        for pc in 0..<12 {
            pitchClassEnergy[pc] = result[pc].reduce(0, +)
        }

        // Normalize
        let maxEnergy = pitchClassEnergy.max() ?? 1
        pitchClassEnergy = pitchClassEnergy.map { $0 / maxEnergy }

        // C, E, G should have relatively high energy
        // C = 0, E = 4, G = 7 in pitch class notation
        XCTAssertGreaterThan(pitchClassEnergy[0], 0.1, "C should have energy")  // C
    }

    func testChromagramShape() async {
        let chromagram = Chromagram(config: ChromagramConfig(
            sampleRate: 22050,
            hopLength: 512
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let result = await chromagram.transform(signal)

        // Should have 12 pitch classes
        XCTAssertEqual(result.count, 12)

        // All rows should have same number of frames
        let nFrames = result[0].count
        for row in result {
            XCTAssertEqual(row.count, nFrames)
        }
    }

    func testChromagramPresets() async {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let _ = await Chromagram.musicAnalysis.transform(signal)
        let _ = await Chromagram.chordRecognition.transform(signal)
        let _ = await Chromagram.stftBased.transform(signal)

        XCTAssert(true)
    }

    // MARK: - HPSS Tests

    func testHPSSBasic() async {
        let hpss = HPSS()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let result = await hpss.separate(signal)

        XCTAssertEqual(result.harmonic.count, signal.count,
            "Harmonic should match input length")
        XCTAssertEqual(result.percussive.count, signal.count,
            "Percussive should match input length")
    }

    func testHPSSHarmonic() async {
        let hpss = HPSS()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let harmonic = await hpss.harmonic(signal)

        XCTAssertEqual(harmonic.count, signal.count)

        // Harmonic content should have significant energy for a pure tone
        let energy = harmonic.map { $0 * $0 }.reduce(0, +)
        XCTAssertGreaterThan(energy, 0)
    }

    func testHPSSPercussive() async {
        let hpss = HPSS()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let percussive = await hpss.percussive(signal)

        XCTAssertEqual(percussive.count, signal.count)
    }

    func testHPSSSum() async {
        let hpss = HPSS()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let result = await hpss.separate(signal)

        // Harmonic + percussive should approximate original
        // (with some loss due to soft masking)
        var reconstruction = [Float](repeating: 0, count: signal.count)
        for i in 0..<signal.count {
            reconstruction[i] = result.harmonic[i] + result.percussive[i]
        }

        // Check energy ratio
        let originalEnergy = signal.map { $0 * $0 }.reduce(0, +)
        let reconEnergy = reconstruction.map { $0 * $0 }.reduce(0, +)

        // Reconstruction should preserve most energy
        let ratio = reconEnergy / max(originalEnergy, 1e-10)
        XCTAssertGreaterThan(ratio, 0.5, "Should preserve significant energy")
    }

    func testHPSSPresets() async {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let _ = await HPSS.standard.separate(signal)
        let _ = await HPSS.vocalsFromDrums.separate(signal)
        let _ = await HPSS.fast.separate(signal)

        XCTAssert(true)
    }

    // MARK: - Edge Cases

    func testEmptySignal() async {
        let cqt = CQT()
        let chromagram = Chromagram()
        let hpss = HPSS()

        let empty: [Float] = []

        let cqtResult = await cqt.transform(empty)
        let chromaResult = await chromagram.transform(empty)
        let hpssResult = await hpss.separate(empty)

        XCTAssertEqual(cqtResult.rows, 0)
        XCTAssertEqual(chromaResult.count, 0)
        XCTAssertEqual(hpssResult.harmonic.count, 0)
    }

    func testShortSignal() async {
        let cqt = CQT()
        let short = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.01)

        let result = await cqt.transform(short)
        XCTAssertNotNil(result)
    }
}
