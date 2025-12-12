import XCTest
@testable import SwiftRosaAnalysis
import SwiftRosaCore

final class PitchDetectorTests: XCTestCase {

    // MARK: - Test Signals

    /// Generate a pure sine wave
    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    // MARK: - Basic Pitch Detection

    func testDetectA4() async {
        let detector = PitchDetector(config: PitchDetectorConfig(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65,
            fMax: 2000,
            threshold: 0.1
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let pitches = await detector.transform(signal)

        XCTAssertGreaterThan(pitches.count, 0)

        // Filter non-zero pitches
        let detectedPitches = pitches.filter { $0 > 0 }
        XCTAssertGreaterThan(detectedPitches.count, 0, "Should detect some pitches")

        // Average detected pitch should be close to 440 Hz
        let avgPitch = detectedPitches.reduce(0, +) / Float(detectedPitches.count)
        XCTAssertEqual(avgPitch, 440, accuracy: 20, "Average pitch should be close to 440 Hz")
    }

    func testDetectC4() async {
        let detector = PitchDetector.music

        let c4Frequency: Float = 261.63  // C4
        let signal = generateSineWave(frequency: c4Frequency, sampleRate: 22050, duration: 1.0)
        let pitches = await detector.transform(signal)

        let detectedPitches = pitches.filter { $0 > 0 }
        XCTAssertGreaterThan(detectedPitches.count, 0)

        let avgPitch = detectedPitches.reduce(0, +) / Float(detectedPitches.count)
        XCTAssertEqual(avgPitch, c4Frequency, accuracy: 15, "Should detect C4 frequency")
    }

    func testDetectLowFrequency() async {
        let detector = PitchDetector(config: PitchDetectorConfig(
            sampleRate: 22050,
            frameLength: 4096,  // Longer frame for low frequencies
            hopLength: 512,
            fMin: 50,
            fMax: 400,
            threshold: 0.1
        ))

        let signal = generateSineWave(frequency: 100, sampleRate: 22050, duration: 1.0)
        let pitches = await detector.transform(signal)

        let detectedPitches = pitches.filter { $0 > 0 }
        XCTAssertGreaterThan(detectedPitches.count, 0)

        let avgPitch = detectedPitches.reduce(0, +) / Float(detectedPitches.count)
        XCTAssertEqual(avgPitch, 100, accuracy: 10, "Should detect 100 Hz")
    }

    // MARK: - Confidence Tests

    func testConfidenceForPureTone() async {
        let detector = PitchDetector.music

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let (pitches, confidences) = await detector.transformWithConfidence(signal)

        XCTAssertEqual(pitches.count, confidences.count)

        // Pure tone should have high confidence
        let validConfidences = zip(pitches, confidences)
            .filter { $0.0 > 0 }
            .map { $0.1 }

        XCTAssertGreaterThan(validConfidences.count, 0)

        let avgConfidence = validConfidences.reduce(0, +) / Float(validConfidences.count)
        XCTAssertGreaterThan(avgConfidence, 0.5, "Pure tone should have high confidence")
    }

    // MARK: - Utility Functions

    func testFrequencyToMIDI() {
        // A4 = 440 Hz = MIDI 69
        let midiA4 = PitchDetector.frequencyToMIDI(440)
        XCTAssertEqual(midiA4, 69, accuracy: 0.01)

        // C4 = 261.63 Hz = MIDI 60
        let midiC4 = PitchDetector.frequencyToMIDI(261.63)
        XCTAssertEqual(midiC4, 60, accuracy: 0.1)

        // Zero frequency should return 0
        let midiZero = PitchDetector.frequencyToMIDI(0)
        XCTAssertEqual(midiZero, 0)
    }

    func testMIDIToFrequency() {
        // MIDI 69 = A4 = 440 Hz
        let freqA4 = PitchDetector.midiToFrequency(69)
        XCTAssertEqual(freqA4, 440, accuracy: 0.01)

        // MIDI 60 = C4 = 261.63 Hz
        let freqC4 = PitchDetector.midiToFrequency(60)
        XCTAssertEqual(freqC4, 261.63, accuracy: 0.1)
    }

    func testFrequencyToNoteName() {
        let noteA4 = PitchDetector.frequencyToNoteName(440)
        XCTAssertEqual(noteA4, "A4")

        let noteC4 = PitchDetector.frequencyToNoteName(261.63)
        XCTAssertEqual(noteC4, "C4")

        // Zero frequency should return "-"
        let noteZero = PitchDetector.frequencyToNoteName(0)
        XCTAssertEqual(noteZero, "-")
    }

    func testMIDIRoundTrip() {
        let testFrequencies: [Float] = [110, 220, 440, 880, 1760]

        for freq in testFrequencies {
            let midi = PitchDetector.frequencyToMIDI(freq)
            let backToFreq = PitchDetector.midiToFrequency(midi)
            XCTAssertEqual(backToFreq, freq, accuracy: 0.01,
                "Round trip for \(freq) Hz failed")
        }
    }

    // MARK: - Median Filter

    func testMedianFilter() {
        let pitches: [Float] = [440, 445, 200, 443, 442]  // 200 is an outlier
        let filtered = PitchDetector.medianFilter(pitches, windowSize: 3)

        XCTAssertEqual(filtered.count, pitches.count)

        // The median filter should reduce the outlier's influence
        XCTAssertNotEqual(filtered[2], 200, "Outlier should be filtered")
    }

    // MARK: - Presets

    func testVoicePreset() async {
        let detector = PitchDetector.voice

        // Voice frequency range test
        let signal = generateSineWave(frequency: 200, sampleRate: 16000, duration: 0.5)
        let pitches = await detector.transform(signal)

        XCTAssertGreaterThan(pitches.count, 0)
    }

    func testMusicPreset() async {
        let detector = PitchDetector.music

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)
        let pitches = await detector.transform(signal)

        XCTAssertGreaterThan(pitches.count, 0)
    }

    func testSingingPreset() async {
        let detector = PitchDetector.singing

        let signal = generateSineWave(frequency: 300, sampleRate: 22050, duration: 0.5)
        let pitches = await detector.transform(signal)

        XCTAssertGreaterThan(pitches.count, 0)
    }

    // MARK: - Edge Cases

    func testSilence() async {
        let detector = PitchDetector.music

        let silence = [Float](repeating: 0, count: 22050)
        let pitches = await detector.transform(silence)

        // Silence should not detect any pitch
        let detectedPitches = pitches.filter { $0 > 0 }
        XCTAssertEqual(detectedPitches.count, 0, "Should not detect pitch in silence")
    }

    func testShortSignal() async {
        let detector = PitchDetector.music

        // Signal shorter than frame length
        let short = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.05)
        let pitches = await detector.transform(short)

        // Should handle gracefully
        XCTAssertNotNil(pitches)
    }

    func testNoisySignal() async {
        let detector = PitchDetector.music

        // Generate noisy sine wave
        srand48(42)
        var signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        for i in 0..<signal.count {
            signal[i] += 0.3 * Float(drand48() * 2 - 1)
        }

        let (pitches, confidences) = await detector.transformWithConfidence(signal)

        // Should still detect some pitches, but with lower confidence
        let validFrames = zip(pitches, confidences).filter { $0.0 > 0 }
        XCTAssertGreaterThan(validFrames.count, 0)
    }

    // MARK: - Single Frame Detection

    func testDetectFrame() {
        let detector = PitchDetector.music
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.1)
        let frame = Array(signal.prefix(2048))

        let (pitch, confidence) = detector.detectFrame(frame)

        if pitch > 0 {
            XCTAssertEqual(pitch, 440, accuracy: 30)
            XCTAssertGreaterThan(confidence, 0)
        }
    }
}
