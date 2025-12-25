import XCTest
@testable import SwiftRosaEffects
import SwiftRosaCore

/// Tests for the Phase Vocoder time-stretching and pitch-shifting algorithms.
final class PhaseVocoderTests: XCTestCase {

    // MARK: - Basic Functionality Tests

    /// Test that time stretching produces output of expected length
    func testTimeStretchOutputLength() async {
        let vocoder = PhaseVocoder.standard
        let signal = generateSineWave(frequency: 440, duration: 1.0, sampleRate: 22050)

        // Slow down by 2x (rate = 0.5) should produce ~2x length
        let slowed = await vocoder.timeStretch(signal, rate: 0.5)
        let expectedLength = signal.count * 2
        XCTAssertGreaterThan(slowed.count, Int(Float(expectedLength) * 0.9),
            "0.5x rate should produce approximately 2x length")
        XCTAssertLessThan(slowed.count, Int(Float(expectedLength) * 1.1),
            "0.5x rate should produce approximately 2x length")

        // Speed up by 2x (rate = 2.0) should produce ~0.5x length
        let sped = await vocoder.timeStretch(signal, rate: 2.0)
        let expectedLength2 = signal.count / 2
        XCTAssertGreaterThan(sped.count, Int(Float(expectedLength2) * 0.9),
            "2x rate should produce approximately 0.5x length")
        XCTAssertLessThan(sped.count, Int(Float(expectedLength2) * 1.1),
            "2x rate should produce approximately 0.5x length")
    }

    /// Test that rate = 1.0 returns original signal
    func testTimeStretchIdentity() async {
        let vocoder = PhaseVocoder.standard
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        let result = await vocoder.timeStretch(signal, rate: 1.0)

        XCTAssertEqual(result.count, signal.count, "Rate 1.0 should preserve length")
        for (i, (orig, res)) in zip(signal, result).enumerated() {
            XCTAssertEqual(orig, res, accuracy: 1e-6, "Rate 1.0 should preserve signal at index \(i)")
        }
    }

    /// Test empty signal handling
    func testTimeStretchEmptySignal() async {
        let vocoder = PhaseVocoder.standard
        let result = await vocoder.timeStretch([], rate: 0.5)
        XCTAssertTrue(result.isEmpty, "Empty input should produce empty output")
    }

    /// Test invalid rate (0 or negative)
    func testTimeStretchInvalidRate() async {
        let vocoder = PhaseVocoder.standard
        let signal = generateSineWave(frequency: 440, duration: 0.1, sampleRate: 22050)

        // Rate = 0 should return original (guard condition)
        let result = await vocoder.timeStretch(signal, rate: 0)
        XCTAssertEqual(result.count, signal.count, "Rate 0 should return original")
    }

    // MARK: - Extreme Rate Tests

    /// Test extreme slow down (4x slower, rate = 0.25)
    func testTimeStretchExtremeSlow() async {
        let vocoder = PhaseVocoder.fast  // Use fast for reasonable test time
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        let slowed = await vocoder.timeStretch(signal, rate: 0.25)

        // Should not produce NaN or infinity
        XCTAssertFalse(slowed.contains { $0.isNaN }, "4x slowdown should not produce NaN")
        XCTAssertFalse(slowed.contains { $0.isInfinite }, "4x slowdown should not produce infinity")

        // Should produce approximately 4x length
        let expectedLength = signal.count * 4
        XCTAssertGreaterThan(slowed.count, Int(Float(expectedLength) * 0.8),
            "0.25x rate should produce approximately 4x length")
    }

    /// Test extreme speed up (4x faster, rate = 4.0)
    func testTimeStretchExtremeFast() async {
        let vocoder = PhaseVocoder.fast
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        let sped = await vocoder.timeStretch(signal, rate: 4.0)

        // Should not produce NaN or infinity
        XCTAssertFalse(sped.contains { $0.isNaN }, "4x speedup should not produce NaN")
        XCTAssertFalse(sped.contains { $0.isInfinite }, "4x speedup should not produce infinity")

        // Should produce approximately 0.25x length
        let expectedLength = signal.count / 4
        XCTAssertGreaterThan(sped.count, Int(Float(expectedLength) * 0.8),
            "4x rate should produce approximately 0.25x length")
    }

    /// Test very extreme slow down (8x slower)
    func testTimeStretchVeryExtremeSlow() async {
        let vocoder = PhaseVocoder.fast
        let signal = generateSineWave(frequency: 440, duration: 0.25, sampleRate: 22050)

        let slowed = await vocoder.timeStretch(signal, rate: 0.125)

        // Should not produce NaN or infinity
        XCTAssertFalse(slowed.contains { $0.isNaN }, "8x slowdown should not produce NaN")
        XCTAssertFalse(slowed.contains { $0.isInfinite }, "8x slowdown should not produce infinity")
    }

    /// Test very extreme speed up (8x faster)
    func testTimeStretchVeryExtremeFast() async {
        let vocoder = PhaseVocoder.fast
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        let sped = await vocoder.timeStretch(signal, rate: 8.0)

        // Should not produce NaN or infinity
        XCTAssertFalse(sped.contains { $0.isNaN }, "8x speedup should not produce NaN")
        XCTAssertFalse(sped.contains { $0.isInfinite }, "8x speedup should not produce infinity")

        // Should still have some output
        XCTAssertGreaterThan(sped.count, 0, "8x speedup should produce some output")
    }

    // MARK: - Pitch Shift Tests

    /// Test pitch shift produces output of original length
    func testPitchShiftPreservesLength() async {
        let vocoder = PhaseVocoder.fast
        let sampleRate: Float = 22050
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: sampleRate)

        // Shift up 2 semitones
        let shifted = await vocoder.pitchShift(signal, semitones: 2, sampleRate: sampleRate)

        XCTAssertEqual(shifted.count, signal.count, "Pitch shift should preserve original length")
    }

    /// Test pitch shift identity (0 semitones)
    func testPitchShiftIdentity() async {
        let vocoder = PhaseVocoder.standard
        let sampleRate: Float = 22050
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: sampleRate)

        let result = await vocoder.pitchShift(signal, semitones: 0, sampleRate: sampleRate)

        XCTAssertEqual(result.count, signal.count, "0 semitones should preserve length")
        for (i, (orig, res)) in zip(signal, result).enumerated() {
            XCTAssertEqual(orig, res, accuracy: 1e-6, "0 semitones should preserve signal at index \(i)")
        }
    }

    /// Test extreme pitch shift up (+12 semitones = 1 octave)
    func testPitchShiftOctaveUp() async {
        let vocoder = PhaseVocoder.fast
        let sampleRate: Float = 22050
        let signal = generateSineWave(frequency: 220, duration: 0.3, sampleRate: sampleRate)

        let shifted = await vocoder.pitchShift(signal, semitones: 12, sampleRate: sampleRate)

        // Should not produce NaN or infinity
        XCTAssertFalse(shifted.contains { $0.isNaN }, "Octave up should not produce NaN")
        XCTAssertFalse(shifted.contains { $0.isInfinite }, "Octave up should not produce infinity")
        XCTAssertEqual(shifted.count, signal.count, "Octave up should preserve length")
    }

    /// Test extreme pitch shift down (-12 semitones = 1 octave)
    func testPitchShiftOctaveDown() async {
        let vocoder = PhaseVocoder.fast
        let sampleRate: Float = 22050
        let signal = generateSineWave(frequency: 880, duration: 0.3, sampleRate: sampleRate)

        let shifted = await vocoder.pitchShift(signal, semitones: -12, sampleRate: sampleRate)

        // Should not produce NaN or infinity
        XCTAssertFalse(shifted.contains { $0.isNaN }, "Octave down should not produce NaN")
        XCTAssertFalse(shifted.contains { $0.isInfinite }, "Octave down should not produce infinity")
        XCTAssertEqual(shifted.count, signal.count, "Octave down should preserve length")
    }

    /// Test extreme pitch shift (+24 semitones = 2 octaves)
    func testPitchShiftTwoOctavesUp() async {
        let vocoder = PhaseVocoder.fast
        let sampleRate: Float = 22050
        let signal = generateSineWave(frequency: 220, duration: 0.2, sampleRate: sampleRate)

        let shifted = await vocoder.pitchShift(signal, semitones: 24, sampleRate: sampleRate)

        // Should not produce NaN or infinity
        XCTAssertFalse(shifted.contains { $0.isNaN }, "2 octaves up should not produce NaN")
        XCTAssertFalse(shifted.contains { $0.isInfinite }, "2 octaves up should not produce infinity")
    }

    // MARK: - Phase Coherence Tests

    /// Test that phase is coherent (output is smooth, not discontinuous)
    func testPhaseCoherence() async {
        let vocoder = PhaseVocoder.standard
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        let stretched = await vocoder.timeStretch(signal, rate: 0.8)

        // Check for large discontinuities (phase jumps would cause clicks)
        var maxDiff: Float = 0
        for i in 1..<stretched.count {
            let diff = abs(stretched[i] - stretched[i-1])
            maxDiff = max(maxDiff, diff)
        }

        // Maximum sample-to-sample difference should be reasonable
        // For a smooth signal, this should be much less than 2.0
        XCTAssertLessThan(maxDiff, 1.0,
            "Phase-coherent output should not have large discontinuities")
    }

    /// Test spectrogram stretching directly
    func testSpectrogramStretch() async {
        let vocoder = PhaseVocoder.standard
        let signal = generateSineWave(frequency: 440, duration: 0.5, sampleRate: 22050)

        // First get the STFT
        let stftConfig = STFTConfig(uncheckedNFFT: 2048, hopLength: 512)
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // Stretch the spectrogram
        let stretchedSpec = vocoder.stretchSpectrogram(spectrogram, rate: 0.5)

        // Output should have approximately 2x frames
        let expectedFrames = spectrogram.cols * 2
        XCTAssertGreaterThan(stretchedSpec.cols, Int(Float(expectedFrames) * 0.9),
            "0.5x rate should produce ~2x frames")
        XCTAssertLessThan(stretchedSpec.cols, Int(Float(expectedFrames) * 1.1),
            "0.5x rate should produce ~2x frames")

        // Should preserve frequency count
        XCTAssertEqual(stretchedSpec.rows, spectrogram.rows,
            "Stretching should preserve frequency count")

        // Check for NaN/Inf
        for row in stretchedSpec.real {
            XCTAssertFalse(row.contains { $0.isNaN }, "Stretched spectrogram should not contain NaN")
            XCTAssertFalse(row.contains { $0.isInfinite }, "Stretched spectrogram should not contain infinity")
        }
        for row in stretchedSpec.imag {
            XCTAssertFalse(row.contains { $0.isNaN }, "Stretched spectrogram should not contain NaN")
            XCTAssertFalse(row.contains { $0.isInfinite }, "Stretched spectrogram should not contain infinity")
        }
    }

    // MARK: - Presets Tests

    /// Test standard preset configuration
    func testStandardPreset() async {
        let vocoder = PhaseVocoder.standard
        XCTAssertEqual(vocoder.config.nFFT, 2048)
        XCTAssertEqual(vocoder.config.hopLength, 512)
        XCTAssertEqual(vocoder.config.windowType, .hann)

        // Should work without error
        let signal = generateSineWave(frequency: 440, duration: 0.1, sampleRate: 22050)
        let result = await vocoder.timeStretch(signal, rate: 0.8)
        XCTAssertFalse(result.isEmpty)
    }

    /// Test high-quality preset configuration
    func testHighQualityPreset() async {
        let vocoder = PhaseVocoder.highQuality
        XCTAssertEqual(vocoder.config.nFFT, 4096)
        XCTAssertEqual(vocoder.config.hopLength, 1024)
        XCTAssertEqual(vocoder.config.windowType, .hann)

        // Should work without error
        let signal = generateSineWave(frequency: 440, duration: 0.1, sampleRate: 22050)
        let result = await vocoder.timeStretch(signal, rate: 0.8)
        XCTAssertFalse(result.isEmpty)
    }

    /// Test fast preset configuration
    func testFastPreset() async {
        let vocoder = PhaseVocoder.fast
        XCTAssertEqual(vocoder.config.nFFT, 1024)
        XCTAssertEqual(vocoder.config.hopLength, 256)
        XCTAssertEqual(vocoder.config.windowType, .hann)

        // Should work without error
        let signal = generateSineWave(frequency: 440, duration: 0.1, sampleRate: 22050)
        let result = await vocoder.timeStretch(signal, rate: 0.8)
        XCTAssertFalse(result.isEmpty)
    }

    // MARK: - Edge Cases

    /// Test with very short signal (shorter than FFT)
    func testVeryShortSignal() async {
        let vocoder = PhaseVocoder.fast  // nFFT = 1024
        let signal = [Float](repeating: 0.5, count: 100)  // Much shorter than FFT

        let result = await vocoder.timeStretch(signal, rate: 0.5)

        // Should handle gracefully (may pad internally)
        XCTAssertFalse(result.contains { $0.isNaN }, "Short signal should not produce NaN")
    }

    /// Test with DC signal (constant value)
    func testDCSignal() async {
        let vocoder = PhaseVocoder.fast
        let signal = [Float](repeating: 0.5, count: 4000)

        let result = await vocoder.timeStretch(signal, rate: 0.8)

        // DC should remain relatively constant (no time-varying frequency content)
        XCTAssertFalse(result.contains { $0.isNaN }, "DC signal should not produce NaN")
        XCTAssertFalse(result.contains { $0.isInfinite }, "DC signal should not produce infinity")
    }

    /// Test with silence
    func testSilentSignal() async {
        let vocoder = PhaseVocoder.fast
        let signal = [Float](repeating: 0, count: 4000)

        let result = await vocoder.timeStretch(signal, rate: 0.5)

        // Should produce silence (or near-silence)
        XCTAssertFalse(result.contains { $0.isNaN }, "Silent signal should not produce NaN")
        let maxAbs = result.map { abs($0) }.max() ?? 0
        XCTAssertLessThan(maxAbs, 1e-6, "Silent input should produce silent output")
    }

    /// Test with impulse signal
    func testImpulseSignal() async {
        let vocoder = PhaseVocoder.fast
        var signal = [Float](repeating: 0, count: 4000)
        signal[2000] = 1.0  // Single impulse

        let result = await vocoder.timeStretch(signal, rate: 0.5)

        // Should not produce NaN or infinity
        XCTAssertFalse(result.contains { $0.isNaN }, "Impulse signal should not produce NaN")
        XCTAssertFalse(result.contains { $0.isInfinite }, "Impulse signal should not produce infinity")
    }

    // MARK: - Helper Functions

    private func generateSineWave(frequency: Float, duration: Float, sampleRate: Float) -> [Float] {
        let numSamples = Int(duration * sampleRate)
        var signal = [Float](repeating: 0, count: numSamples)
        let omega = 2.0 * Float.pi * frequency / sampleRate
        for i in 0..<numSamples {
            signal[i] = sin(omega * Float(i))
        }
        return signal
    }
}
