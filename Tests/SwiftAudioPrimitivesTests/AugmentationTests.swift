import XCTest
@testable import SwiftAudioPrimitives

final class AugmentationTests: XCTestCase {

    // MARK: - Test Signals

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    // MARK: - SpecAugment Tests

    func testSpecAugmentFrequencyMask() {
        let specAugment = SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 10,
            timeMaskParam: 0,
            numFrequencyMasks: 2,
            numTimeMasks: 0
        ))

        // Create simple spectrogram (nFreqs x nFrames)
        var spectrogram: [[Float]] = []
        for f in 0..<80 {
            let row = [Float](repeating: Float(f), count: 100)
            spectrogram.append(row)
        }

        let augmented = specAugment.transform(spectrogram)

        XCTAssertEqual(augmented.count, spectrogram.count)
        XCTAssertEqual(augmented[0].count, spectrogram[0].count)
    }

    func testSpecAugmentTimeMask() {
        let specAugment = SpecAugment(config: SpecAugmentConfig(
            frequencyMaskParam: 0,
            timeMaskParam: 20,
            numFrequencyMasks: 0,
            numTimeMasks: 2
        ))

        var spectrogram: [[Float]] = []
        for _ in 0..<80 {
            spectrogram.append([Float](repeating: 1.0, count: 200))
        }

        let augmented = specAugment.transform(spectrogram)

        XCTAssertEqual(augmented.count, spectrogram.count)
    }

    func testSpecAugmentPresets() {
        let mild = SpecAugment.mild
        let libriSpeech = SpecAugment.libriSpeech
        let aggressive = SpecAugment.aggressive

        var spectrogram: [[Float]] = []
        for _ in 0..<80 {
            spectrogram.append([Float](repeating: 1.0, count: 100))
        }

        let _ = mild.transform(spectrogram)
        let _ = libriSpeech.transform(spectrogram)
        let _ = aggressive.transform(spectrogram)

        XCTAssert(true)
    }

    // MARK: - AudioAugmentation Tests

    func testAddNoise() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        // Use white noise
        let noisy = AudioAugmentation.addNoise(signal, noise: .white, snrDb: 10)

        XCTAssertEqual(noisy.count, signal.count)

        // Should be different from original
        var hasDifference = false
        for i in 0..<signal.count {
            if abs(noisy[i] - signal[i]) > 1e-6 {
                hasDifference = true
                break
            }
        }
        XCTAssertTrue(hasDifference, "Noisy signal should differ from original")
    }

    func testAddNoiseAtDifferentSNR() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let highSNR = AudioAugmentation.addNoise(signal, noise: .white, snrDb: 30)
        let lowSNR = AudioAugmentation.addNoise(signal, noise: .white, snrDb: 0)

        // High SNR should be closer to original
        var diffHighSNR: Float = 0
        var diffLowSNR: Float = 0

        for i in 0..<signal.count {
            diffHighSNR += abs(highSNR[i] - signal[i])
            diffLowSNR += abs(lowSNR[i] - signal[i])
        }

        XCTAssertLessThan(diffHighSNR, diffLowSNR,
            "High SNR should be closer to original")
    }

    func testRandomGain() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let gained = AudioAugmentation.randomGain(signal, minDb: 6, maxDb: 6)

        XCTAssertEqual(gained.count, signal.count)

        // Should have different amplitude
        let originalMax = signal.map { abs($0) }.max() ?? 0
        let gainedMax = gained.map { abs($0) }.max() ?? 0

        XCTAssertNotEqual(originalMax, gainedMax, accuracy: 0.01)
    }

    func testTimeShift() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let shifted = AudioAugmentation.timeShift(signal, shiftSamples: 1000)

        XCTAssertEqual(shifted.count, signal.count)
    }

    func testTrimOrPad() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        // Test padding
        let padded = AudioAugmentation.trimOrPad(signal, targetLength: signal.count * 2)
        XCTAssertEqual(padded.count, signal.count * 2)

        // Test trimming
        let trimmed = AudioAugmentation.trimOrPad(signal, targetLength: signal.count / 2)
        XCTAssertEqual(trimmed.count, signal.count / 2)
    }

    // MARK: - FeatureNormalization Tests

    func testMeanNormalization() {
        // Create features with non-zero mean
        var features: [[Float]] = []
        for f in 0..<13 {
            let row = [Float](repeating: Float(f) + 10, count: 50)
            features.append(row)
        }

        let normalized = FeatureNormalization.meanNormalize(features)

        XCTAssertEqual(normalized.count, features.count)

        // Each row should have approximately zero mean
        for row in normalized {
            let mean = row.reduce(0, +) / Float(row.count)
            XCTAssertEqual(mean, 0, accuracy: 0.01, "Mean should be near zero")
        }
    }

    func testMVNormalization() {
        var features: [[Float]] = []
        for f in 0..<13 {
            var row = [Float](repeating: 0, count: 50)
            for t in 0..<50 {
                row[t] = Float(f) * 2.0 + Float(t) * 0.1
            }
            features.append(row)
        }

        let normalized = FeatureNormalization.mvNormalize(features)

        XCTAssertEqual(normalized.count, features.count)

        // Each row should have approximately zero mean and unit variance
        for row in normalized {
            let mean = row.reduce(0, +) / Float(row.count)
            XCTAssertEqual(mean, 0, accuracy: 0.1, "Mean should be near zero")

            let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(row.count)
            XCTAssertEqual(variance, 1.0, accuracy: 0.1, "Variance should be near 1")
        }
    }

    func testMinMaxScaling() {
        var features: [[Float]] = []
        for f in 0..<13 {
            var row = [Float](repeating: 0, count: 50)
            for t in 0..<50 {
                row[t] = Float(f) * 10 + Float(t)
            }
            features.append(row)
        }

        let scaled = FeatureNormalization.minMaxScale(features, min: 0, max: 1)

        XCTAssertEqual(scaled.count, features.count)

        // All values should be in [0, 1]
        for row in scaled {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0)
                XCTAssertLessThanOrEqual(val, 1)
            }
        }
    }

    func testL2Normalization() {
        var features: [[Float]] = []
        for f in 0..<13 {
            features.append([Float](repeating: Float(f + 1), count: 50))
        }

        let normalized = FeatureNormalization.l2Normalize(features)

        // Each column should have unit L2 norm
        for t in 0..<50 {
            var sumSquares: Float = 0
            for f in 0..<13 {
                sumSquares += normalized[f][t] * normalized[f][t]
            }
            let norm = sqrt(sumSquares)
            XCTAssertEqual(norm, 1.0, accuracy: 0.01, "L2 norm should be 1")
        }
    }

    // MARK: - Edge Cases

    func testEmptySpectrogram() {
        let specAugment = SpecAugment.mild
        let result = specAugment.transform([])

        XCTAssertEqual(result.count, 0)
    }

    func testEmptyFeatures() {
        let normalized = FeatureNormalization.meanNormalize([])
        XCTAssertEqual(normalized.count, 0)
    }

    func testConstantFeatures() {
        // Constant features have zero variance
        var features: [[Float]] = []
        for _ in 0..<13 {
            features.append([Float](repeating: 5.0, count: 50))
        }

        let normalized = FeatureNormalization.mvNormalize(features)

        // Should handle gracefully (no NaN)
        for row in normalized {
            for val in row {
                XCTAssertFalse(val.isNaN)
            }
        }
    }
}
