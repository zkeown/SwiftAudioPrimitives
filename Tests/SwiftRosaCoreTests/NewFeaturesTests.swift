import XCTest
@testable import SwiftRosaCore

/// Tests for newly implemented librosa gap features.
final class NewFeaturesTests: XCTestCase {

    // MARK: - Test Utilities

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    private func generateNoise(length: Int, seed: Int = 42) -> [Float] {
        srand48(seed)
        return (0..<length).map { _ in Float(drand48() * 2 - 1) }
    }

    // MARK: - RMS Energy Tests

    func testRMSEnergyBasic() {
        let rms = RMSEnergy()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let energy = rms.transform(signal)

        XCTAssertGreaterThan(energy.count, 0, "Should produce output frames")

        // RMS of sine wave should be around 1/sqrt(2) ≈ 0.707
        let avgEnergy = energy.reduce(0, +) / Float(energy.count)
        XCTAssertGreaterThan(avgEnergy, 0.5, "Sine wave RMS should be > 0.5")
        XCTAssertLessThan(avgEnergy, 0.9, "Sine wave RMS should be < 0.9")
    }

    func testRMSEnergySilence() {
        let rms = RMSEnergy()
        let silence = [Float](repeating: 0, count: 22050)

        let energy = rms.transform(silence)

        for val in energy {
            XCTAssertEqual(val, 0, accuracy: 1e-6, "Silence should have zero energy")
        }
    }

    func testRMSEnergyDB() {
        let rms = RMSEnergy()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let dbEnergy = rms.transformDB(signal)

        XCTAssertGreaterThan(dbEnergy.count, 0)

        // dB values should be negative for signal < 1
        let avgDB = dbEnergy.reduce(0, +) / Float(dbEnergy.count)
        XCTAssertLessThan(avgDB, 0, "dB should be negative for normalized signal")
    }

    // MARK: - Delta Tests

    func testDeltaBasic() {
        // Create simple linear feature
        let features: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        let deltas = Delta.compute(features)

        XCTAssertEqual(deltas.count, 1)
        XCTAssertEqual(deltas[0].count, 10)

        // Delta of linear sequence should be constant (approximately 1)
        let avgDelta = deltas[0].reduce(0, +) / Float(deltas[0].count)
        XCTAssertGreaterThan(avgDelta, 0.5, "Delta of increasing sequence should be positive")
    }

    func testDeltaSecondOrder() {
        let features: [[Float]] = [[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]]  // Quadratic

        let deltaDeltas = Delta.compute(features, order: 2)

        XCTAssertEqual(deltaDeltas.count, 1)

        // Second derivative of quadratic should be approximately constant
        // (though edge effects will reduce this)
        for val in deltaDeltas[0] {
            XCTAssertFalse(val.isNaN, "Delta-delta should not be NaN")
        }
    }

    // MARK: - A-Weighting Tests

    func testAWeightingCurve() {
        // Use static method directly
        let weights = AWeighting.weights(nFFT: 2048, sampleRate: 22050)

        XCTAssertEqual(weights.count, 1025)  // nFFT/2 + 1

        // Find weight at ~1kHz (should be close to 0 dB)
        let binAt1kHz = Int(1000.0 / (22050.0 / 2048.0))
        XCTAssertGreaterThan(weights[binAt1kHz], -3, "1kHz should have weight near 0 dB")
        XCTAssertLessThan(weights[binAt1kHz], 3, "1kHz should have weight near 0 dB")

        // Low frequencies should have large negative weights
        XCTAssertLessThan(weights[1], weights[binAt1kHz], "Low freq should be attenuated")
    }

    func testAWeightingApplyToSpectrogram() {
        // Create a simple power spectrogram
        let nFreqs = 1025
        let nFrames = 10
        var powerSpec = [[Float]]()
        for _ in 0..<nFreqs {
            powerSpec.append([Float](repeating: 1.0, count: nFrames))
        }

        // Apply A-weighting to power spectrogram
        let weighted = AWeighting.apply(to: powerSpec, nFFT: 2048, sampleRate: 22050)

        XCTAssertEqual(weighted.count, nFreqs)
        XCTAssertEqual(weighted[0].count, nFrames)

        // Output should be in dB
        for row in weighted {
            for val in row {
                XCTAssertFalse(val.isNaN, "Weighted values should not be NaN")
            }
        }
    }

    func testAWeightingFrequencies() {
        // Test A-weighting at specific frequencies
        let frequencies: [Float] = [20, 100, 1000, 4000, 10000]
        let weights = AWeighting.weights(frequencies: frequencies)

        XCTAssertEqual(weights.count, 5)

        // At 1kHz, A-weighting should be approximately 0 dB
        XCTAssertEqual(weights[2], 0.0, accuracy: 0.5, "A-weighting at 1kHz should be ~0 dB")

        // At 20Hz, A-weighting should be very negative
        XCTAssertLessThan(weights[0], -40, "A-weighting at 20Hz should be < -40 dB")
    }

    // MARK: - Tone Generator Tests

    func testToneGeneratorBasic() {
        let tone = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 1.0)

        XCTAssertEqual(tone.count, 22050, "1 second at 22050 Hz = 22050 samples")

        // Check amplitude range
        let maxVal = tone.max() ?? 0
        let minVal = tone.min() ?? 0
        XCTAssertGreaterThan(maxVal, 0.9, "Sine should reach near 1")
        XCTAssertLessThan(minVal, -0.9, "Sine should reach near -1")
    }

    func testToneGeneratorFrequency() {
        // Generate 440 Hz tone
        let tone = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 0.1)

        // Count zero crossings to verify frequency
        var crossings = 0
        for i in 1..<tone.count {
            if (tone[i-1] < 0 && tone[i] >= 0) || (tone[i-1] >= 0 && tone[i] < 0) {
                crossings += 1
            }
        }

        // Expected: 440 cycles * 2 crossings per cycle * 0.1 seconds = 88 crossings
        let expectedCrossings = 440 * 2 * 0.1
        XCTAssertEqual(Float(crossings), Float(expectedCrossings), accuracy: 5,
                       "Zero crossings should match expected frequency")
    }

    func testToneGeneratorMultiTone() {
        // API is multiTone, not chord
        let multiTone = ToneGenerator.multiTone(frequencies: [261.63, 329.63, 392.00], sampleRate: 22050, duration: 0.5)

        XCTAssertEqual(multiTone.count, Int(22050 * 0.5))

        // Multi-tone should have lower amplitude than single tone (due to normalization)
        let maxVal = multiTone.max() ?? 0
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Multi-tone should be normalized")
    }

    // MARK: - Clicks Generator Tests

    func testClicksGeneratorBasic() {
        let times: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        // Use ClickGenerator (singular), not ClicksGenerator
        let clicks = ClickGenerator.clicks(times: times, sampleRate: 22050, length: 22050)

        XCTAssertEqual(clicks.count, 22050)

        // Check that we have energy at click positions
        let clickSample1 = Int(0.1 * 22050)
        let clickSample2 = Int(0.2 * 22050)

        // Area around click should have non-zero values
        let hasClick1 = clicks[(clickSample1-5)..<(clickSample1+5)].contains { abs($0) > 0.1 }
        let hasClick2 = clicks[(clickSample2-5)..<(clickSample2+5)].contains { abs($0) > 0.1 }

        XCTAssertTrue(hasClick1, "Should have click at 0.1s")
        XCTAssertTrue(hasClick2, "Should have click at 0.2s")
    }

    func testClicksGeneratorFrames() {
        let frames = [0, 100, 200, 300]
        let clicks = ClickGenerator.clicks(frames: frames, sampleRate: 22050, hopLength: 512, length: 22050)

        XCTAssertEqual(clicks.count, 22050)
    }

    // MARK: - Chirp Generator Tests

    func testChirpGeneratorLinear() {
        let chirp = ChirpGenerator.chirp(fMin: 100, fMax: 1000, sampleRate: 22050, duration: 1.0, type: .linear)

        XCTAssertEqual(chirp.count, 22050)

        // Check amplitude range
        let maxVal = chirp.max() ?? 0
        let minVal = chirp.min() ?? 0
        XCTAssertGreaterThan(maxVal, 0.9)
        XCTAssertLessThan(minVal, -0.9)
    }

    func testChirpGeneratorExponential() {
        let chirp = ChirpGenerator.chirp(fMin: 100, fMax: 10000, sampleRate: 22050, duration: 1.0, type: .exponential)

        XCTAssertEqual(chirp.count, 22050)

        for val in chirp {
            XCTAssertFalse(val.isNaN, "Chirp should not contain NaN")
            XCTAssertFalse(val.isInfinite, "Chirp should not contain Inf")
        }
    }

    // MARK: - Stack Memory Tests

    func testStackMemoryBasic() {
        let features: [[Float]] = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ]

        let stacked = StackMemory.stack(features, nSteps: 2, delay: 1)

        // Should have 2 original + 2 delayed = 4 feature rows
        XCTAssertEqual(stacked.count, 4)

        // Output frames should be same as input (padding handles edges)
        XCTAssertEqual(stacked[0].count, 5)
    }

    func testStackMemoryModeEdge() {
        let features: [[Float]] = [
            [1, 2, 3, 4, 5]
        ]

        // Use .edge instead of .post (which doesn't exist)
        let stacked = StackMemory.stack(features, nSteps: 3, delay: 1, mode: .edge)

        XCTAssertEqual(stacked.count, 3)
        XCTAssertEqual(stacked[0].count, 5)
    }

    func testStackMemoryModeConstant() {
        let features: [[Float]] = [
            [1, 2, 3, 4, 5]
        ]

        let stacked = StackMemory.stack(features, nSteps: 2, delay: 2, mode: .constant(0))

        XCTAssertEqual(stacked.count, 2)
        // First two values of delayed row should be 0 (constant padding)
        XCTAssertEqual(stacked[1][0], 0.0)
        XCTAssertEqual(stacked[1][1], 0.0)
    }

    // MARK: - Sync Features Tests

    func testSyncFeaturesBasic() {
        let features: [[Float]] = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]
        let boundaries = [0, 3, 7, 10]

        let synced = SyncFeatures.sync(features, boundaries: boundaries)

        // Should have 3 segments
        XCTAssertEqual(synced[0].count, 3)

        // First segment mean of [1,2,3] = 2
        XCTAssertEqual(synced[0][0], 2.0, accuracy: 0.01)
    }

    func testSyncFeaturesMedian() {
        let features: [[Float]] = [
            [1, 2, 100, 4, 5]  // 100 is outlier
        ]
        let boundaries = [0, 5]

        let syncedMean = SyncFeatures.sync(features, boundaries: boundaries, aggregation: .mean)
        let syncedMedian = SyncFeatures.sync(features, boundaries: boundaries, aggregation: .median)

        // Median should be more robust to outlier
        XCTAssertLessThan(syncedMedian[0][0], syncedMean[0][0])
    }

    func testTimesToFrames() {
        let times: [Float] = [0.0, 0.5, 1.0, 1.5]
        let frames = SyncFeatures.timesToFrames(times, sampleRate: 22050, hopLength: 512)

        XCTAssertEqual(frames[0], 0)
        XCTAssertEqual(frames[1], Int(0.5 * 22050 / 512))
        XCTAssertEqual(frames[2], Int(1.0 * 22050 / 512))
    }

    // MARK: - Perceptual Weighting Tests

    func testPerceptualWeightingCurveA() {
        // Use static method
        let aWeights = PerceptualWeighting.weightingCurve(.aWeighting, nFFT: 2048, sampleRate: 22050)

        XCTAssertEqual(aWeights.count, 1025)

        // At 1kHz bin, should be near 0 dB
        let binAt1kHz = Int(1000.0 / (22050.0 / 2048.0))
        XCTAssertEqual(aWeights[binAt1kHz], 0.0, accuracy: 1.0, "A-weighting at 1kHz should be ~0 dB")
    }

    func testPerceptualWeightingCurves() {
        let aWeights = PerceptualWeighting.weightingCurve(.aWeighting, nFFT: 2048, sampleRate: 22050)
        let cWeights = PerceptualWeighting.weightingCurve(.cWeighting, nFFT: 2048, sampleRate: 22050)

        XCTAssertEqual(aWeights.count, cWeights.count)

        // C-weighting is flatter than A-weighting at low frequencies
        let lowBin = 5
        XCTAssertGreaterThan(cWeights[lowBin], aWeights[lowBin],
                             "C-weighting should attenuate low frequencies less than A-weighting")
    }

    func testPerceptualWeightingApply() {
        // Create simple power spectrogram
        let nFreqs = 1025
        let nFrames = 10
        var powerSpec = [[Float]]()
        for _ in 0..<nFreqs {
            powerSpec.append([Float](repeating: 1.0, count: nFrames))
        }

        let weighted = PerceptualWeighting.weight(powerSpec, nFFT: 2048, sampleRate: 22050, curve: .aWeighting)

        XCTAssertEqual(weighted.count, nFreqs)
        XCTAssertEqual(weighted[0].count, nFrames)
    }

    // MARK: - PCEN Tests

    func testPCENBasic() {
        let pcen = PCEN()

        // Create simple spectrogram
        let spectrogram: [[Float]] = (0..<128).map { _ in
            (0..<100).map { _ in Float.random(in: 0.1...1.0) }
        }

        // PCEN transform is synchronous, not async
        let enhanced = pcen.transform(spectrogram)

        XCTAssertEqual(enhanced.count, 128)
        XCTAssertEqual(enhanced[0].count, 100)

        // PCEN output should be bounded
        for row in enhanced {
            for val in row {
                XCTAssertFalse(val.isNaN)
                XCTAssertFalse(val.isInfinite)
            }
        }
    }

    func testPCENDynamicRange() {
        let pcen = PCEN()

        // Create spectrogram with large dynamic range
        var spectrogram: [[Float]] = []
        for _ in 0..<64 {
            var row = [Float]()
            for j in 0..<50 {
                row.append(j < 25 ? 0.01 : 10.0)  // Low then high
            }
            spectrogram.append(row)
        }

        // Synchronous call
        let enhanced = pcen.transform(spectrogram)

        // PCEN should compress dynamic range
        let maxVal = enhanced.flatMap { $0 }.max() ?? 0
        let minVal = enhanced.flatMap { $0 }.min() ?? 0
        let range = maxVal - minVal

        XCTAssertLessThan(range, 10, "PCEN should compress dynamic range")
    }

    func testPCENWithState() {
        let pcen = PCEN()

        let spectrogram: [[Float]] = (0..<64).map { _ in
            (0..<20).map { _ in Float.random(in: 0.1...1.0) }
        }

        let (output, finalM) = pcen.transformWithState(spectrogram)

        XCTAssertEqual(output.count, 64)
        XCTAssertEqual(finalM.count, 64)

        // Final M should be positive
        for m in finalM {
            XCTAssertGreaterThan(m, 0)
        }
    }

    // MARK: - Poly Features Tests

    func testPolyFeaturesLinear() {
        let poly = PolyFeatures(config: PolyFeaturesConfig(order: 1))

        // Create spectrogram with linear spectral tilt
        var spectrogram: [[Float]] = []
        for f in 0..<1025 {
            let row = [Float](repeating: Float(f) / 1025.0, count: 10)
            spectrogram.append(row)
        }

        let coeffs = poly.transform(magnitudeSpec: spectrogram)

        // Order 1 = 2 coefficients (intercept + slope)
        XCTAssertEqual(coeffs.count, 2)
        XCTAssertEqual(coeffs[0].count, 10)

        // Linear spectrum should have positive slope coefficient
        let avgSlope = coeffs[1].reduce(0, +) / Float(coeffs[1].count)
        XCTAssertGreaterThan(avgSlope, 0, "Linear spectrum should have positive slope")
    }

    // MARK: - MR Frequencies Tests

    func testMRFrequenciesCQ() {
        let config = MRFrequenciesConfig(
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            scale: .cq
        )

        let bins = MRFrequencies.compute(config: config)

        XCTAssertEqual(bins.count, 84)

        // Check frequency progression is geometric
        let ratio = bins[12].centerFrequency / bins[0].centerFrequency
        XCTAssertEqual(ratio, 2.0, accuracy: 0.01, "12 bins should span one octave")

        // Q should be constant for CQ
        let q0 = bins[0].Q
        let q50 = bins[50].Q
        XCTAssertEqual(q0, q50, accuracy: 0.01, "Q should be constant for CQ")
    }

    func testMRFrequenciesVQT() {
        let config = MRFrequenciesConfig(
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            scale: .vqt,
            gamma: 20
        )

        let bins = MRFrequencies.compute(config: config)

        // VQT should have varying Q
        let q0 = bins[0].Q
        let q83 = bins[83].Q

        XCTAssertGreaterThan(q83, q0, "VQT Q should increase with frequency")
    }

    func testMRFrequenciesERB() {
        // Test ERB calculation
        let erb1000 = MRFrequencies.erb(1000)

        // ERB at 1000 Hz should be approximately 132 Hz
        XCTAssertEqual(erb1000, 132.639, accuracy: 1.0)
    }
}
