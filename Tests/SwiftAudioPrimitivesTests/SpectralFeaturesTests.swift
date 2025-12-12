import XCTest
@testable import SwiftAudioPrimitives

final class SpectralFeaturesTests: XCTestCase {

    // MARK: - Test Signals

    /// Generate a pure sine wave
    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    /// Generate white noise
    private func generateNoise(length: Int) -> [Float] {
        srand48(42)
        return (0..<length).map { _ in Float(drand48() * 2 - 1) }
    }

    // MARK: - SpectralFeatures Tests

    func testSpectralCentroid() async {
        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512
        ))

        // High frequency signal should have higher centroid
        let lowFreqSignal = generateSineWave(frequency: 200, sampleRate: 22050, duration: 1.0)
        let highFreqSignal = generateSineWave(frequency: 2000, sampleRate: 22050, duration: 1.0)

        let lowCentroid = await spectralFeatures.centroid(lowFreqSignal)
        let highCentroid = await spectralFeatures.centroid(highFreqSignal)

        XCTAssertGreaterThan(lowCentroid.count, 0)
        XCTAssertGreaterThan(highCentroid.count, 0)

        let avgLowCentroid = lowCentroid.reduce(0, +) / Float(lowCentroid.count)
        let avgHighCentroid = highCentroid.reduce(0, +) / Float(highCentroid.count)

        XCTAssertLessThan(avgLowCentroid, avgHighCentroid,
            "Higher frequency signal should have higher spectral centroid")
    }

    func testSpectralBandwidth() async {
        let spectralFeatures = SpectralFeatures()

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let bandwidth = await spectralFeatures.bandwidth(signal)

        XCTAssertGreaterThan(bandwidth.count, 0)

        // All values should be non-negative
        for val in bandwidth {
            XCTAssertGreaterThanOrEqual(val, 0)
            XCTAssertFalse(val.isNaN)
        }
    }

    func testSpectralRolloff() async {
        let spectralFeatures = SpectralFeatures()

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let rolloff = await spectralFeatures.rolloff(signal)

        XCTAssertGreaterThan(rolloff.count, 0)

        // Rolloff should be in valid frequency range
        for val in rolloff {
            XCTAssertGreaterThanOrEqual(val, 0)
            XCTAssertLessThanOrEqual(val, 22050 / 2)
        }
    }

    func testSpectralFlatness() async {
        let spectralFeatures = SpectralFeatures()

        // White noise should have high flatness
        let noise = generateNoise(length: 22050)
        let noiseFlat = await spectralFeatures.flatness(noise)

        // Pure tone should have low flatness
        let tone = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let toneFlat = await spectralFeatures.flatness(tone)

        XCTAssertGreaterThan(noiseFlat.count, 0)
        XCTAssertGreaterThan(toneFlat.count, 0)

        // Average flatness values
        let avgNoiseFlat = noiseFlat.reduce(0, +) / Float(noiseFlat.count)
        let avgToneFlat = toneFlat.reduce(0, +) / Float(toneFlat.count)

        // Noise should be flatter than tone (higher flatness)
        XCTAssertGreaterThan(avgNoiseFlat, avgToneFlat,
            "White noise should have higher spectral flatness than pure tone")
    }

    func testSpectralFlux() async {
        let spectralFeatures = SpectralFeatures()

        // Stationary signal should have low flux
        let stationary = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let stationaryFlux = await spectralFeatures.flux(stationary)

        XCTAssertGreaterThan(stationaryFlux.count, 0)

        // All flux values should be non-negative
        for val in stationaryFlux {
            XCTAssertGreaterThanOrEqual(val, 0)
        }
    }

    func testExtractAll() async {
        let spectralFeatures = SpectralFeatures()
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let result = await spectralFeatures.extractAll(signal)

        XCTAssertGreaterThan(result.centroid.count, 0)
        XCTAssertGreaterThan(result.bandwidth.count, 0)
        XCTAssertGreaterThan(result.rolloff.count, 0)
        XCTAssertGreaterThan(result.flatness.count, 0)
        XCTAssertGreaterThan(result.flux.count, 0)

        // All arrays should have same length
        XCTAssertEqual(result.centroid.count, result.bandwidth.count)
        XCTAssertEqual(result.centroid.count, result.rolloff.count)
    }

    // MARK: - ZeroCrossingRate Tests

    func testZCRFramewise() {
        let zcr = ZeroCrossingRate()

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let zcrValues = zcr.transform(signal)

        XCTAssertGreaterThan(zcrValues.count, 0)

        // ZCR should be between 0 and 1
        for val in zcrValues {
            XCTAssertGreaterThanOrEqual(val, 0)
            XCTAssertLessThanOrEqual(val, 1)
        }
    }

    func testZCRGlobal() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let globalZCR = ZeroCrossingRate.compute(signal)

        XCTAssertGreaterThan(globalZCR, 0)
        XCTAssertLessThan(globalZCR, 1)
    }

    func testZCRHigherForHighFrequency() {
        // Higher frequency signal should have higher ZCR
        let lowFreq = generateSineWave(frequency: 100, sampleRate: 22050, duration: 1.0)
        let highFreq = generateSineWave(frequency: 1000, sampleRate: 22050, duration: 1.0)

        let lowZCR = ZeroCrossingRate.compute(lowFreq)
        let highZCR = ZeroCrossingRate.compute(highFreq)

        XCTAssertLessThan(lowZCR, highZCR,
            "Higher frequency signal should have higher ZCR")
    }

    func testZCRWithThreshold() {
        let noise = generateNoise(length: 22050)

        let zcrNoThreshold = ZeroCrossingRate.compute(noise)
        let zcrWithThreshold = ZeroCrossingRate.computeWithThreshold(noise, threshold: 0.1)

        // ZCR with threshold should be lower (fewer crossings count)
        XCTAssertLessThanOrEqual(zcrWithThreshold, zcrNoThreshold)
    }

    func testZCRPositions() {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.1)
        let positions = ZeroCrossingRate.positions(signal)

        XCTAssertGreaterThan(positions.count, 0)

        // Positions should be monotonically increasing
        for i in 1..<positions.count {
            XCTAssertGreaterThan(positions[i], positions[i-1])
        }
    }

    // MARK: - Edge Cases

    func testEmptySignal() async {
        let spectralFeatures = SpectralFeatures()
        let result = await spectralFeatures.extractAll([])

        XCTAssertEqual(result.centroid.count, 0)
    }

    func testConstantSignal() {
        let constant = [Float](repeating: 0.5, count: 22050)
        let zcr = ZeroCrossingRate.compute(constant)

        // Constant signal should have zero crossings only if it's around zero
        XCTAssertEqual(zcr, 0)
    }

    // MARK: - Presets

    func testPresets() {
        let speech = ZeroCrossingRate.speechAnalysis
        XCTAssertEqual(speech.frameLength, 512)

        let music = ZeroCrossingRate.musicAnalysis
        XCTAssertEqual(music.frameLength, 2048)
    }
}
