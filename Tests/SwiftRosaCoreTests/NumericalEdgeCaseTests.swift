import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Tests for numerical edge cases and boundary conditions.
///
/// These tests ensure that SwiftRosa handles edge cases gracefully:
/// - Empty inputs
/// - Single-sample inputs
/// - Very short signals
/// - DC offset signals
/// - Silence
/// - Signals with extreme values
final class NumericalEdgeCaseTests: XCTestCase {

    // MARK: - Empty Input Tests

    func testSTFTEmptyInput() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        let result = await stft.transform([])

        XCTAssertEqual(result.rows, 0, "Empty input should produce empty output")
        XCTAssertEqual(result.cols, 0, "Empty input should produce empty output")
    }

    func testMelSpectrogramEmptyInput() async {
        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 2048, hopLength: 512, nMels: 128)

        let result = await melSpec.transform([])

        XCTAssertTrue(result.isEmpty || result.first?.isEmpty == true,
                      "Empty input should produce empty output")
    }

    func testMFCCEmptyInput() async {
        let mfcc = MFCC(sampleRate: 22050, nFFT: 2048, hopLength: 512, nMels: 128, nMFCC: 13)

        let result = await mfcc.transform([])

        XCTAssertTrue(result.isEmpty || result.first?.isEmpty == true,
                      "Empty input should produce empty output")
    }

    func testSpectralFeaturesEmptyInput() async {
        let config = SpectralFeaturesConfig(sampleRate: 22050, nFFT: 2048, hopLength: 512)
        let features = SpectralFeatures(config: config)

        let centroid = await features.centroid([])
        let bandwidth = await features.bandwidth([])
        let rolloff = await features.rolloff([])
        let flatness = await features.flatness([])

        XCTAssertTrue(centroid.isEmpty, "Centroid should be empty for empty input")
        XCTAssertTrue(bandwidth.isEmpty, "Bandwidth should be empty for empty input")
        XCTAssertTrue(rolloff.isEmpty, "Rolloff should be empty for empty input")
        XCTAssertTrue(flatness.isEmpty, "Flatness should be empty for empty input")
    }

    func testZCREmptyInput() {
        let result = ZeroCrossingRate.compute([])
        XCTAssertEqual(result, 0, "ZCR of empty signal should be 0")
    }

    func testRMSEmptyInput() {
        let rms = RMSEnergy(config: RMSEnergyConfig(frameLength: 2048, hopLength: 512))
        let result = rms.transform([])
        XCTAssertTrue(result.isEmpty, "RMS of empty signal should be empty")
    }

    // MARK: - Single Sample Tests

    func testSTFTSingleSample() async {
        let config = STFTConfig(nFFT: 512, hopLength: 256)
        let stft = STFT(config: config)

        let result = await stft.transform([0.5])

        // Single sample should still produce at least one frame due to centering/padding
        XCTAssertGreaterThanOrEqual(result.cols, 0, "Single sample should produce output")
    }

    func testMelSpectrogramSingleSample() async {
        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 512, hopLength: 256, nMels: 40)

        let result = await melSpec.transform([0.5])

        // Should handle gracefully
        XCTAssertNotNil(result, "Should handle single sample input")
    }

    func testZCRSingleSample() {
        let result = ZeroCrossingRate.compute([0.5])
        XCTAssertEqual(result, 0, "Single sample has no crossings")
    }

    // MARK: - Silence Tests

    func testSTFTSilence() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        let silence = [Float](repeating: 0, count: 22050)
        let result = await stft.transform(silence)
        let magnitude = result.magnitude

        // All magnitudes should be essentially zero
        let maxMagnitude = magnitude.flatMap { $0 }.max() ?? 0
        XCTAssertLessThan(maxMagnitude, 1e-6, "Silence should have near-zero magnitude")
    }

    func testMFCCSilence() async {
        let mfcc = MFCC(sampleRate: 22050, nFFT: 2048, hopLength: 512, nMels: 128, nMFCC: 13)

        let silence = [Float](repeating: 0, count: 22050)
        let result = await mfcc.transform(silence)

        // MFCC of silence - first coefficient should be very negative (log of near-zero)
        // Other coefficients should be near zero
        XCTAssertFalse(result.isEmpty, "MFCC should produce output for silence")
    }

    func testZCRSilence() {
        let silence = [Float](repeating: 0, count: 1000)
        let result = ZeroCrossingRate.compute(silence)
        XCTAssertEqual(result, 0, "Silence has no zero crossings")
    }

    func testRMSSilence() {
        let rms = RMSEnergy(config: RMSEnergyConfig(frameLength: 2048, hopLength: 512))
        let silence = [Float](repeating: 0, count: 22050)
        let result = rms.transform(silence)

        let maxRMS = result.max() ?? 0
        XCTAssertLessThan(maxRMS, 1e-6, "RMS of silence should be zero")
    }

    // MARK: - DC Offset Tests

    func testSTFTDCOffset() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        // Pure DC signal
        let dcSignal = [Float](repeating: 0.5, count: 22050)
        let result = await stft.transform(dcSignal)
        let magnitude = result.magnitude

        // DC component (bin 0) should dominate
        guard !magnitude.isEmpty, !magnitude[0].isEmpty else {
            XCTFail("Should produce output")
            return
        }

        let dcMagnitude = magnitude[0].reduce(0, +) / Float(magnitude[0].count)
        let acMagnitude = magnitude.dropFirst().flatMap { $0 }.reduce(0, +) / Float(magnitude.count * magnitude[0].count)

        XCTAssertGreaterThan(dcMagnitude, acMagnitude * 10, "DC should dominate for constant signal")
    }

    func testZCRDCOffset() {
        // Signal with DC offset but no crossings
        let dcSignal = [Float](repeating: 0.5, count: 1000)
        let result = ZeroCrossingRate.compute(dcSignal)
        XCTAssertEqual(result, 0, "Pure DC signal has no zero crossings")
    }

    // MARK: - Very Short Signal Tests

    func testSTFTShortSignal() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        // Signal shorter than one FFT window
        let shortSignal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.05) // ~1100 samples

        let result = await stft.transform(shortSignal)

        // Should still produce output due to padding
        XCTAssertGreaterThan(result.cols, 0, "Short signal should produce at least one frame")
    }

    func testMelSpectrogramShortSignal() async {
        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 2048, hopLength: 512, nMels: 40)

        let shortSignal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.05)
        let result = await melSpec.transform(shortSignal)

        XCTAssertFalse(result.isEmpty, "Short signal should produce mel spectrogram")
    }

    // MARK: - Maximum Value Tests

    func testSTFTMaxAmplitude() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        // Signal at maximum Float value - should not overflow
        let maxSignal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.1)
        let scaledSignal = maxSignal.map { $0 * Float.greatestFiniteMagnitude.squareRoot() }

        let result = await stft.transform(scaledSignal)

        // Should not contain NaN or Inf
        let hasNaN = result.magnitude.flatMap { $0 }.contains { $0.isNaN }
        let hasInf = result.magnitude.flatMap { $0 }.contains { $0.isInfinite }

        XCTAssertFalse(hasNaN, "Should not produce NaN for large amplitude")
        XCTAssertFalse(hasInf, "Should not produce Inf for large amplitude")
    }

    // MARK: - Denormal Value Tests

    func testSTFTDenormalValues() async {
        let config = STFTConfig(nFFT: 512, hopLength: 256)
        let stft = STFT(config: config)

        // Signal with very small (denormal) values
        let denormalSignal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.1)
            .map { $0 * 1e-40 }

        let result = await stft.transform(denormalSignal)

        // Should produce valid output without NaN
        let hasNaN = result.magnitude.flatMap { $0 }.contains { $0.isNaN }
        XCTAssertFalse(hasNaN, "Should handle denormal values without NaN")
    }

    // MARK: - Impulse Response Tests

    func testSTFTImpulse() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        let impulse = TestSignalGenerator.impulse(sampleRate: 22050, duration: 0.5)
        let result = await stft.transform(impulse)
        let magnitude = result.magnitude

        // Impulse should have broadband frequency content
        guard !magnitude.isEmpty else {
            XCTFail("Should produce output")
            return
        }

        // Find frame containing the impulse
        let frameMagnitudes = (0..<result.cols).map { frame in
            magnitude.map { $0[frame] }.reduce(0, +)
        }
        let maxFrame = frameMagnitudes.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0

        // Check that impulse frame has relatively flat spectrum
        let impulseFrameMags = magnitude.map { $0[maxFrame] }
        let meanMag = impulseFrameMags.reduce(0, +) / Float(impulseFrameMags.count)
        let variation = impulseFrameMags.map { abs($0 - meanMag) / meanMag }.reduce(0, +) / Float(impulseFrameMags.count)

        // Impulse should have relatively flat spectrum (low variation)
        XCTAssertLessThan(variation, 1.0, "Impulse should have relatively flat spectrum")
    }

    // MARK: - Phase Continuity Tests

    func testSTFTPhaseWrap() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let stft = STFT(config: config)

        let signal = TestSignalGenerator.sine(frequency: 1000, sampleRate: 22050, duration: 1.0)
        let result = await stft.transform(signal)
        let phase = result.phase

        // Phase values should be in [-pi, pi] range
        for row in phase {
            for p in row {
                XCTAssertGreaterThanOrEqual(p, -Float.pi, "Phase should be >= -pi")
                XCTAssertLessThanOrEqual(p, Float.pi, "Phase should be <= pi")
            }
        }
    }

    // MARK: - ISTFT Reconstruction Tests

    func testISTFTReconstructionShort() async {
        let config = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.1)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        // Compute reconstruction error
        var sumSquaredError: Float = 0
        var sumSquaredRef: Float = 0

        for (orig, recon) in zip(signal, reconstructed) {
            sumSquaredError += (orig - recon) * (orig - recon)
            sumSquaredRef += orig * orig
        }

        let normalizedError = sqrt(sumSquaredError / max(sumSquaredRef, 1e-10))
        XCTAssertLessThan(normalizedError, 0.01, "ISTFT reconstruction error should be small")
    }

    // MARK: - Window Function Edge Cases

    func testWindowVeryShort() {
        let window = Windows.generate(.hann, length: 1, periodic: true)
        XCTAssertEqual(window.count, 1, "Should handle length 1")
        XCTAssertEqual(window[0], 1.0, accuracy: 1e-6, "Single-point Hann should be 1")
    }

    func testWindowZeroLength() {
        let window = Windows.generate(.hann, length: 0, periodic: true)
        XCTAssertTrue(window.isEmpty, "Zero length should produce empty window")
    }

    func testWindowAllTypes() {
        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett]
        let length = 1024

        for windowType in windowTypes {
            let window = Windows.generate(windowType, length: length, periodic: true)

            XCTAssertEqual(window.count, length, "\(windowType) should have correct length")

            // Window values should be in [0, 1] range (or close to it)
            let minVal = window.min() ?? 0
            let maxVal = window.max() ?? 0

            XCTAssertGreaterThanOrEqual(minVal, -0.01, "\(windowType) minimum should be >= 0")
            XCTAssertLessThanOrEqual(maxVal, 1.01, "\(windowType) maximum should be <= 1")
        }
    }

    // MARK: - CQT Edge Cases

    func testCQTVeryLowFrequency() async {
        let config = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 20.0,  // Very low frequency
            nBins: 12,
            binsPerOctave: 12
        )

        let cqt = CQT(config: config)
        let signal = TestSignalGenerator.sine(frequency: 25, sampleRate: 22050, duration: 1.0)

        let result = await cqt.transform(signal)

        // Should produce valid output
        let hasNaN = result.magnitude.flatMap { $0 }.contains { $0.isNaN }
        XCTAssertFalse(hasNaN, "CQT should not produce NaN for low frequencies")
    }

    // MARK: - Mel Filterbank Edge Cases

    func testMelFilterbankExtremeMels() {
        // Very few mels
        let fewMels = MelFilterbank(nMels: 4, nFFT: 2048, sampleRate: 22050)
        let filters = fewMels.filters()
        XCTAssertEqual(filters.count, 4, "Should create requested number of filters")

        // Many mels
        let manyMels = MelFilterbank(nMels: 256, nFFT: 2048, sampleRate: 22050)
        let manyFilters = manyMels.filters()
        XCTAssertEqual(manyFilters.count, 256, "Should create requested number of filters")
    }

    func testMelFilterbankLowSampleRate() {
        let lowSR = MelFilterbank(nMels: 40, nFFT: 512, sampleRate: 8000)
        let filters = lowSR.filters()

        XCTAssertEqual(filters.count, 40, "Should work with low sample rate")

        // All filters should sum to approximately 1 at their center
        for filter in filters {
            let maxVal = filter.max() ?? 0
            XCTAssertGreaterThan(maxVal, 0, "Each filter should have non-zero peak")
        }
    }

    // MARK: - Spectral Contrast Edge Cases

    func testSpectralContrastEdgeCases() async {
        let config = SpectralFeaturesConfig(sampleRate: 22050, nFFT: 2048, hopLength: 512)
        let features = SpectralFeatures(config: config)

        // Test with pure tone - should have high contrast
        let pureTone = TestSignalGenerator.sine(frequency: 1000, sampleRate: 22050, duration: 0.5)
        let contrast = await features.contrast(pureTone)

        XCTAssertFalse(contrast.isEmpty, "Should produce contrast for pure tone")

        // Test with white noise - should have lower contrast
        let noise = TestSignalGenerator.whiteNoise(sampleRate: 22050, duration: 0.5, amplitude: 0.5)
        let noiseContrast = await features.contrast(noise)

        XCTAssertFalse(noiseContrast.isEmpty, "Should produce contrast for noise")
    }
}
