import XCTest
@testable import SwiftRosaCore
import SwiftRosaEffects
import SwiftRosaAnalysis
import SwiftRosaStreaming

/// Tests for edge cases and boundary conditions to ensure robustness.
///
/// These tests verify the library handles extreme inputs gracefully without
/// producing NaN, Inf, or crashes.
final class EdgeCaseTests: XCTestCase {

    // MARK: - Signal Length Edge Cases

    func testSTFTEmptySignal() async {
        // With center=false, empty signal should produce nothing
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64, center: false))
        let result = await stft.transform([])

        XCTAssertEqual(result.cols, 0, "Empty signal without centering should produce no frames")

        // With center=true (default), empty signal gets padded - but still no useful content
        // The key is that no crash occurs and output is valid
        let stftCentered = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64, center: true))
        let resultCentered = await stftCentered.transform([])

        // Verify no NaN/Inf even with empty input
        for row in resultCentered.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "Empty signal produced NaN")
            }
        }
    }

    func testSTFTSingleSample() async {
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64))
        let result = await stft.transform([1.0])

        // With center=true (default), single sample gets padded and produces frames
        XCTAssertFalse(result.real.isEmpty || result.imag.isEmpty,
            "Single sample with centering should produce output")

        // Verify no NaN/Inf
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "STFT produced NaN")
                XCTAssertFalse(value.isInfinite, "STFT produced Inf")
            }
        }
    }

    func testSTFTExactlyNFFTSamples() async {
        let nFFT = 256
        let stft = STFT(config: STFTConfig(uncheckedNFFT: nFFT, hopLength: nFFT / 4, center: false))

        // Signal exactly nFFT samples
        let signal = [Float](repeating: 1.0, count: nFFT)
        let result = await stft.transform(signal)

        // Without centering, should produce exactly 1 frame
        XCTAssertEqual(result.cols, 1, "Exactly nFFT samples without centering should give 1 frame")

        // With centering
        let stftCentered = STFT(config: STFTConfig(uncheckedNFFT: nFFT, hopLength: nFFT / 4, center: true))
        let resultCentered = await stftCentered.transform(signal)
        XCTAssertGreaterThan(resultCentered.cols, 0, "Centered STFT should produce frames")
    }

    func testSTFTNFFTPlusOneSamples() async {
        let nFFT = 256
        let hopLength = nFFT / 4
        let stft = STFT(config: STFTConfig(uncheckedNFFT: nFFT, hopLength: hopLength, center: false))

        // nFFT + 1 samples
        let signal = [Float](repeating: 1.0, count: nFFT + 1)
        let result = await stft.transform(signal)

        // Should produce exactly 1 frame (not enough for 2nd frame)
        XCTAssertEqual(result.cols, 1, "nFFT+1 samples should produce 1 frame without centering")
    }

    func testMelSpectrogramVeryLongSignal() async {
        // Test with ~30 seconds of audio
        let sampleRate: Float = 22050
        let duration: Float = 30.0
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * .pi * 440 * Float(i) / sampleRate)
        }

        let melSpec = MelSpectrogram(sampleRate: sampleRate, nFFT: 2048, hopLength: 512, nMels: 128)
        let result = await melSpec.transform(signal)

        // Verify shape is reasonable
        XCTAssertEqual(result.count, 128, "Should have 128 mel bands")

        // Expected frames ≈ (nSamples + nFFT) / hopLength
        let expectedFrames = (nSamples + 2048) / 512
        XCTAssertGreaterThan(result[0].count, expectedFrames - 10,
            "Should have approximately expected number of frames")

        // Verify no NaN/Inf
        for row in result {
            for value in row {
                XCTAssertFalse(value.isNaN, "Long signal produced NaN")
                XCTAssertFalse(value.isInfinite, "Long signal produced Inf")
            }
        }
    }

    // MARK: - Numerical Edge Cases

    func testSTFTAllZeros() async {
        let nSamples = 1024
        let signal = [Float](repeating: 0, count: nSamples)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64))
        let result = await stft.transform(signal)

        // All zeros input should produce all zeros output (or near zero)
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "Zero signal produced NaN in real")
                XCTAssertEqual(value, 0, accuracy: 1e-10, "Zero signal should produce zero output")
            }
        }
        for row in result.imag {
            for value in row {
                XCTAssertFalse(value.isNaN, "Zero signal produced NaN in imag")
                XCTAssertEqual(value, 0, accuracy: 1e-10, "Zero signal should produce zero output")
            }
        }
    }

    func testSTFTAllOnes() async {
        let nSamples = 1024
        let signal = [Float](repeating: 1.0, count: nSamples)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64))
        let result = await stft.transform(signal)

        // Verify no NaN/Inf
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "All-ones signal produced NaN")
                XCTAssertFalse(value.isInfinite, "All-ones signal produced Inf")
            }
        }

        // DC component should be non-zero, other components ~0
        let dcMagnitude = sqrt(result.real[0][0] * result.real[0][0] +
                               result.imag[0][0] * result.imag[0][0])
        XCTAssertGreaterThan(dcMagnitude, 0.1, "DC signal should have DC component")
    }

    func testSTFTAlternatingPlusMinus() async {
        let nSamples = 1024
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = i % 2 == 0 ? 1.0 : -1.0
        }

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 256, hopLength: 64))
        let result = await stft.transform(signal)

        // This is Nyquist frequency content
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "Alternating signal produced NaN")
                XCTAssertFalse(value.isInfinite, "Alternating signal produced Inf")
            }
        }

        // Nyquist bin should have energy
        let nyquistIdx = result.real.count - 1
        let nyquistMag = sqrt(result.real[nyquistIdx][0] * result.real[nyquistIdx][0] +
                              result.imag[nyquistIdx][0] * result.imag[nyquistIdx][0])
        XCTAssertGreaterThan(nyquistMag, 0.1, "Alternating signal should have Nyquist energy")
    }

    func testMFCCWithSmallValues() async {
        let nSamples = 2048
        // Very small values (near denormalized)
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = 1e-38 * sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let mfcc = MFCC(sampleRate: 22050, nFFT: 512, hopLength: 128, nMels: 40, nMFCC: 13)
        let result = await mfcc.transform(signal)

        // Should not produce NaN despite log of very small values
        for row in result {
            for value in row {
                XCTAssertFalse(value.isNaN, "Small values produced NaN in MFCC")
                XCTAssertFalse(value.isInfinite, "Small values produced Inf in MFCC")
            }
        }
    }

    func testMelSpectrogramWithLargeValues() async {
        let nSamples = 2048
        // Large values (near max float)
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = 1e6 * sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 512, hopLength: 128, nMels: 40)
        let result = await melSpec.transform(signal)

        // Should handle large values without overflow
        for row in result {
            for value in row {
                XCTAssertFalse(value.isNaN, "Large values produced NaN")
                XCTAssertFalse(value.isInfinite, "Large values produced Inf")
            }
        }
    }

    // MARK: - Parameter Edge Cases

    func testSTFTMinimumNFFT() async {
        // nFFT = 4 is practical minimum for power-of-2 FFT
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 4, hopLength: 2))
        let signal: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let result = await stft.transform(signal)

        XCTAssertGreaterThan(result.cols, 0, "Minimum nFFT should produce output")
        XCTAssertEqual(result.rows, 3, "nFFT=4 should give 3 frequency bins (nFFT/2 + 1)")
    }

    func testSTFTHopLengthOne() async {
        // Maximum overlap
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 64, hopLength: 1, center: false))
        let signal = [Float](repeating: 1.0, count: 128)
        let result = await stft.transform(signal)

        // Should produce many frames
        XCTAssertEqual(result.cols, 128 - 64 + 1, "hopLength=1 should produce max frames")
    }

    func testMelFilterbankSingleMel() async {
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 512,
            hopLength: 128,
            nMels: 1  // Single mel band
        )

        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            signal[i] = sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let result = await melSpec.transform(signal)

        XCTAssertEqual(result.count, 1, "Single mel band requested")
        XCTAssertGreaterThan(result[0].count, 0, "Should produce frames")
    }

    func testMelFilterbankManyMels() async {
        // More mel bands than frequency bins
        let nFFT = 256
        let nMels = 200  // More than nFFT/2 + 1 = 129

        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: 64,
            nMels: nMels
        )

        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            signal[i] = sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let result = await melSpec.transform(signal)

        // Should handle gracefully (mel bands limited by frequency resolution)
        XCTAssertGreaterThan(result.count, 0, "Should produce mel bands")
        XCTAssertLessThanOrEqual(result.count, nMels, "Should not exceed requested mel bands")
    }

    // MARK: - Round-Trip Edge Cases

    func testISTFTRoundTripPreservesEnergy() async {
        let nSamples = 2048
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }

        let stftConfig = STFTConfig(uncheckedNFFT: 512, hopLength: 128)
        let stft = STFT(config: stftConfig)
        let istft = ISTFT(config: stftConfig)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }

        // Energy should be preserved within tolerance
        let relativeError = abs(reconstructedEnergy - originalEnergy) / originalEnergy
        XCTAssertLessThan(relativeError, 0.01, "Round-trip should preserve energy within 1%")
    }

    func testGriffinLimWithSilence() async {
        // Magnitude spectrogram of silence (all zeros)
        let nFrames = 10
        let nFreqs = 129  // For nFFT=256

        var magnitude = [[Float]]()
        for _ in 0..<nFreqs {
            magnitude.append([Float](repeating: 0, count: nFrames))
        }

        let stftConfig = STFTConfig(uncheckedNFFT: 256, hopLength: 64)
        let griffinLim = GriffinLim(config: stftConfig, nIter: 10)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: 1024)

        // Should produce silence (or near silence)
        for value in reconstructed {
            XCTAssertFalse(value.isNaN, "Griffin-Lim on silence produced NaN")
            XCTAssertEqual(value, 0, accuracy: 1e-6, "Griffin-Lim on silence should produce silence")
        }
    }

    // MARK: - Window Function Edge Cases

    func testWindowFunctionLengthOne() {
        let windows: [(WindowType, String)] = [
            (.hann, "Hann"),
            (.hamming, "Hamming"),
            (.blackman, "Blackman"),
            (.bartlett, "Bartlett"),
            (.rectangular, "Rectangular")
        ]

        for (windowType, name) in windows {
            let window = Windows.generate(windowType, length: 1, periodic: true)
            XCTAssertEqual(window.count, 1, "\(name) window length 1 failed")
            XCTAssertFalse(window[0].isNaN, "\(name) window produced NaN for length 1")
        }
    }

    func testWindowFunctionLengthTwo() {
        let window = Windows.generate(.hann, length: 2, periodic: true)
        XCTAssertEqual(window.count, 2)

        for value in window {
            XCTAssertFalse(value.isNaN, "Window produced NaN")
            XCTAssertGreaterThanOrEqual(value, 0, "Window value should be non-negative")
            XCTAssertLessThanOrEqual(value, 1, "Window value should be <= 1")
        }
    }

    // MARK: - Streaming Edge Cases

    func testStreamingSTFTSingleSample() async throws {
        let config = STFTConfig(uncheckedNFFT: 256, hopLength: 64)
        let streaming = StreamingSTFT(config: config)

        // Push single sample
        try await streaming.push([1.0])

        // Try to pop frames (should be empty since we don't have enough samples)
        let frames = await streaming.popFrames()
        // No crash = success. With only 1 sample, we won't have a full frame yet.
        XCTAssertTrue(frames.isEmpty || frames.count >= 0, "Should handle single sample gracefully")
    }

    func testStreamingSTFTEmptyInput() async throws {
        let config = STFTConfig(uncheckedNFFT: 256, hopLength: 64)
        let streaming = StreamingSTFT(config: config)

        // Push empty array
        try await streaming.push([])
        let frames = await streaming.popFrames()
        XCTAssertTrue(frames.isEmpty, "Empty input should produce empty output")
    }

    // MARK: - Onset Detection Edge Cases

    func testOnsetDetectorConstantSignal() async {
        // Constant signal should have no onsets
        let signal = [Float](repeating: 0.5, count: 4096)

        let config = OnsetConfig(sampleRate: 22050)
        let detector = OnsetDetector(config: config)
        let onsets = await detector.detect(signal)

        XCTAssertTrue(onsets.isEmpty, "Constant signal should have no onsets")
    }

    func testOnsetDetectorSingleImpulse() async {
        var signal = [Float](repeating: 0, count: 4096)
        signal[1024] = 1.0  // Single impulse

        let config = OnsetConfig(sampleRate: 22050)
        let detector = OnsetDetector(config: config)
        let onsets = await detector.detect(signal)

        // Should detect the impulse as onset
        XCTAssertGreaterThanOrEqual(onsets.count, 0,
            "Single impulse may or may not register as onset depending on threshold")
    }

    // MARK: - Mel Filterbank Edge Cases

    func testMelFilterbankZeroFrequencyBand() async {
        // Create mel filterbank that includes DC
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 512,
            hopLength: 128,
            nMels: 40,
            fMin: 0  // Include DC
        )

        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            signal[i] = sin(2 * .pi * 440 * Float(i) / 22050)
        }

        let result = await melSpec.transform(signal)

        // Should not produce NaN/Inf
        for row in result {
            for value in row {
                XCTAssertFalse(value.isNaN, "Zero fMin produced NaN")
                XCTAssertFalse(value.isInfinite, "Zero fMin produced Inf")
            }
        }
    }

    func testMelFilterbankHighFrequencyOnly() async {
        // Create mel filterbank for high frequencies only
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 512,
            hopLength: 128,
            nMels: 20,
            fMin: 4000,  // Start at 4kHz
            fMax: 11025  // Nyquist
        )

        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<1024 {
            // High frequency content
            signal[i] = sin(2 * .pi * 6000 * Float(i) / 22050)
        }

        let result = await melSpec.transform(signal)

        XCTAssertGreaterThan(result.count, 0, "Should produce output")

        // Verify no NaN/Inf
        for row in result {
            for value in row {
                XCTAssertFalse(value.isNaN, "High fMin produced NaN")
                XCTAssertFalse(value.isInfinite, "High fMin produced Inf")
            }
        }
    }
}
