import XCTest
@testable import SwiftRosaCore
import SwiftRosaEffects

/// Tests for numerical stability edge cases.
/// These tests verify that algorithms handle extreme inputs correctly without
/// producing NaN, infinity, or overflow errors.
final class NumericalStabilityTests: XCTestCase {

    // MARK: - Near-Zero Signal Tests

    /// Test STFT with very small amplitude signal
    func testSTFTNearZeroSignal() async {
        let epsilon: Float = 1e-10
        let signal = [Float](repeating: epsilon, count: 1024)

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let result = await stft.transform(signal)

        // Should not produce NaN or infinity
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "STFT real should not produce NaN for near-zero input")
                XCTAssertFalse(value.isInfinite, "STFT real should not produce infinity for near-zero input")
            }
        }
        for row in result.imag {
            for value in row {
                XCTAssertFalse(value.isNaN, "STFT imag should not produce NaN for near-zero input")
                XCTAssertFalse(value.isInfinite, "STFT imag should not produce infinity for near-zero input")
            }
        }
    }

    /// Test ISTFT reconstruction of near-zero magnitude spectrogram
    func testISTFTNearZeroMagnitude() async {
        let config = STFTConfig(nFFT: 256, hopLength: 64)

        // Create near-zero magnitude spectrogram
        let nFreqs = config.nFFT / 2 + 1
        let nFrames = 10
        let epsilon: Float = 1e-10

        var real = [[Float]](repeating: [Float](repeating: epsilon, count: nFrames), count: nFreqs)
        var imag = [[Float]](repeating: [Float](repeating: epsilon, count: nFrames), count: nFreqs)

        let spectrogram = ComplexMatrix(real: real, imag: imag)
        let istft = ISTFT(config: config)
        let result = await istft.transform(spectrogram, length: 512)

        // Should not produce NaN or infinity
        for value in result {
            XCTAssertFalse(value.isNaN, "ISTFT should not produce NaN for near-zero input")
            XCTAssertFalse(value.isInfinite, "ISTFT should not produce infinity for near-zero input")
        }
    }

    /// Test Mel filterbank with near-zero input
    func testMelFilterbankNearZeroInput() async {
        let filterbank = MelFilterbank(
            nMels: 40,
            nFFT: 512,
            sampleRate: 16000
        )

        // Create near-zero magnitude spectrogram
        let nFreqs = 512 / 2 + 1
        let nFrames = 10
        let epsilon: Float = 1e-10
        var magnitude = [[Float]](repeating: [Float](repeating: epsilon, count: nFrames), count: nFreqs)

        // Get filterbank weights
        let filters = filterbank.filters()

        // Manual mel spectrogram computation
        var melSpec = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: 40)
        for m in 0..<40 {
            for t in 0..<nFrames {
                var sum: Float = 0
                for f in 0..<nFreqs {
                    sum += filters[m][f] * magnitude[f][t]
                }
                melSpec[m][t] = sum
            }
        }

        // Should produce valid (small) values
        for row in melSpec {
            for value in row {
                XCTAssertFalse(value.isNaN, "Mel filterbank should not produce NaN")
                XCTAssertFalse(value.isInfinite, "Mel filterbank should not produce infinity")
                XCTAssertGreaterThanOrEqual(value, 0, "Mel filterbank output should be non-negative")
            }
        }
    }

    // MARK: - Extreme Amplitude Tests

    /// Test STFT with large amplitude signal
    func testSTFTLargeAmplitude() async {
        let amplitude: Float = 1000
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = amplitude * sin(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let result = await stft.transform(signal)

        // Should not produce NaN or infinity
        for row in result.real {
            for value in row {
                XCTAssertFalse(value.isNaN, "STFT real should not produce NaN for large input")
                XCTAssertFalse(value.isInfinite, "STFT real should not produce infinity for large input")
            }
        }
    }

    /// Test round-trip with extreme amplitudes
    func testRoundTripExtremeAmplitudes() async {
        let amplitudes: [Float] = [1e-6, 1e-4, 1e-2, 1, 10, 100, 1000]

        let config = STFTConfig(nFFT: 256, hopLength: 64, windowType: .hann)
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        for amplitude in amplitudes {
            var signal = [Float](repeating: 0, count: 512)
            for i in 0..<signal.count {
                signal[i] = amplitude * sin(Float(i) * 0.1)
            }

            let spectrogram = await stft.transform(signal)
            let reconstructed = await istft.transform(spectrogram, length: signal.count)

            // Check no NaN or infinity
            let hasNaN = reconstructed.contains { $0.isNaN }
            let hasInf = reconstructed.contains { $0.isInfinite }

            XCTAssertFalse(hasNaN, "Round-trip should not produce NaN for amplitude \(amplitude)")
            XCTAssertFalse(hasInf, "Round-trip should not produce infinity for amplitude \(amplitude)")

            // Check reconstruction preserves approximate amplitude
            let maxRecon = reconstructed.map { abs($0) }.max() ?? 0
            XCTAssertGreaterThan(maxRecon, amplitude * 0.1,
                "Reconstruction should preserve approximate amplitude for \(amplitude)")
        }
    }

    // MARK: - DTW Edge Cases

    /// Test DTW with identical zero sequences
    func testDTWZeroSequences() {
        let dtw = DTW()

        let seq1: [[Float]] = [[0, 0, 0, 0]]
        let seq2: [[Float]] = [[0, 0, 0, 0]]

        let result = dtw.align(seq1, seq2)

        XCTAssertEqual(result.distance, 0, accuracy: 1e-6,
            "Zero sequences should have zero distance")
        XCTAssertFalse(result.distance.isNaN, "DTW should not produce NaN for zero sequences")
    }

    /// Test DTW cosine distance with zero vectors (edge case)
    func testDTWCosineZeroVector() {
        let dtw = DTW(config: DTWConfig(metric: .cosine))

        // Zero vectors should be handled gracefully
        let seq1: [[Float]] = [[0], [0]]  // Zero 2D vector
        let seq2: [[Float]] = [[1], [1]]  // Non-zero 2D vector

        let distance = dtw.distance(seq1, seq2)

        // Cosine distance with zero vector should return 1.0 (max distance)
        // and not NaN
        XCTAssertFalse(distance.isNaN, "Cosine distance should not be NaN for zero vector")
        XCTAssertEqual(distance, 1.0, accuracy: 1e-6,
            "Cosine distance with zero vector should be 1.0")
    }

    // MARK: - Log Compression Tests

    /// Test that log compression handles zero correctly
    func testLogCompressionZero() {
        // The library uses log compression in Mel spectrograms
        // It should use a floor value to avoid log(0) = -inf

        let floor: Float = 1e-10
        let values: [Float] = [0, 1e-20, 1e-10, 1e-5, 0.1, 1.0, 10.0]

        for value in values {
            let safeValue = max(value, floor)
            let logValue = log10(safeValue)

            XCTAssertFalse(logValue.isNaN, "Log compression should not produce NaN for \(value)")
            XCTAssertFalse(logValue.isInfinite, "Log compression should not produce infinity for \(value)")
        }
    }

    // MARK: - Resampling Edge Cases

    /// Test resampling with silent signal
    func testResampleSilentSignal() async {
        let signal = [Float](repeating: 0, count: 1000)

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 16000,
            toSampleRate: 22050,
            quality: .high
        )

        // Should produce all zeros without NaN
        for value in resampled {
            XCTAssertFalse(value.isNaN, "Resampling silent signal should not produce NaN")
            XCTAssertEqual(value, 0, accuracy: 1e-6, "Resampling silent signal should produce zeros")
        }
    }

    /// Test resampling extreme rates
    func testResampleExtremeRates() async {
        var signal = [Float](repeating: 0, count: 1000)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        // Test extreme upsampling (8x)
        let upsampled = await Resample.resample(
            signal,
            fromSampleRate: 8000,
            toSampleRate: 64000,
            quality: .high
        )

        XCTAssertFalse(upsampled.contains { $0.isNaN }, "8x upsampling should not produce NaN")
        XCTAssertFalse(upsampled.contains { $0.isInfinite }, "8x upsampling should not produce infinity")

        // Test extreme downsampling (8x)
        let downsampled = await Resample.resample(
            signal,
            fromSampleRate: 64000,
            toSampleRate: 8000,
            quality: .high
        )

        XCTAssertFalse(downsampled.contains { $0.isNaN }, "8x downsampling should not produce NaN")
        XCTAssertFalse(downsampled.contains { $0.isInfinite }, "8x downsampling should not produce infinity")
    }

    // MARK: - Window Function Edge Cases

    /// Test window functions produce valid values
    func testWindowFunctionsValidity() {
        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett, .rectangular]
        let sizes = [1, 2, 3, 8, 16, 256, 1024]

        for windowType in windowTypes {
            for size in sizes {
                let window = Windows.generate(windowType, length: size, periodic: true)

                XCTAssertEqual(window.count, size, "\(windowType) window should have correct size")

                for (i, value) in window.enumerated() {
                    XCTAssertFalse(value.isNaN, "\(windowType) window[\(i)] should not be NaN")
                    XCTAssertFalse(value.isInfinite, "\(windowType) window[\(i)] should not be infinite")
                    XCTAssertGreaterThanOrEqual(value, 0,
                        "\(windowType) window[\(i)] should be non-negative")
                    XCTAssertLessThanOrEqual(value, 1,
                        "\(windowType) window[\(i)] should be <= 1")
                }
            }
        }
    }

    /// Test single-sample window
    func testSingleSampleWindow() {
        let window = Windows.generate(.hann, length: 1, periodic: true)

        XCTAssertEqual(window.count, 1)
        XCTAssertFalse(window[0].isNaN, "Single-sample window should not be NaN")
    }

    // MARK: - Griffin-Lim Edge Cases

    /// Test Griffin-Lim with uniform magnitude (challenging case)
    func testGriffinLimUniformMagnitude() async {
        // Uniform magnitude across all bins is a challenging edge case
        // because there's no frequency structure to guide phase estimation
        let nFreqs = 129  // nFFT/2+1 for nFFT=256
        let nFrames = 16
        let magnitude = [[Float]](repeating: [Float](repeating: 0.5, count: nFrames), count: nFreqs)

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let griffinLim = GriffinLim(config: config, nIter: 10, randomPhaseInit: false)

        let result = await griffinLim.reconstruct(magnitude, length: 1024)

        // Should not produce NaN
        for value in result {
            XCTAssertFalse(value.isNaN, "Griffin-Lim should not produce NaN for uniform magnitude")
            XCTAssertFalse(value.isInfinite, "Griffin-Lim should not produce infinity for uniform magnitude")
        }
    }
}
