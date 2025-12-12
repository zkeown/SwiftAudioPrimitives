import XCTest
@testable import SwiftRosaCore

final class RoundTripTests: XCTestCase {

    // MARK: - Accuracy Thresholds Documentation
    //
    // For COLA-compliant configurations (Hann with 75% overlap):
    //   - Max error: 1e-5
    //   - MSE: 1e-10
    //   - Normalized RMSE: 1e-5
    //
    // For COLA-compliant configurations (Hann with 50% overlap):
    //   - Max error: 1e-4
    //   - MSE: 1e-8
    //
    // Accuracy threshold constants
    private let colaMaxError: Float = 1e-4
    private let colaMSE: Float = 1e-8
    private let nonColaMaxError: Float = 1e-3

    // MARK: - STFT -> ISTFT Round Trip

    func testRoundTripWithSineWave() async {
        // Generate a sine wave
        let sampleRate: Float = 22050
        let frequency: Float = 440
        let nSamples = 8192

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        // STFT configuration with proper overlap for reconstruction (75% overlap with Hann)
        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,  // 75% overlap = nFFT/4
            windowType: .hann,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Forward transform
        let spectrogram = await stft.transform(signal)

        // Inverse transform
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        // Check that reconstruction matches original
        XCTAssertEqual(reconstructed.count, nSamples)

        // Calculate max absolute error and RMSE
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(nSamples))

        // COLA-compliant config should achieve high accuracy
        XCTAssertLessThan(maxError, colaMaxError,
            "Round-trip max error too large: \(maxError)")
        XCTAssertLessThan(rmse, colaMaxError,
            "Round-trip RMSE too large: \(rmse)")
    }

    func testRoundTripWithNoise() async {
        // Generate random noise (seeded for reproducibility)
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        srand48(42)  // Seed for reproducibility
        for i in 0..<nSamples {
            signal[i] = Float(drand48() * 2 - 1)
        }

        let config = STFTConfig(
            nFFT: 512,
            hopLength: 128,  // 75% overlap
            windowType: .hann,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        XCTAssertEqual(reconstructed.count, nSamples)

        // Calculate MSE and max error
        var mse: Float = 0
        var maxError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let diff = signal[i] - reconstructed[i]
            mse += diff * diff
            maxError = max(maxError, abs(diff))
        }
        mse /= Float(nSamples)

        XCTAssertLessThan(mse, colaMSE,
            "Round-trip MSE too large: \(mse)")
        XCTAssertLessThan(maxError, colaMaxError,
            "Round-trip max error too large: \(maxError)")
    }

    func testRoundTripWithDifferentWindowTypes() async {
        let nSamples = 2048
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.05)
        }

        // Test both COLA-compliant windows
        // Hann with 75% overlap and Hamming with 75% overlap
        let windowConfigs: [(WindowType, Float, String)] = [
            (.hann, colaMaxError, "Hann (COLA)"),
            (.hamming, nonColaMaxError, "Hamming (non-COLA)")  // Hamming is not perfectly COLA
        ]

        for (windowType, expectedMaxError, description) in windowConfigs {
            let config = STFTConfig(
                nFFT: 256,
                hopLength: 64,  // 75% overlap
                windowType: windowType,
                center: true
            )

            let stft = STFT(config: config)
            let istft = ISTFT(config: config)

            let spectrogram = await stft.transform(signal)
            let reconstructed = await istft.transform(spectrogram, length: nSamples)

            var maxError: Float = 0
            for i in 0..<min(signal.count, reconstructed.count) {
                maxError = max(maxError, abs(signal[i] - reconstructed[i]))
            }

            XCTAssertLessThan(maxError, expectedMaxError,
                "\(description) round-trip error too large: \(maxError), expected < \(expectedMaxError)")
        }
    }

    // MARK: - COLA Property Tests

    func testCOLAPropertyWith50PercentOverlap() async {
        // Hann window with 50% overlap satisfies COLA
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.1) + 0.3 * cos(Float(i) * 0.3)
        }

        let nFFT = 512
        let hopLength = nFFT / 2  // 50% overlap
        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        // With COLA, reconstruction should be very accurate
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(nSamples))

        // 50% overlap Hann should still achieve good accuracy
        XCTAssertLessThan(maxError, 1e-4,
            "50% overlap COLA max error: \(maxError)")
        XCTAssertLessThan(rmse, 1e-5,
            "50% overlap COLA RMSE: \(rmse)")
    }

    func testCOLAPropertyWith75PercentOverlap() async {
        // Hann window with 75% overlap is the best COLA configuration
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.1) + 0.3 * cos(Float(i) * 0.3)
        }

        let nFFT = 512
        let hopLength = nFFT / 4  // 75% overlap
        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(nSamples))

        // 75% overlap should achieve best accuracy
        XCTAssertLessThan(maxError, 1e-4,
            "75% overlap COLA max error: \(maxError)")
        XCTAssertLessThan(rmse, 1e-5,
            "75% overlap COLA RMSE: \(rmse)")
    }

    // MARK: - Normalized RMSE Tests

    func testNormalizedRMSE() async {
        // Test with different amplitude signals to verify scale-invariant accuracy
        let amplitudes: [Float] = [0.1, 0.5, 1.0, 2.0]
        let nSamples = 4096

        for amplitude in amplitudes {
            var signal = [Float](repeating: 0, count: nSamples)
            for i in 0..<nSamples {
                signal[i] = amplitude * sin(Float(i) * 0.1)
            }

            let config = STFTConfig(
                nFFT: 512,
                hopLength: 128,
                windowType: .hann,
                center: true
            )

            let stft = STFT(config: config)
            let istft = ISTFT(config: config)

            let spectrogram = await stft.transform(signal)
            let reconstructed = await istft.transform(spectrogram, length: nSamples)

            // Calculate normalized RMSE (relative to signal RMS)
            var sumSquaredError: Float = 0
            var sumSquaredSignal: Float = 0
            for i in 0..<min(signal.count, reconstructed.count) {
                let error = signal[i] - reconstructed[i]
                sumSquaredError += error * error
                sumSquaredSignal += signal[i] * signal[i]
            }

            let rmse = sqrt(sumSquaredError / Float(nSamples))
            let signalRMS = sqrt(sumSquaredSignal / Float(nSamples))
            let normalizedRMSE = rmse / max(signalRMS, 1e-10)

            // Normalized RMSE should be consistent regardless of amplitude
            XCTAssertLessThan(normalizedRMSE, 1e-4,
                "Normalized RMSE for amplitude \(amplitude): \(normalizedRMSE)")
        }
    }

    // MARK: - Edge Cases

    func testRoundTripShortSignal() async {
        // Signal shorter than FFT size
        let signal: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]

        let config = STFTConfig(
            nFFT: 16,
            hopLength: 4,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        XCTAssertEqual(reconstructed.count, signal.count)

        // Even short signals should reconstruct reasonably
        var maxError: Float = 0
        for i in 0..<signal.count {
            maxError = max(maxError, abs(signal[i] - reconstructed[i]))
        }
        XCTAssertLessThan(maxError, 0.1,
            "Short signal round-trip max error: \(maxError)")
    }

    func testRoundTripConstantSignal() async {
        let signal = [Float](repeating: 0.5, count: 1024)

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        // For constant signal, reconstruction should be very close
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(signal.count))

        XCTAssertLessThan(maxError, colaMaxError,
            "Constant signal max error: \(maxError)")
        XCTAssertLessThan(rmse, colaMaxError,
            "Constant signal RMSE: \(rmse)")
    }

    // MARK: - Non-Centered STFT Tests

    func testRoundTripUncentered() async {
        // Test uncentered STFT/ISTFT (no padding)
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.1)
        }

        let nFFT = 512
        let hopLength = 128
        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false  // Uncentered
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        // Check accuracy in the valid region (excluding edges)
        // Uncentered STFT doesn't pad, so edge regions may have artifacts
        let validStart = nFFT / 2
        let validEnd = nSamples - nFFT / 2

        var maxError: Float = 0
        for i in validStart..<min(validEnd, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
        }

        XCTAssertLessThan(maxError, colaMaxError,
            "Uncentered round-trip max error in valid region: \(maxError)")
    }

    // MARK: - Complex Signal Tests

    func testRoundTripComplexSignal() async {
        // Test with a more complex signal (multiple frequencies + noise)
        let sampleRate: Float = 22050
        let nSamples = 8192
        srand48(123)  // Seed for reproducibility

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / sampleRate
            // Multiple frequency components
            signal[i] = 0.3 * sin(2 * Float.pi * 220 * t)   // 220 Hz
            signal[i] += 0.2 * sin(2 * Float.pi * 440 * t)  // 440 Hz
            signal[i] += 0.1 * sin(2 * Float.pi * 880 * t)  // 880 Hz
            // Add small noise
            signal[i] += 0.05 * Float(drand48() * 2 - 1)
        }

        let config = STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: nSamples)

        // Calculate metrics
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(nSamples))

        XCTAssertLessThan(maxError, colaMaxError,
            "Complex signal max error: \(maxError)")
        XCTAssertLessThan(rmse, 1e-5,
            "Complex signal RMSE: \(rmse)")
    }
}
