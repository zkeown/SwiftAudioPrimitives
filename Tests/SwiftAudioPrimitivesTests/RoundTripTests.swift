import XCTest
@testable import SwiftAudioPrimitives

final class RoundTripTests: XCTestCase {

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

        // STFT configuration with proper overlap for reconstruction
        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
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

        // Calculate max absolute error
        var maxError: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
        }

        // Allow small numerical error
        XCTAssertLessThan(maxError, 1e-4, "Round-trip error too large: \(maxError)")
    }

    func testRoundTripWithNoise() async {
        // Generate random noise
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = Float.random(in: -1...1)
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

        XCTAssertEqual(reconstructed.count, nSamples)

        // Calculate mean squared error
        var mse: Float = 0
        for i in 0..<min(signal.count, reconstructed.count) {
            let diff = signal[i] - reconstructed[i]
            mse += diff * diff
        }
        mse /= Float(nSamples)

        XCTAssertLessThan(mse, 1e-6, "Round-trip MSE too large: \(mse)")
    }

    func testRoundTripWithDifferentWindowTypes() async {
        let nSamples = 2048
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.05)
        }

        for windowType in [WindowType.hann, .hamming] {
            let config = STFTConfig(
                nFFT: 256,
                hopLength: 64,
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

            XCTAssertLessThan(
                maxError, 1e-3,
                "Round-trip error too large for \(windowType): \(maxError)"
            )
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
        for i in 0..<min(signal.count, reconstructed.count) {
            maxError = max(maxError, abs(signal[i] - reconstructed[i]))
        }

        XCTAssertLessThan(maxError, 1e-4)
    }
}
