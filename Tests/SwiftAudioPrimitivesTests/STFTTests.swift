import XCTest
@testable import SwiftAudioPrimitives

final class STFTTests: XCTestCase {

    // MARK: - Basic Tests

    func testSTFTOutputShape() async {
        let signal = [Float](repeating: 0, count: 1024)
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)

        let result = await stft.transform(signal)

        // nFreqs = nFFT/2 + 1 = 129
        XCTAssertEqual(result.rows, 129)

        // nFrames depends on centering and signal length
        // With center=true, signal is padded by nFFT/2 on each side
        // So effective length = 1024 + 256 = 1280
        // nFrames = 1 + (1280 - 256) / 64 = 1 + 16 = 17
        XCTAssertGreaterThan(result.cols, 0)
    }

    func testSTFTWithSineWave() async {
        // Generate a simple sine wave
        let sampleRate: Float = 22050
        let frequency: Float = 440  // A4
        let duration: Float = 0.1   // 100ms
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let config = STFTConfig(nFFT: 1024, hopLength: 256)
        let stft = STFT(config: config)

        let result = await stft.transform(signal)

        // Check that we get a valid spectrogram
        XCTAssertEqual(result.rows, 513)  // nFFT/2 + 1
        XCTAssertGreaterThan(result.cols, 0)

        // Check that magnitude is reasonable
        let magnitude = result.magnitude
        let maxMag = magnitude.flatMap { $0 }.max() ?? 0
        XCTAssertGreaterThan(maxMag, 0)
    }

    func testMagnitudeSpectrogram() async {
        let signal = (0..<2048).map { sin(Float($0) * 0.1) }
        let stft = STFT(config: STFTConfig(nFFT: 512))

        let magnitude = await stft.magnitude(signal)

        // All magnitudes should be non-negative
        for row in magnitude {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0)
            }
        }
    }

    func testPowerSpectrogram() async {
        let signal = (0..<2048).map { sin(Float($0) * 0.1) }
        let stft = STFT(config: STFTConfig(nFFT: 512))

        let power = await stft.power(signal)

        // All power values should be non-negative
        for row in power {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0)
            }
        }
    }

    // MARK: - Configuration Tests

    func testDifferentFFTSizes() async {
        let signal = [Float](repeating: 0.5, count: 4096)

        for nFFT in [256, 512, 1024, 2048] {
            let config = STFTConfig(nFFT: nFFT)
            let stft = STFT(config: config)

            let result = await stft.transform(signal)

            XCTAssertEqual(result.rows, nFFT / 2 + 1, "Wrong nFreqs for nFFT=\(nFFT)")
        }
    }

    func testDifferentHopLengths() async {
        let signal = [Float](repeating: 0.5, count: 4096)
        let nFFT = 1024

        for hopLength in [128, 256, 512] {
            let config = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: false)
            let stft = STFT(config: config)

            let result = await stft.transform(signal)

            let expectedFrames = 1 + (signal.count - nFFT) / hopLength
            XCTAssertEqual(result.cols, expectedFrames, "Wrong nFrames for hopLength=\(hopLength)")
        }
    }

    func testCenteringOption() async {
        let signal = [Float](repeating: 0.5, count: 2048)
        let nFFT = 512
        let hopLength = 128

        let configCentered = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: true)
        let configUncentered = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: false)

        let stftCentered = STFT(config: configCentered)
        let stftUncentered = STFT(config: configUncentered)

        let resultCentered = await stftCentered.transform(signal)
        let resultUncentered = await stftUncentered.transform(signal)

        // Centered should have more frames due to padding
        XCTAssertGreaterThan(resultCentered.cols, resultUncentered.cols)
    }
}
