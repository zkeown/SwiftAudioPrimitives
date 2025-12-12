import XCTest
@testable import SwiftRosaCore

/// Validates MelSpectrogram output against librosa reference values.
final class MelSpectrogramValidationTests: XCTestCase {

    // MARK: - Librosa Reference Values
    // Generated with:
    // signal = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050).astype(np.float32)
    // S = librosa.feature.melspectrogram(y=signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128, power=2.0, center=True, pad_mode='reflect')

    func testMelSpectrogramAgainstLibrosa_440HzSine() async throws {
        let sr: Float = 22050
        let nSamples = 22050  // 1 second

        // Generate 440 Hz sine wave
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * 440 * Float(i) / sr)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sr,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            fMin: 0,
            fMax: nil,
            htk: false,
            norm: .slaney,
            useGPU: false  // Force CPU for reproducibility
            // padMode defaults to .reflect to match librosa
        )

        let result = await melSpec.transform(signal)

        // Librosa reference values (from Python script)
        let librosaShape = (128, 44)  // (nMels, nFrames)
        let librosaSum: Float = 670774.6875
        let librosaMean: Float = 119.1006
        let librosaMaxBand = 16  // Band with max avg energy

        // Check shape
        XCTAssertEqual(result.count, librosaShape.0, "Should have \(librosaShape.0) mel bands")
        XCTAssertEqual(result[0].count, librosaShape.1, "Should have \(librosaShape.1) frames")

        // Compute sum and mean
        var totalSum: Float = 0
        for row in result {
            totalSum += row.reduce(0, +)
        }
        let totalMean = totalSum / Float(result.count * result[0].count)

        print("SwiftRosa shape: \(result.count) x \(result[0].count)")
        print("SwiftRosa sum: \(totalSum), librosa: \(librosaSum)")
        print("SwiftRosa mean: \(totalMean), librosa: \(librosaMean)")

        // Find band with max average energy
        var maxBand = 0
        var maxAvg: Float = 0
        for (i, row) in result.enumerated() {
            let avg = row.reduce(0, +) / Float(row.count)
            if avg > maxAvg {
                maxAvg = avg
                maxBand = i
            }
        }
        print("SwiftRosa max energy band: \(maxBand) (avg: \(maxAvg)), librosa: \(librosaMaxBand)")

        // Tolerance: 5% relative error (we're comparing two different implementations)
        let sumRelError = abs(totalSum - librosaSum) / librosaSum
        let meanRelError = abs(totalMean - librosaMean) / librosaMean

        print("Sum relative error: \(sumRelError * 100)%")
        print("Mean relative error: \(meanRelError * 100)%")

        XCTAssertLessThan(sumRelError, 0.05, "Sum should be within 5% of librosa")
        XCTAssertLessThan(meanRelError, 0.05, "Mean should be within 5% of librosa")

        // Max band should match (or be very close)
        XCTAssertEqual(maxBand, librosaMaxBand, accuracy: 2,
                       "Max energy band should match librosa (\(librosaMaxBand))")
    }

    func testMelSpectrogramFirstFrameValues() async throws {
        let sr: Float = 22050
        let nSamples = 22050

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * 440 * Float(i) / sr)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sr,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            useGPU: false
            // padMode defaults to .reflect to match librosa
        )

        let result = await melSpec.transform(signal)

        // Librosa first frame, first 5 bands:
        // [23.130714 24.408821 25.388672 25.72305  29.372797]
        let librosaFrame0First5: [Float] = [23.130714, 24.408821, 25.388672, 25.72305, 29.372797]

        print("SwiftRosa frame 0 first 5: \(result.prefix(5).map { String(format: "%.4f", $0[0]) })")
        print("Librosa frame 0 first 5: \(librosaFrame0First5.map { String(format: "%.4f", $0) })")

        // Compare first 5 mel bands of first frame
        for i in 0..<5 {
            let swiftVal = result[i][0]
            let librosaVal = librosaFrame0First5[i]
            let relError = abs(swiftVal - librosaVal) / max(librosaVal, 1e-10)
            print("Band \(i): SwiftRosa=\(swiftVal), librosa=\(librosaVal), relError=\(relError * 100)%")

            XCTAssertLessThan(relError, 0.10, // 10% tolerance
                              "Band \(i) frame 0 should match librosa within 10%")
        }
    }

    func testSTFTPowerSpectrogramDirectly() async throws {
        // Test the underlying STFT power spectrogram before mel filterbank
        let sr: Float = 22050
        let nSamples = 22050

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * 440 * Float(i) / sr)
        }

        let config = STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)

        let powerSpec = await stft.power(signal)

        // Librosa reference:
        // STFT shape: (1025, 44)
        // STFT power sum: 17303018.0
        let librosaShape = (1025, 44)
        let librosaPowerSum: Float = 17303018.0

        XCTAssertEqual(powerSpec.count, librosaShape.0, "Should have \(librosaShape.0) frequency bins")
        XCTAssertEqual(powerSpec[0].count, librosaShape.1, "Should have \(librosaShape.1) frames")

        var totalPowerSum: Float = 0
        for row in powerSpec {
            totalPowerSum += row.reduce(0, +)
        }

        print("SwiftRosa power spec shape: \(powerSpec.count) x \(powerSpec[0].count)")
        print("SwiftRosa power sum: \(totalPowerSum), librosa: \(librosaPowerSum)")

        let relError = abs(totalPowerSum - librosaPowerSum) / librosaPowerSum
        print("Power sum relative error: \(relError * 100)%")

        // Power spectrogram should be close
        XCTAssertLessThan(relError, 0.05, "Power sum should be within 5% of librosa")
    }

    func testPaddingModeImpact() async throws {
        // librosa uses pad_mode='reflect' by default, we use constant(0)
        // This test documents the difference
        let sr: Float = 22050
        let nSamples = 22050

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * 440 * Float(i) / sr)
        }

        // Our default: constant(0)
        let melConstant = MelSpectrogram(
            sampleRate: sr,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            useGPU: false
        )

        // With reflect padding (like librosa default)
        let melReflect = MelSpectrogram(
            sampleRate: sr,
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true,
            padMode: .reflect,
            nMels: 128,
            useGPU: false
        )

        let resultConstant = await melConstant.transform(signal)
        let resultReflect = await melReflect.transform(signal)

        var sumConstant: Float = 0
        var sumReflect: Float = 0
        for row in resultConstant { sumConstant += row.reduce(0, +) }
        for row in resultReflect { sumReflect += row.reduce(0, +) }

        print("Constant padding sum: \(sumConstant)")
        print("Reflect padding sum: \(sumReflect)")
        print("Difference: \(abs(sumConstant - sumReflect))")

        // Both should produce valid results
        XCTAssertGreaterThan(sumConstant, 0)
        XCTAssertGreaterThan(sumReflect, 0)
    }
}
