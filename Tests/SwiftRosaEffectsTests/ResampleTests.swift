import XCTest
@testable import SwiftRosaEffects
import SwiftRosaCore

final class ResampleTests: XCTestCase {

    // MARK: - Basic Resampling Tests

    func testUpsample2x() async {
        // Simple 2x upsampling
        let signal: [Float] = [0, 1, 0, -1, 0, 1, 0, -1]

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 22050,
            toSampleRate: 44100,
            quality: .medium
        )

        // Output should be approximately 2x length
        XCTAssertEqual(resampled.count, signal.count * 2, accuracy: 2)
    }

    func testDownsample2x() async {
        // Generate 2x oversampled signal
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            // Low frequency content that survives decimation
            signal[i] = sin(Float(i) * 0.01)
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 44100,
            toSampleRate: 22050,
            quality: .medium
        )

        // Output should be approximately half length
        XCTAssertEqual(resampled.count, signal.count / 2, accuracy: 2)
    }

    func testNoResampleNeeded() async {
        let signal: [Float] = [1, 2, 3, 4, 5]

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 44100,
            toSampleRate: 44100,
            quality: .medium
        )

        // Should return same signal when rates match
        XCTAssertEqual(resampled.count, signal.count)
        for (i, val) in resampled.enumerated() {
            XCTAssertEqual(val, signal[i], accuracy: 0.0001)
        }
    }

    func testEmptySignal() async {
        let signal: [Float] = []

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 22050,
            toSampleRate: 44100,
            quality: .low
        )

        XCTAssertTrue(resampled.isEmpty)
    }

    // MARK: - Quality Level Tests

    func testDifferentQualities() async {
        // Generate test signal with known frequency content
        var signal = [Float](repeating: 0, count: 512)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        let qualities: [ResampleQuality] = [.low, .medium, .high, .veryHigh]

        for quality in qualities {
            let resampled = await Resample.resample(
                signal,
                fromSampleRate: 44100,
                toSampleRate: 22050,
                quality: quality
            )

            // All should produce output
            XCTAssertGreaterThan(resampled.count, 0,
                "Quality \(quality) should produce output")

            // Output length should be consistent
            XCTAssertEqual(resampled.count, signal.count / 2, accuracy: 2,
                "Quality \(quality) should produce correct length")
        }
    }

    func testQualityFilterLengths() {
        XCTAssertEqual(ResampleQuality.low.filterLength, 8)
        XCTAssertEqual(ResampleQuality.medium.filterLength, 16)
        XCTAssertEqual(ResampleQuality.high.filterLength, 32)
        XCTAssertEqual(ResampleQuality.veryHigh.filterLength, 64)
    }

    func testQualityLevelsProgressivelyImprove() async {
        // Higher quality settings should produce better results
        // Test with a bandlimited signal that requires careful interpolation
        let sampleRate = 44100
        let frequency: Float = 1000  // Well within Nyquist for both rates
        let nSamples = 4096

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / Float(sampleRate))
        }

        // Downsample then upsample back - measure reconstruction error
        func roundTripError(quality: ResampleQuality) async -> Float {
            let downsampled = await Resample.resample(
                signal,
                fromSampleRate: sampleRate,
                toSampleRate: 22050,
                quality: quality
            )
            let reconstructed = await Resample.resample(
                downsampled,
                fromSampleRate: 22050,
                toSampleRate: sampleRate,
                quality: quality
            )

            // Calculate MSE in central region (avoid edge effects)
            let margin = nSamples / 8
            var sumSquaredError: Float = 0
            let compareLen = min(signal.count, reconstructed.count) - 2 * margin
            for i in margin..<(margin + compareLen) {
                let diff = signal[i] - reconstructed[i]
                sumSquaredError += diff * diff
            }
            return sumSquaredError / Float(compareLen)
        }

        let mseLow = await roundTripError(quality: .low)
        let mseMedium = await roundTripError(quality: .medium)
        let mseHigh = await roundTripError(quality: .high)
        let mseVeryHigh = await roundTripError(quality: .veryHigh)

        // Higher quality should produce lower error (or at least not worse)
        XCTAssertLessThanOrEqual(mseMedium, mseLow * 1.1,
            "Medium quality (\(mseMedium)) should not be worse than low (\(mseLow))")
        XCTAssertLessThanOrEqual(mseHigh, mseMedium * 1.1,
            "High quality (\(mseHigh)) should not be worse than medium (\(mseMedium))")
        XCTAssertLessThanOrEqual(mseVeryHigh, mseHigh * 1.1,
            "Very high quality (\(mseVeryHigh)) should not be worse than high (\(mseHigh))")

        // Very high should be noticeably better than low
        XCTAssertLessThan(mseVeryHigh, mseLow,
            "Very high quality (\(mseVeryHigh)) should be better than low (\(mseLow))")
    }

    // MARK: - Frequency Preservation Tests

    func testLowFrequencyPreservation() async {
        // Low frequency signal should be preserved through resampling
        // Test by checking that the signal still looks sinusoidal with correct frequency
        let sampleRate = 44100
        let frequency: Float = 100  // Well below Nyquist for both rates
        let nSamples = 4096

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / Float(sampleRate))
        }

        // Downsample to 22050
        let downsampled = await Resample.resample(
            signal,
            fromSampleRate: sampleRate,
            toSampleRate: 22050,
            quality: .high
        )

        // Upsample back to 44100
        let resampled = await Resample.resample(
            downsampled,
            fromSampleRate: 22050,
            toSampleRate: sampleRate,
            quality: .high
        )

        // Verify output has reasonable length
        let expectedLength = nSamples
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 10,
            "Resampled length should be close to original")

        // Verify RMS (energy) is preserved with tighter tolerance
        // Use central portion to avoid edge filter effects
        let margin = nSamples / 8
        var originalRMS: Float = 0
        var resampledRMS: Float = 0

        let compareStart = margin
        let compareEnd = min(signal.count, resampled.count) - margin
        let compareLen = compareEnd - compareStart

        for i in compareStart..<compareEnd {
            originalRMS += signal[i] * signal[i]
            resampledRMS += resampled[i] * resampled[i]
        }
        originalRMS = sqrt(originalRMS / Float(compareLen))
        resampledRMS = sqrt(resampledRMS / Float(compareLen))

        // RMS should be similar (within 10% for central region)
        let rmsRatio = resampledRMS / max(originalRMS, 0.001)
        XCTAssertGreaterThan(rmsRatio, 0.9, "Resampled RMS too low: \(resampledRMS) vs \(originalRMS), ratio: \(rmsRatio)")
        XCTAssertLessThan(rmsRatio, 1.1, "Resampled RMS too high: \(resampledRMS) vs \(originalRMS), ratio: \(rmsRatio)")
    }

    func testAntiAliasingFilter() async {
        // Test that frequencies near Nyquist are attenuated when downsampling
        let originalRate = 44100
        let targetRate = 22050
        let nyquistNew = Float(targetRate) / 2  // 11025 Hz

        // Create signal with frequency at 80% of new Nyquist (should pass)
        let freqPass: Float = 0.8 * nyquistNew  // ~8820 Hz
        let nSamples = 8192

        var signalPass = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signalPass[i] = sin(2 * Float.pi * freqPass * Float(i) / Float(originalRate))
        }

        let resampledPass = await Resample.resample(
            signalPass,
            fromSampleRate: originalRate,
            toSampleRate: targetRate,
            quality: .high
        )

        // Calculate RMS of resampled signal
        let rmsPass = sqrt(resampledPass.map { $0 * $0 }.reduce(0, +) / Float(resampledPass.count))

        // Signal at 80% Nyquist should have reasonable energy preserved
        XCTAssertGreaterThan(rmsPass, 0.3,
            "Signal at 80% Nyquist should be largely preserved, got RMS: \(rmsPass)")

        // Create signal with frequency well above new Nyquist (should be attenuated)
        // Use 2x Nyquist to be clearly in the stopband beyond the filter's transition width
        let freqStop: Float = 2.0 * nyquistNew  // ~22050 Hz (2x above new Nyquist)

        var signalStop = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signalStop[i] = sin(2 * Float.pi * freqStop * Float(i) / Float(originalRate))
        }

        let resampledStop = await Resample.resample(
            signalStop,
            fromSampleRate: originalRate,
            toSampleRate: targetRate,
            quality: .high
        )

        // Calculate RMS of resampled signal
        let rmsStop = sqrt(resampledStop.map { $0 * $0 }.reduce(0, +) / Float(resampledStop.count))

        // Signal at 2x Nyquist should be significantly attenuated (stopband > -20dB)
        XCTAssertLessThan(rmsStop, 0.3,
            "Signal at 2x Nyquist should be attenuated to < 0.3 RMS, got: \(rmsStop)")

        // The passband signal should have more energy than stopband
        XCTAssertGreaterThan(rmsPass, rmsStop,
            "Passband (\(rmsPass)) should have more energy than stopband (\(rmsStop))")
    }

    // MARK: - Common Rate Conversion Tests

    func test44100to48000() async {
        var signal = [Float](repeating: 0, count: 4410)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 44100,
            toSampleRate: 48000,
            quality: .high
        )

        // Expected length: 4410 * 48000 / 44100 ≈ 4800
        let expectedLength = Int(Float(signal.count) * 48000.0 / 44100.0)
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 5)
    }

    func test48000to44100() async {
        var signal = [Float](repeating: 0, count: 4800)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 48000,
            toSampleRate: 44100,
            quality: .high
        )

        // Expected length: 4800 * 44100 / 48000 ≈ 4410
        let expectedLength = Int(Float(signal.count) * 44100.0 / 48000.0)
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 5)
    }

    func test44100to16000() async {
        // Common speech sample rate conversion
        var signal = [Float](repeating: 0, count: 4410)
        for i in 0..<signal.count {
            signal[i] = Float.random(in: -1...1)
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 44100,
            toSampleRate: 16000,
            quality: .high
        )

        let expectedLength = Int(Float(signal.count) * 16000.0 / 44100.0)
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 5)
    }

    // MARK: - Round-Trip Tests

    func testRoundTrip2x() async {
        // Generate bandlimited signal
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            // Low frequency only
            signal[i] = sin(Float(i) * 0.02) + 0.5 * cos(Float(i) * 0.04)
        }

        // Upsample then downsample
        let upsampled = await Resample.resample(
            signal,
            fromSampleRate: 22050,
            toSampleRate: 44100,
            quality: .high
        )

        let roundTrip = await Resample.resample(
            upsampled,
            fromSampleRate: 44100,
            toSampleRate: 22050,
            quality: .high
        )

        // Should match original (with some error at edges)
        XCTAssertEqual(roundTrip.count, signal.count, accuracy: 2)

        // Compare central portion - use tighter margin (10% on each side)
        let margin = signal.count / 10
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        var compareCount = 0
        for i in margin..<(signal.count - margin) {
            if i < roundTrip.count {
                let error = abs(signal[i] - roundTrip[i])
                maxError = max(maxError, error)
                sumSquaredError += error * error
                compareCount += 1
            }
        }
        let rmse = sqrt(sumSquaredError / Float(compareCount))

        // Tighter tolerances for round-trip with high quality
        XCTAssertLessThan(maxError, 0.02,
            "Round-trip max error too large: \(maxError)")
        XCTAssertLessThan(rmse, 0.01,
            "Round-trip RMSE too large: \(rmse)")
    }

    // MARK: - Edge Cases

    func testVeryShortSignal() async {
        let signal: [Float] = [1.0, 0.5]

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 22050,
            toSampleRate: 44100,
            quality: .low
        )

        // Should handle without crashing
        XCTAssertGreaterThan(resampled.count, 0)
    }

    func testSingleSample() async {
        let signal: [Float] = [1.0]

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 22050,
            toSampleRate: 44100,
            quality: .low
        )

        XCTAssertGreaterThanOrEqual(resampled.count, 1)
    }

    // MARK: - DC Preservation

    func testDCPreservation() async {
        // Constant signal should remain constant after resampling
        let dcLevel: Float = 0.5
        let signal = [Float](repeating: dcLevel, count: 1024)

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 44100,
            toSampleRate: 22050,
            quality: .high
        )

        // Check that DC level is preserved in central portion (use tighter margin)
        let margin = resampled.count / 10
        var maxDCError: Float = 0
        for i in margin..<(resampled.count - margin) {
            let error = abs(resampled[i] - dcLevel)
            maxDCError = max(maxDCError, error)
            XCTAssertEqual(resampled[i], dcLevel, accuracy: 0.001,
                "DC level not preserved at index \(i): got \(resampled[i]), expected \(dcLevel)")
        }

        // Overall DC error should be very small
        XCTAssertLessThan(maxDCError, 0.001,
            "DC preservation max error: \(maxDCError)")
    }

    // MARK: - Extreme Ratio Tests

    func testExtremeUpsampleRatio() async {
        // Test 8x upsampling (e.g., 6000 -> 48000)
        var signal = [Float](repeating: 0, count: 256)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)  // Low frequency
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 6000,
            toSampleRate: 48000,
            quality: .high
        )

        // Output should be 8x length
        let expectedLength = signal.count * 8
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 10,
            "8x upsample should produce ~\(expectedLength) samples, got \(resampled.count)")

        // Energy should be preserved (RMS should match)
        let originalRMS = sqrt(signal.map { $0 * $0 }.reduce(0, +) / Float(signal.count))
        let resampledRMS = sqrt(resampled.map { $0 * $0 }.reduce(0, +) / Float(resampled.count))
        let rmsRatio = resampledRMS / max(originalRMS, 0.001)

        XCTAssertGreaterThan(rmsRatio, 0.9, "Extreme upsample RMS ratio too low: \(rmsRatio)")
        XCTAssertLessThan(rmsRatio, 1.1, "Extreme upsample RMS ratio too high: \(rmsRatio)")
    }

    func testExtremeDownsampleRatio() async {
        // Test 8x downsampling (e.g., 48000 -> 6000)
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            // Use frequency well below target Nyquist (3000 Hz)
            signal[i] = sin(Float(i) * 0.01)
        }

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 48000,
            toSampleRate: 6000,
            quality: .high
        )

        // Output should be 1/8 length
        let expectedLength = signal.count / 8
        XCTAssertEqual(resampled.count, expectedLength, accuracy: 5,
            "8x downsample should produce ~\(expectedLength) samples, got \(resampled.count)")
    }
}
