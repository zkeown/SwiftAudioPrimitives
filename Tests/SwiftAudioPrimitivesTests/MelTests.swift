import XCTest
@testable import SwiftAudioPrimitives

final class MelTests: XCTestCase {

    // MARK: - Mel Scale Reference Values
    // Slaney formula: mel = 2595 * log10(1 + f/700)
    // HTK formula: mel = 1127 * ln(1 + f/700)

    /// Convert Hz to mel (Slaney formula)
    private func hzToMelSlaney(_ hz: Float) -> Float {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }

    /// Convert mel to Hz (Slaney formula)
    private func melToHzSlaney(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    // MARK: - Mel Filterbank Tests

    func testMelFilterbankShape() {
        let filterbank = MelFilterbank(nMels: 128, nFFT: 2048, sampleRate: 22050)
        let filters = filterbank.filters()

        XCTAssertEqual(filters.count, 128)
        XCTAssertEqual(filters[0].count, 1025)  // nFFT/2 + 1
    }

    func testMelFilterbankNormalization() {
        let filterbank = MelFilterbank(nMels: 40, nFFT: 512, sampleRate: 16000, norm: .slaney)
        let filters = filterbank.filters()

        // With Slaney normalization, each triangular filter has area = 2 / (f_right - f_left)
        // This means the filter sum depends on the bandwidth
        for (i, filter) in filters.enumerated() {
            let sum = filter.reduce(0, +)
            // Sum should be positive
            XCTAssertGreaterThan(sum, 0, "Filter \(i) should have positive sum")

            // Count non-zero elements to verify triangular shape
            let nonZeroCount = filter.filter { $0 > 0 }.count
            XCTAssertGreaterThan(nonZeroCount, 0, "Filter \(i) should have non-zero elements")
        }
    }

    func testMelFilterbankTriangularShape() {
        // Verify filters have proper triangular shape
        let filterbank = MelFilterbank(nMels: 10, nFFT: 256, sampleRate: 8000)
        let filters = filterbank.filters()

        for (filterIdx, filter) in filters.enumerated() {
            // Find the peak
            var peakIdx = 0
            var peakVal: Float = 0
            for (i, val) in filter.enumerated() {
                if val > peakVal {
                    peakVal = val
                    peakIdx = i
                }
            }

            if peakVal > 0 {
                // Values should increase up to peak, then decrease
                // Check left side (non-decreasing to peak)
                var prevVal: Float = 0
                for i in 0..<peakIdx {
                    if filter[i] > 0 {
                        XCTAssertGreaterThanOrEqual(filter[i], prevVal - 1e-6,
                            "Filter \(filterIdx) should be non-decreasing before peak")
                        prevVal = filter[i]
                    }
                }

                // Check right side (non-increasing from peak)
                prevVal = peakVal
                for i in (peakIdx + 1)..<filter.count {
                    if filter[i] > 0 {
                        XCTAssertLessThanOrEqual(filter[i], prevVal + 1e-6,
                            "Filter \(filterIdx) should be non-increasing after peak")
                        prevVal = filter[i]
                    }
                }
            }
        }
    }

    func testMelFilterbankFrequencyRange() {
        let fMin: Float = 100
        let fMax: Float = 8000
        let sampleRate: Float = 22050
        let nFFT = 1024

        let filterbank = MelFilterbank(
            nMels: 64,
            nFFT: nFFT,
            sampleRate: sampleRate,
            fMin: fMin,
            fMax: fMax
        )
        let filters = filterbank.filters()

        XCTAssertEqual(filters.count, 64)

        // First filter should start around fMin
        let firstFilter = filters[0]
        let firstNonZero = firstFilter.firstIndex { $0 > 0 } ?? 0
        let binFreq = sampleRate / Float(nFFT) * Float(firstNonZero)

        // Should start near fMin (allow some margin due to discrete bins)
        let binResolution = sampleRate / Float(nFFT)
        XCTAssertGreaterThanOrEqual(binFreq, fMin - binResolution,
            "First filter should start near fMin (\(fMin) Hz), got \(binFreq) Hz")

        // Last filter should end around fMax
        let lastFilter = filters[filters.count - 1]
        let lastNonZero = lastFilter.lastIndex { $0 > 0 } ?? 0
        let lastBinFreq = sampleRate / Float(nFFT) * Float(lastNonZero)

        XCTAssertLessThanOrEqual(lastBinFreq, fMax + binResolution,
            "Last filter should end near fMax (\(fMax) Hz), got \(lastBinFreq) Hz")
    }

    func testHTKvsSlaneyMel() {
        // HTK and Slaney formulas should give different results
        let htk = MelFilterbank(nMels: 40, nFFT: 512, sampleRate: 16000, htk: true)
        let slaney = MelFilterbank(nMels: 40, nFFT: 512, sampleRate: 16000, htk: false)

        let htkFilters = htk.filters()
        let slaneyFilters = slaney.filters()

        // Filters should be different
        var different = false
        for i in 0..<40 {
            for j in 0..<htkFilters[i].count {
                if abs(htkFilters[i][j] - slaneyFilters[i][j]) > 1e-6 {
                    different = true
                    break
                }
            }
            if different { break }
        }
        XCTAssertTrue(different, "HTK and Slaney filters should differ")
    }

    func testMelScaleAccuracy() {
        // Test that mel scale conversion matches expected values
        // Reference: 1000 Hz ≈ 1000 mels (approximately, by design of Slaney scale)
        let testFreqs: [Float] = [0, 100, 500, 1000, 2000, 4000, 8000]

        for freq in testFreqs {
            let mel = hzToMelSlaney(freq)
            let recoveredHz = melToHzSlaney(mel)

            XCTAssertEqual(recoveredHz, freq, accuracy: 0.01,
                "Mel scale round-trip failed for \(freq) Hz")
        }

        // Specific reference: 1000 Hz should be close to 1000 mels
        let mel1000 = hzToMelSlaney(1000)
        XCTAssertEqual(mel1000, 1000, accuracy: 50,
            "1000 Hz should be approximately 1000 mels, got \(mel1000)")
    }

    // MARK: - Mel Spectrogram Tests

    func testMelSpectrogramShape() async {
        let nSamples = 22050  // 1 second at 22050 Hz
        let signal = [Float](repeating: 0.1, count: nSamples)

        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )

        let result = await melSpec.transform(signal)

        XCTAssertEqual(result.count, 128)  // nMels
        XCTAssertGreaterThan(result[0].count, 0)  // nFrames
    }

    func testMelSpectrogramWithSineWave() async {
        // Generate a sine wave at 1000 Hz
        let sampleRate: Float = 22050
        let frequency: Float = 1000
        let nSamples = 22050
        let nMels = 80

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: nMels
        )

        let result = await melSpec.transform(signal)

        // Find the mel bin with maximum energy
        var maxBin = 0
        var maxEnergy: Float = 0

        for melBin in 0..<result.count {
            let avgEnergy = result[melBin].reduce(0, +) / Float(result[melBin].count)
            if avgEnergy > maxEnergy {
                maxEnergy = avgEnergy
                maxBin = melBin
            }
        }

        // Calculate expected mel bin for 1000 Hz
        // mel(1000) ≈ 1000 mels
        // For nMels=80, fMin=0, fMax=sr/2=11025
        // mel(11025) ≈ 2840 mels
        // Expected bin ≈ 1000 / 2840 * 80 ≈ 28
        let melOfFreq = hzToMelSlaney(frequency)
        let melOfMax = hzToMelSlaney(sampleRate / 2)
        let expectedBin = Int(melOfFreq / melOfMax * Float(nMels))

        // Allow ±5 bins tolerance due to filter overlap
        XCTAssertEqual(maxBin, expectedBin, accuracy: 5,
            "Peak at mel bin \(maxBin), expected near \(expectedBin) for \(frequency) Hz")
    }

    func testMelSpectrogramNonNegative() async {
        srand48(42)
        let signal = (0..<4096).map { _ in Float(drand48() * 2 - 1) }

        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 1024, hopLength: 256, nMels: 64)
        let result = await melSpec.transform(signal)

        for (row, values) in result.enumerated() {
            for (col, value) in values.enumerated() {
                XCTAssertGreaterThanOrEqual(value, 0,
                    "Mel spectrogram should be non-negative at [\(row)][\(col)]")
            }
        }
    }

    // MARK: - dB Conversion Tests with Relative Tolerance

    func testPowerToDb() {
        let power: [[Float]] = [[1.0, 10.0, 100.0, 1000.0]]
        let db = Convert.powerToDb(power, ref: 1.0, topDb: nil)

        // Use consistent absolute tolerance for dB values
        let tolerance: Float = 1e-4
        XCTAssertEqual(db[0][0], 0, accuracy: tolerance)      // 10*log10(1) = 0
        XCTAssertEqual(db[0][1], 10, accuracy: tolerance)     // 10*log10(10) = 10
        XCTAssertEqual(db[0][2], 20, accuracy: tolerance)     // 10*log10(100) = 20
        XCTAssertEqual(db[0][3], 30, accuracy: tolerance)     // 10*log10(1000) = 30
    }

    func testAmplitudeToDb() {
        let amplitude: [[Float]] = [[1.0, 10.0, 100.0]]
        let db = Convert.amplitudeToDb(amplitude, ref: 1.0, topDb: nil)

        let tolerance: Float = 1e-4
        XCTAssertEqual(db[0][0], 0, accuracy: tolerance)      // 20*log10(1) = 0
        XCTAssertEqual(db[0][1], 20, accuracy: tolerance)     // 20*log10(10) = 20
        XCTAssertEqual(db[0][2], 40, accuracy: tolerance)     // 20*log10(100) = 40
    }

    func testDbToPower() {
        let db: [[Float]] = [[0, 10, 20, 30]]
        let power = Convert.dbToPower(db, ref: 1.0)

        // Use RELATIVE tolerance for power values
        // For exponential conversion, relative error is more appropriate
        let expected: [Float] = [1, 10, 100, 1000]
        for i in 0..<expected.count {
            let relativeError = abs(power[0][i] - expected[i]) / expected[i]
            XCTAssertLessThan(relativeError, 1e-5,
                "dbToPower[\(i)]: expected \(expected[i]), got \(power[0][i]), relative error \(relativeError)")
        }
    }

    func testDbToAmplitude() {
        let db: [[Float]] = [[0, 20, 40]]
        let amplitude = Convert.dbToAmplitude(db, ref: 1.0)

        // Use RELATIVE tolerance
        let expected: [Float] = [1, 10, 100]
        for i in 0..<expected.count {
            let relativeError = abs(amplitude[0][i] - expected[i]) / expected[i]
            XCTAssertLessThan(relativeError, 1e-5,
                "dbToAmplitude[\(i)]: expected \(expected[i]), got \(amplitude[0][i]), relative error \(relativeError)")
        }
    }

    func testDbConversionExtendedRange() {
        // Test over a wider range to ensure consistent relative accuracy
        let powers: [Float] = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        let power2D: [[Float]] = [powers]

        let db = Convert.powerToDb(power2D, ref: 1.0, topDb: nil)
        let recovered = Convert.dbToPower(db, ref: 1.0)

        for i in 0..<powers.count {
            let relativeError = abs(recovered[0][i] - powers[i]) / powers[i]
            XCTAssertLessThan(relativeError, 1e-5,
                "Round-trip failed for power=\(powers[i]): got \(recovered[0][i]), relative error \(relativeError)")
        }
    }

    func testTopDbClipping() {
        let power: [[Float]] = [[1.0, 0.001, 0.000001]]
        let db = Convert.powerToDb(power, ref: 1.0, topDb: 20)

        // With topDb=20, values more than 20dB below max should be clipped
        let maxDb = db[0][0]  // Should be 0
        let minAllowed = maxDb - 20

        for (i, value) in db[0].enumerated() {
            XCTAssertGreaterThanOrEqual(value, minAllowed - 1e-5,
                "Value at index \(i) should be >= \(minAllowed), got \(value)")
        }

        // The second value (0.001 = -30dB) should be clipped to -20
        XCTAssertEqual(db[0][1], -20, accuracy: 1e-4,
            "0.001 power (-30dB) should be clipped to -20dB")

        // The third value (0.000001 = -60dB) should be clipped to -20
        XCTAssertEqual(db[0][2], -20, accuracy: 1e-4,
            "0.000001 power (-60dB) should be clipped to -20dB")
    }

    func testDbRoundTrip() {
        let original: [[Float]] = [[0.5, 1.0, 2.0, 5.0, 10.0]]

        let db = Convert.powerToDb(original, topDb: nil)
        let recovered = Convert.dbToPower(db)

        for i in 0..<original[0].count {
            let relativeError = abs(recovered[0][i] - original[0][i]) / original[0][i]
            XCTAssertLessThan(relativeError, 1e-5,
                "Round-trip failed for \(original[0][i]): got \(recovered[0][i])")
        }
    }

    // MARK: - Integration Tests

    func testMelSpectrogramToDbPipeline() async {
        // Test the full pipeline: signal → mel spectrogram → dB
        let sampleRate: Float = 22050
        let nSamples = 4096

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = 0.5 * sin(Float(i) * 0.1)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 1024,
            hopLength: 256,
            nMels: 64
        )

        let mel = await melSpec.transform(signal)
        let melDb = Convert.powerToDb(mel, topDb: 80)

        // Verify shape
        XCTAssertEqual(melDb.count, 64)
        XCTAssertGreaterThan(melDb[0].count, 0)

        // All values should be finite
        for row in melDb {
            for value in row {
                XCTAssertFalse(value.isNaN, "dB value should not be NaN")
                XCTAssertFalse(value.isInfinite, "dB value should not be infinite")
            }
        }

        // With topDb=80, max - min should be <= 80
        let allValues = melDb.flatMap { $0 }
        let maxVal = allValues.max() ?? 0
        let minVal = allValues.min() ?? 0
        XCTAssertLessThanOrEqual(maxVal - minVal, 80.1,
            "Dynamic range should be limited by topDb")
    }
}
