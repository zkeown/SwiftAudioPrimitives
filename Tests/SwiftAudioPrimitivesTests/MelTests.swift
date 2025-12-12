import XCTest
@testable import SwiftAudioPrimitives

final class MelTests: XCTestCase {

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

        // With Slaney normalization, filters should have area = 1
        // Each filter is triangular, so sum should be related to bandwidth
        for filter in filters {
            let sum = filter.reduce(0, +)
            // Sum should be positive and reasonable
            XCTAssertGreaterThan(sum, 0)
        }
    }

    func testMelFilterbankFrequencyRange() {
        let filterbank = MelFilterbank(
            nMels: 64,
            nFFT: 1024,
            sampleRate: 22050,
            fMin: 100,
            fMax: 8000
        )
        let filters = filterbank.filters()

        // Filters below fMin should be mostly zero
        // Filters above fMax should be mostly zero
        XCTAssertEqual(filters.count, 64)

        // First filter should start around fMin
        let firstFilter = filters[0]
        let firstNonZero = firstFilter.firstIndex { $0 > 0 } ?? 0
        let binFreq = 22050.0 / 1024.0 * Float(firstNonZero)
        XCTAssertGreaterThanOrEqual(binFreq, 50)  // Should be near fMin
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

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 80
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

        // The maximum should be somewhere in the middle bands (around 1000 Hz)
        XCTAssertGreaterThan(maxBin, 10)
        XCTAssertLessThan(maxBin, 60)
    }

    func testMelSpectrogramNonNegative() async {
        let signal = (0..<4096).map { Float.random(in: -1...1) * Float($0) / 4096 }

        let melSpec = MelSpectrogram(sampleRate: 22050, nFFT: 1024, hopLength: 256, nMels: 64)
        let result = await melSpec.transform(signal)

        for row in result {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0, "Mel spectrogram should be non-negative")
            }
        }
    }

    // MARK: - dB Conversion Tests

    func testPowerToDb() {
        let power: [[Float]] = [[1.0, 10.0, 100.0, 1000.0]]
        let db = Convert.powerToDb(power, ref: 1.0, topDb: nil)

        XCTAssertEqual(db[0][0], 0, accuracy: 1e-5)      // 10*log10(1) = 0
        XCTAssertEqual(db[0][1], 10, accuracy: 1e-5)    // 10*log10(10) = 10
        XCTAssertEqual(db[0][2], 20, accuracy: 1e-5)    // 10*log10(100) = 20
        XCTAssertEqual(db[0][3], 30, accuracy: 1e-5)    // 10*log10(1000) = 30
    }

    func testAmplitudeToDb() {
        let amplitude: [[Float]] = [[1.0, 10.0, 100.0]]
        let db = Convert.amplitudeToDb(amplitude, ref: 1.0, topDb: nil)

        XCTAssertEqual(db[0][0], 0, accuracy: 1e-5)      // 20*log10(1) = 0
        XCTAssertEqual(db[0][1], 20, accuracy: 1e-5)    // 20*log10(10) = 20
        XCTAssertEqual(db[0][2], 40, accuracy: 1e-5)    // 20*log10(100) = 40
    }

    func testDbToPower() {
        let db: [[Float]] = [[0, 10, 20, 30]]
        let power = Convert.dbToPower(db, ref: 1.0)

        XCTAssertEqual(power[0][0], 1, accuracy: 1e-5)
        XCTAssertEqual(power[0][1], 10, accuracy: 1e-4)
        XCTAssertEqual(power[0][2], 100, accuracy: 1e-3)
        XCTAssertEqual(power[0][3], 1000, accuracy: 1e-2)
    }

    func testDbToAmplitude() {
        let db: [[Float]] = [[0, 20, 40]]
        let amplitude = Convert.dbToAmplitude(db, ref: 1.0)

        XCTAssertEqual(amplitude[0][0], 1, accuracy: 1e-5)
        XCTAssertEqual(amplitude[0][1], 10, accuracy: 1e-4)
        XCTAssertEqual(amplitude[0][2], 100, accuracy: 1e-3)
    }

    func testTopDbClipping() {
        let power: [[Float]] = [[1.0, 0.001, 0.000001]]
        let db = Convert.powerToDb(power, ref: 1.0, topDb: 20)

        // With topDb=20, values more than 20dB below max should be clipped
        let maxDb = db[0][0]  // Should be 0
        let minAllowed = maxDb - 20

        for value in db[0] {
            XCTAssertGreaterThanOrEqual(value, minAllowed - 1e-5)
        }
    }

    func testDbRoundTrip() {
        let original: [[Float]] = [[0.5, 1.0, 2.0, 5.0, 10.0]]

        let db = Convert.powerToDb(original, topDb: nil)
        let recovered = Convert.dbToPower(db)

        for i in 0..<original[0].count {
            XCTAssertEqual(original[0][i], recovered[0][i], accuracy: 1e-5)
        }
    }
}
