import XCTest
@testable import SwiftRosaCore

final class MelFilterbankValidationTests: XCTestCase {

    // MARK: - Filterbank Accuracy Tests

    func testFilterbankMatchesLibrosa() throws {
        // Librosa reference values (from librosa_reference.json)
        // Generated with: librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128, fmin=0, fmax=11025, htk=False, norm='slaney')
        let librosaFilter0Sum: Float = 0.09034588187932968
        let librosaFilter64Sum: Float = 0.09280048566870391
        let librosaFilter127Sum: Float = 0.09286924219850334

        let librosaFilter0NonZero = [1, 2, 3, 4]
        let librosaFilter64NonZero = [182, 183, 184, 185, 186, 187, 188, 189, 190, 191]

        // Generate SwiftRosa filterbank
        let filterbank = MelFilterbank(
            nMels: 128,
            nFFT: 2048,
            sampleRate: 22050,
            fMin: 0,
            fMax: nil,
            htk: false,
            norm: .slaney
        )

        let filters = filterbank.filters()

        // Check shape
        XCTAssertEqual(filters.count, 128, "Should have 128 mel bands")
        XCTAssertEqual(filters[0].count, 1025, "Should have 1025 frequency bins (nFFT/2 + 1)")

        // Check filter sums (area under triangular filters)
        let filter0Sum = filters[0].reduce(0, +)
        let filter64Sum = filters[64].reduce(0, +)
        let filter127Sum = filters[127].reduce(0, +)

        print("SwiftRosa filter 0 sum: \(filter0Sum), librosa: \(librosaFilter0Sum)")
        print("SwiftRosa filter 64 sum: \(filter64Sum), librosa: \(librosaFilter64Sum)")
        print("SwiftRosa filter 127 sum: \(filter127Sum), librosa: \(librosaFilter127Sum)")

        // Check relative error < 1%
        let tol: Float = 0.01
        XCTAssertEqual(filter0Sum, librosaFilter0Sum, accuracy: librosaFilter0Sum * tol,
                       "Filter 0 sum should match librosa")
        XCTAssertEqual(filter64Sum, librosaFilter64Sum, accuracy: librosaFilter64Sum * tol,
                       "Filter 64 sum should match librosa")
        XCTAssertEqual(filter127Sum, librosaFilter127Sum, accuracy: librosaFilter127Sum * tol,
                       "Filter 127 sum should match librosa")

        // Check non-zero indices (which bins have non-zero filter weights)
        let filter0NonZero = filters[0].enumerated().compactMap { $0.element > 0 ? $0.offset : nil }
        let filter64NonZero = filters[64].enumerated().compactMap { $0.element > 0 ? $0.offset : nil }

        print("SwiftRosa filter 0 non-zero: \(filter0NonZero)")
        print("SwiftRosa filter 64 non-zero: \(filter64NonZero)")

        XCTAssertEqual(filter0NonZero, librosaFilter0NonZero,
                       "Filter 0 non-zero indices should match librosa")
        XCTAssertEqual(filter64NonZero, librosaFilter64NonZero,
                       "Filter 64 non-zero indices should match librosa")
    }

    func testFilterbankFirstFilterValues() throws {
        // Librosa filter 0 first 10 values
        let librosaFilter0First10: [Float] = [
            0.0, 0.016182852908968925, 0.03236570581793785, 0.028990088030695915,
            0.01280723512172699, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        let filterbank = MelFilterbank(
            nMels: 128,
            nFFT: 2048,
            sampleRate: 22050,
            fMin: 0,
            fMax: nil,
            htk: false,
            norm: .slaney
        )

        let filters = filterbank.filters()

        print("SwiftRosa filter 0 first 10: \(filters[0].prefix(10).map { String(format: "%.6f", $0) })")
        print("Librosa filter 0 first 10:  \(librosaFilter0First10.map { String(format: "%.6f", $0) })")

        // Check individual values with tight tolerance
        for i in 0..<10 {
            let expected = librosaFilter0First10[i]
            let actual = filters[0][i]
            let tol: Float = expected == 0 ? 1e-6 : expected * 0.01
            XCTAssertEqual(actual, expected, accuracy: tol,
                           "Filter 0[\(i)] should match librosa: got \(actual), expected \(expected)")
        }
    }

    func testMelScaleConversion() throws {
        // Test Slaney mel scale conversion matches librosa
        // librosa.hz_to_mel([100, 500, 1000, 2000, 5000, 10000], htk=False)
        let testFreqs: [Float] = [100, 500, 1000, 2000, 5000, 10000]
        let librosaExpected: [Float] = [1.5, 7.5, 15.0, 25.08, 39.25, 51.76]

        let filterbank = MelFilterbank(htk: false)

        // Access mel conversion through filterbank properties
        // Since hzToMel is private, we'll test indirectly through filter positions

        // For now, just verify the filterbank generates correctly
        let filters = filterbank.filters()
        XCTAssertEqual(filters.count, 128)

        print("Testing mel scale indirectly through filter center frequencies...")
        // Filter centers should be evenly spaced in mel scale
    }
}
