import XCTest
@testable import SwiftAudioPrimitives

final class WindowsTests: XCTestCase {

    // MARK: - Basic Tests

    func testHannWindow() {
        let window = Windows.generate(.hann, length: 4, periodic: true)

        XCTAssertEqual(window.count, 4)

        // Hann window: 0.5 - 0.5 * cos(2*pi*k/N)
        // For periodic window with N=4:
        // k=0: 0.5 - 0.5*cos(0) = 0
        // k=1: 0.5 - 0.5*cos(pi/2) = 0.5
        // k=2: 0.5 - 0.5*cos(pi) = 1
        // k=3: 0.5 - 0.5*cos(3*pi/2) = 0.5
        XCTAssertEqual(window[0], 0, accuracy: 1e-6)
        XCTAssertEqual(window[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(window[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(window[3], 0.5, accuracy: 1e-6)
    }

    func testHammingWindow() {
        let window = Windows.generate(.hamming, length: 4, periodic: true)

        XCTAssertEqual(window.count, 4)

        // Hamming: 0.54 - 0.46 * cos(2*pi*k/N)
        XCTAssertEqual(window[0], 0.08, accuracy: 1e-6)  // 0.54 - 0.46
        XCTAssertEqual(window[2], 1.0, accuracy: 1e-6)   // 0.54 + 0.46
    }

    func testBlackmanWindow() {
        let window = Windows.generate(.blackman, length: 5, periodic: true)

        XCTAssertEqual(window.count, 5)

        // Blackman window endpoints should be close to 0
        XCTAssertEqual(window[0], 0, accuracy: 1e-5)
    }

    func testBartlettWindow() {
        let window = Windows.generate(.bartlett, length: 5, periodic: true)

        XCTAssertEqual(window.count, 5)

        // Bartlett (triangular) should peak at center
        XCTAssertLessThan(window[0], window[2])
        XCTAssertLessThan(window[4], window[2])
    }

    func testRectangularWindow() {
        let window = Windows.generate(.rectangular, length: 100, periodic: true)

        XCTAssertEqual(window.count, 100)

        // All values should be 1.0
        for value in window {
            XCTAssertEqual(value, 1.0, accuracy: 1e-10)
        }
    }

    // MARK: - Edge Cases

    func testEmptyWindow() {
        let window = Windows.generate(.hann, length: 0, periodic: true)
        XCTAssertTrue(window.isEmpty)
    }

    func testSingleSampleWindow() {
        let window = Windows.generate(.hann, length: 1, periodic: true)
        XCTAssertEqual(window.count, 1)
        XCTAssertEqual(window[0], 1.0)
    }

    func testSymmetricWindow() {
        // Symmetric windows should be symmetric around center
        let window = Windows.generate(.hann, length: 5, periodic: false)

        XCTAssertEqual(window[0], window[4], accuracy: 1e-6)
        XCTAssertEqual(window[1], window[3], accuracy: 1e-6)
    }

    // MARK: - Librosa Compatibility

    func testHannMatchesLibrosa() {
        // Reference values from librosa.filters.get_window('hann', 8, fftbins=True)
        let window = Windows.generate(.hann, length: 8, periodic: true)

        let expected: [Float] = [
            0.0,
            0.1464466,
            0.5,
            0.8535534,
            1.0,
            0.8535534,
            0.5,
            0.1464466
        ]

        XCTAssertEqual(window.count, expected.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], expected[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }
}
