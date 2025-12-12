import XCTest
@testable import SwiftRosaCore

final class WindowsTests: XCTestCase {

    // MARK: - Tolerance Constants
    private let windowTolerance: Float = 1e-6

    // MARK: - Mathematical Reference Values
    // Computed from exact formulas in Windows.swift

    // Hann (length=8, periodic): 0.5 - 0.5 * cos(2*pi*k/8)
    private let hannRef8: [Float] = [
        0.0,                                    // k=0: 0.5 - 0.5*1
        0.5 - 0.5 * 0.7071067811865476,        // k=1: cos(pi/4)
        0.5,                                    // k=2: cos(pi/2) = 0
        0.5 + 0.5 * 0.7071067811865476,        // k=3: cos(3pi/4)
        1.0,                                    // k=4: cos(pi) = -1
        0.5 + 0.5 * 0.7071067811865476,        // k=5
        0.5,                                    // k=6
        0.5 - 0.5 * 0.7071067811865476         // k=7
    ]

    // Bartlett (length=8, periodic): 1 - |k/4 - 1|
    private let bartlettRef8: [Float] = [
        0.0,   // k=0: 1 - |0/4 - 1| = 0
        0.25,  // k=1: 1 - |0.25 - 1| = 0.25
        0.5,   // k=2: 1 - |0.5 - 1| = 0.5
        0.75,  // k=3: 1 - |0.75 - 1| = 0.75
        1.0,   // k=4: 1 - |1 - 1| = 1.0
        0.75,  // k=5: 1 - |1.25 - 1| = 0.75
        0.5,   // k=6: 1 - |1.5 - 1| = 0.5
        0.25   // k=7: 1 - |1.75 - 1| = 0.25
    ]

    // Hamming (length=8, periodic): 0.54 - 0.46 * cos(2*pi*k/8)
    private let hammingRef8: [Float] = [
        0.08,                                   // k=0: 0.54 - 0.46*1
        0.54 - 0.46 * 0.7071067811865476,      // k=1
        0.54,                                   // k=2: cos(pi/2) = 0
        0.54 + 0.46 * 0.7071067811865476,      // k=3
        1.0,                                    // k=4: 0.54 + 0.46
        0.54 + 0.46 * 0.7071067811865476,      // k=5
        0.54,                                   // k=6
        0.54 - 0.46 * 0.7071067811865476       // k=7
    ]

    // Blackman (length=8, periodic): 0.42 - 0.5*cos(2pi*k/8) + 0.08*cos(4pi*k/8)
    private var blackmanRef8: [Float] {
        (0..<8).map { k in
            let angle = 2.0 * Float.pi * Float(k) / 8.0
            return 0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle)
        }
    }

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

    func testHammingMatchesReference() {
        // Verify Hamming window matches exact mathematical formula
        // Hamming: 0.54 - 0.46 * cos(2*pi*k/N)
        let window = Windows.generate(.hamming, length: 8, periodic: true)

        XCTAssertEqual(window.count, hammingRef8.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], hammingRef8[i], accuracy: windowTolerance,
                "Hamming mismatch at index \(i): got \(window[i]), expected \(hammingRef8[i])")
        }
    }

    func testBlackmanWindow() {
        let window = Windows.generate(.blackman, length: 5, periodic: true)

        XCTAssertEqual(window.count, 5)

        // Blackman window endpoints should be close to 0
        XCTAssertEqual(window[0], 0, accuracy: 1e-5)
    }

    func testBlackmanMatchesReference() {
        // Verify Blackman window matches exact mathematical formula
        // Blackman: 0.42 - 0.5*cos(2pi*k/N) + 0.08*cos(4pi*k/N)
        let window = Windows.generate(.blackman, length: 8, periodic: true)

        XCTAssertEqual(window.count, blackmanRef8.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], blackmanRef8[i], accuracy: windowTolerance,
                "Blackman mismatch at index \(i): got \(window[i]), expected \(blackmanRef8[i])")
        }
    }

    func testBartlettWindow() {
        let window = Windows.generate(.bartlett, length: 5, periodic: true)

        XCTAssertEqual(window.count, 5)

        // Bartlett (triangular) should peak at center
        XCTAssertLessThan(window[0], window[2])
        XCTAssertLessThan(window[4], window[2])
    }

    func testBartlettMatchesReference() {
        // Reference values from np.bartlett with periodic behavior
        let window = Windows.generate(.bartlett, length: 8, periodic: true)

        XCTAssertEqual(window.count, bartlettRef8.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], bartlettRef8[i], accuracy: 1e-5,
                "Bartlett mismatch at index \(i): got \(window[i]), expected \(bartlettRef8[i])")
        }
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

    // MARK: - Periodic vs Symmetric Tests

    func testPeriodicVsSymmetricHann() {
        // Periodic and symmetric windows should differ
        let periodic = Windows.generate(.hann, length: 8, periodic: true)
        let symmetric = Windows.generate(.hann, length: 8, periodic: false)

        // Both should have same length
        XCTAssertEqual(periodic.count, symmetric.count)

        // Periodic should NOT be symmetric (last element != first)
        // Symmetric should be symmetric
        XCTAssertEqual(symmetric[0], symmetric[7], accuracy: 1e-6, "Symmetric window should have equal endpoints")
        XCTAssertEqual(symmetric[1], symmetric[6], accuracy: 1e-6)
        XCTAssertEqual(symmetric[2], symmetric[5], accuracy: 1e-6)
        XCTAssertEqual(symmetric[3], symmetric[4], accuracy: 1e-6)

        // Verify they are actually different
        var hasDifference = false
        for i in 0..<periodic.count {
            if abs(periodic[i] - symmetric[i]) > 1e-6 {
                hasDifference = true
                break
            }
        }
        XCTAssertTrue(hasDifference, "Periodic and symmetric windows should differ")
    }

    func testPeriodicVsSymmetricHamming() {
        let periodic = Windows.generate(.hamming, length: 8, periodic: true)
        let symmetric = Windows.generate(.hamming, length: 8, periodic: false)

        // Symmetric should be symmetric
        XCTAssertEqual(symmetric[0], symmetric[7], accuracy: 1e-6)
        XCTAssertEqual(symmetric[1], symmetric[6], accuracy: 1e-6)
        XCTAssertEqual(symmetric[2], symmetric[5], accuracy: 1e-6)
        XCTAssertEqual(symmetric[3], symmetric[4], accuracy: 1e-6)

        // Verify they differ
        var hasDifference = false
        for i in 0..<periodic.count {
            if abs(periodic[i] - symmetric[i]) > 1e-6 {
                hasDifference = true
                break
            }
        }
        XCTAssertTrue(hasDifference, "Periodic and symmetric Hamming windows should differ")
    }

    // MARK: - Mathematical Property Tests

    func testHannWindowSum() {
        // Sum of Hann window should be approximately N/2 for periodic
        let window = Windows.generate(.hann, length: 1024, periodic: true)
        let sum = window.reduce(0, +)
        let expectedSum = Float(1024) / 2.0
        XCTAssertEqual(sum, expectedSum, accuracy: 1.0, "Hann window sum should be ~N/2")
    }

    func testHammingWindowEndpoints() {
        // Hamming window should have non-zero endpoints (0.08)
        let window = Windows.generate(.hamming, length: 100, periodic: true)
        XCTAssertEqual(window[0], 0.08, accuracy: 1e-5, "Hamming endpoint should be 0.08")
    }

    func testBlackmanWindowEndpoints() {
        // Blackman window should have zero endpoints
        let window = Windows.generate(.blackman, length: 100, periodic: true)
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-5, "Blackman endpoint should be 0")
    }

    func testWindowMonotonicity() {
        // All windows should increase from start to center
        let windows: [WindowType] = [.hann, .hamming, .blackman, .bartlett]

        for windowType in windows {
            let window = Windows.generate(windowType, length: 100, periodic: true)
            let halfLen = window.count / 2

            // First half should be non-decreasing
            for i in 1..<halfLen {
                XCTAssertGreaterThanOrEqual(window[i], window[i-1] - 1e-6,
                    "\(windowType) window should be non-decreasing in first half at index \(i)")
            }
        }
    }

    // MARK: - COLA Property Test

    func testHannCOLAProperty() {
        // Hann window with 50% overlap should satisfy COLA (Constant Overlap-Add)
        // Sum of overlapped windows should be constant
        let windowLength = 256
        let hopLength = windowLength / 2  // 50% overlap

        let window = Windows.generate(.hann, length: windowLength, periodic: true)

        // For 50% overlap Hann, sum of two adjacent windows should be constant
        var sum: Float = 0
        for i in 0..<hopLength {
            // Sum of window[i] + window[i + hopLength] should be constant
            let overlap = window[i] + window[i + hopLength]
            if i == 0 {
                sum = overlap
            } else {
                XCTAssertEqual(overlap, sum, accuracy: 1e-5,
                    "COLA property violated at index \(i): got \(overlap), expected \(sum)")
            }
        }
    }

    // MARK: - Librosa Compatibility

    func testHannMatchesLibrosa() {
        // Reference values from librosa.filters.get_window('hann', 8, fftbins=True)
        let window = Windows.generate(.hann, length: 8, periodic: true)

        XCTAssertEqual(window.count, hannRef8.count)
        for i in 0..<window.count {
            XCTAssertEqual(window[i], hannRef8[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }

    // MARK: - Larger Window Reference Tests

    func testLargerHannWindow() {
        // Test with typical STFT window size
        let window = Windows.generate(.hann, length: 2048, periodic: true)

        XCTAssertEqual(window.count, 2048)

        // Verify key properties
        XCTAssertEqual(window[0], 0.0, accuracy: 1e-6, "Start should be 0")
        XCTAssertEqual(window[1024], 1.0, accuracy: 1e-6, "Center should be 1")

        // Verify symmetry for periodic window
        // Note: periodic Hann is NOT symmetric, but nearly so
        for i in 1..<1024 {
            XCTAssertEqual(window[i], window[2048 - i], accuracy: 1e-5,
                "Hann should be nearly symmetric")
        }
    }
}
