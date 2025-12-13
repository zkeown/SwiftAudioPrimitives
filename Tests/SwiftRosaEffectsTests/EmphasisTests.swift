import XCTest
@testable import SwiftRosaEffects

final class EmphasisTests: XCTestCase {

    // MARK: - Pre-emphasis Tests

    func testPreemphasisBasic() {
        // Simple test signal
        let signal: [Float] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let coef: Float = 0.97

        let result = Emphasis.preemphasis(signal, coef: coef)

        // Verify length matches input
        XCTAssertEqual(result.count, signal.count)

        // First sample should be unchanged (librosa behavior)
        XCTAssertEqual(result[0], signal[0], accuracy: 1e-6)

        // Verify formula: y[n] = x[n] - coef * x[n-1]
        for i in 1..<signal.count {
            let expected = signal[i] - coef * signal[i - 1]
            XCTAssertEqual(result[i], expected, accuracy: 1e-6,
                "Pre-emphasis incorrect at index \(i)")
        }
    }

    func testPreemphasisWithDefaultCoef() {
        let signal: [Float] = [1.0, 2.0, 3.0, 4.0]

        // Default coef should be 0.97
        let result = Emphasis.preemphasis(signal)

        XCTAssertEqual(result.count, signal.count)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(result[1], 2.0 - 0.97 * 1.0, accuracy: 1e-6)
    }

    func testPreemphasisZeroCoef() {
        let signal: [Float] = [1.0, 2.0, 3.0, 4.0]

        // Zero coefficient should return unchanged signal
        let result = Emphasis.preemphasis(signal, coef: 0.0)

        for i in 0..<signal.count {
            XCTAssertEqual(result[i], signal[i], accuracy: 1e-6)
        }
    }

    func testPreemphasisEmptySignal() {
        let signal: [Float] = []
        let result = Emphasis.preemphasis(signal)
        XCTAssertEqual(result.count, 0)
    }

    func testPreemphasisInvalidCoef() {
        let signal: [Float] = [1.0, 2.0, 3.0]

        // Coef >= 1 should return unchanged signal
        let result1 = Emphasis.preemphasis(signal, coef: 1.0)
        let result2 = Emphasis.preemphasis(signal, coef: -0.1)

        for i in 0..<signal.count {
            XCTAssertEqual(result1[i], signal[i], accuracy: 1e-6)
            XCTAssertEqual(result2[i], signal[i], accuracy: 1e-6)
        }
    }

    // MARK: - De-emphasis Tests

    func testDeemphasisBasic() {
        let signal: [Float] = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
        let coef: Float = 0.97

        let result = Emphasis.deemphasis(signal, coef: coef)

        // Verify length matches input
        XCTAssertEqual(result.count, signal.count)

        // First sample unchanged
        XCTAssertEqual(result[0], signal[0], accuracy: 1e-6)

        // Verify formula: y[n] = x[n] + coef * y[n-1]
        var expected: [Float] = [signal[0]]
        for i in 1..<signal.count {
            expected.append(signal[i] + coef * expected[i - 1])
        }

        for i in 0..<signal.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-6,
                "De-emphasis incorrect at index \(i)")
        }
    }

    func testDeemphasisWithDefaultCoef() {
        let signal: [Float] = [1.0, 0.03, 0.06, 0.09]

        let result = Emphasis.deemphasis(signal)

        XCTAssertEqual(result.count, signal.count)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-6)
    }

    func testDeemphasisEmptySignal() {
        let signal: [Float] = []
        let result = Emphasis.deemphasis(signal)
        XCTAssertEqual(result.count, 0)
    }

    // MARK: - Roundtrip Tests

    func testPreDeemphasisRoundtrip() {
        // Pre-emphasis followed by de-emphasis should approximately restore original
        let original: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        let coef: Float = 0.97

        let preemphasized = Emphasis.preemphasis(original, coef: coef)
        let restored = Emphasis.deemphasis(preemphasized, coef: coef)

        // Should match original (with some numerical tolerance)
        for i in 0..<original.count {
            XCTAssertEqual(restored[i], original[i], accuracy: 1e-4,
                "Roundtrip mismatch at index \(i)")
        }
    }

    func testDePreemphasisRoundtrip() {
        // De-emphasis followed by pre-emphasis should also restore
        let original: [Float] = [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]
        let coef: Float = 0.95

        let deemphasized = Emphasis.deemphasis(original, coef: coef)
        let restored = Emphasis.preemphasis(deemphasized, coef: coef)

        for i in 0..<original.count {
            XCTAssertEqual(restored[i], original[i], accuracy: 1e-4,
                "Roundtrip mismatch at index \(i)")
        }
    }

    // MARK: - Array Extension Tests

    func testArrayPreemphasizedExtension() {
        let signal: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]

        let result1 = signal.preemphasized()
        let result2 = Emphasis.preemphasis(signal)

        for i in 0..<signal.count {
            XCTAssertEqual(result1[i], result2[i], accuracy: 1e-6)
        }
    }

    func testArrayDeemphasizedExtension() {
        let signal: [Float] = [1.0, 0.5, 0.3, 0.2, 0.1]

        let result1 = signal.deemphasized()
        let result2 = Emphasis.deemphasis(signal)

        for i in 0..<signal.count {
            XCTAssertEqual(result1[i], result2[i], accuracy: 1e-6)
        }
    }

    // MARK: - Frequency Response Behavior

    func testPreemphasisBoostsHighFrequencies() {
        // A signal with a step change should show larger values after pre-emphasis
        // because pre-emphasis acts as a high-pass filter
        let signal: [Float] = [0, 0, 0, 1, 1, 1, 1]

        let result = Emphasis.preemphasis(signal, coef: 0.97)

        // At the step, pre-emphasis should produce a spike
        let originalChange = signal[3] - signal[2]  // = 1.0
        let emphasizedChange = result[3] - result[2]

        // The change should be amplified
        XCTAssertGreaterThan(emphasizedChange, originalChange * 0.5)
    }
}
