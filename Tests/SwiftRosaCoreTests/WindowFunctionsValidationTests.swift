import XCTest
@testable import SwiftRosaCore

/// Window function validation tests against scipy/librosa reference.
///
/// Validates window generation for:
/// - Hann (periodic and symmetric)
/// - Hamming (periodic and symmetric)
/// - Blackman (periodic and symmetric)
/// - Bartlett (periodic and symmetric)
///
/// Reference: scipy 1.16.3 (used by librosa)
final class WindowFunctionsValidationTests: XCTestCase {

    // MARK: - Reference Data

    private static var referenceData: [String: Any]?
    private static var windowLength: Int = 1024

    override class func setUp() {
        super.setUp()
        loadReferenceData()
    }

    private static func loadReferenceData() {
        let possiblePaths = [
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceData/librosa_reference.json")
                .path,
            FileManager.default.currentDirectoryPath + "/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json"
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path),
               let data = FileManager.default.contents(atPath: path),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                referenceData = json

                // Extract window length
                if let windowsRef = json["windows"] as? [String: Any],
                   let length = windowsRef["window_length"] as? Int {
                    windowLength = length
                }
                return
            }
        }
    }

    private func getWindowReference(_ name: String) throws -> [Float] {
        guard let ref = Self.referenceData,
              let windowsRef = ref["windows"] as? [String: Any],
              let windows = windowsRef["windows"] as? [String: Any],
              let windowData = windows[name] as? [Double] else {
            throw XCTSkip("Window reference data for '\(name)' not found")
        }
        return windowData.map { Float($0) }
    }

    // MARK: - Hann Window Tests

    /// Test Hann window (periodic) against scipy reference
    func testHannWindow_Periodic() throws {
        let expected = try getWindowReference("hann_periodic")
        let actual = Windows.generate(.hann, length: Self.windowLength, periodic: true)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Hann periodic max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Hann periodic max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    /// Test Hann window (symmetric) against scipy reference
    func testHannWindow_Symmetric() throws {
        let expected = try getWindowReference("hann_symmetric")
        let actual = Windows.generate(.hann, length: Self.windowLength, periodic: false)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Hann symmetric max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Hann symmetric max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    // MARK: - Hamming Window Tests

    /// Test Hamming window (periodic) against scipy reference
    func testHammingWindow_Periodic() throws {
        let expected = try getWindowReference("hamming_periodic")
        let actual = Windows.generate(.hamming, length: Self.windowLength, periodic: true)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Hamming periodic max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Hamming periodic max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    /// Test Hamming window (symmetric) against scipy reference
    func testHammingWindow_Symmetric() throws {
        let expected = try getWindowReference("hamming_symmetric")
        let actual = Windows.generate(.hamming, length: Self.windowLength, periodic: false)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Hamming symmetric max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Hamming symmetric max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    // MARK: - Blackman Window Tests

    /// Test Blackman window (periodic) against scipy reference
    func testBlackmanWindow_Periodic() throws {
        let expected = try getWindowReference("blackman_periodic")
        let actual = Windows.generate(.blackman, length: Self.windowLength, periodic: true)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Blackman periodic max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Blackman periodic max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    /// Test Blackman window (symmetric) against scipy reference
    func testBlackmanWindow_Symmetric() throws {
        let expected = try getWindowReference("blackman_symmetric")
        let actual = Windows.generate(.blackman, length: Self.windowLength, periodic: false)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Blackman symmetric max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Blackman symmetric max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    // MARK: - Bartlett Window Tests

    /// Test Bartlett window (periodic) against scipy reference
    func testBartlettWindow_Periodic() throws {
        let expected = try getWindowReference("bartlett_periodic")
        let actual = Windows.generate(.bartlett, length: Self.windowLength, periodic: true)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Bartlett periodic max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Bartlett periodic max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    /// Test Bartlett window (symmetric) against scipy reference
    func testBartlettWindow_Symmetric() throws {
        let expected = try getWindowReference("bartlett_symmetric")
        let actual = Windows.generate(.bartlett, length: Self.windowLength, periodic: false)

        XCTAssertEqual(actual.count, expected.count, "Window length mismatch")

        var maxError: Double = 0
        for i in 0..<expected.count {
            let error = abs(Double(expected[i]) - Double(actual[i]))
            maxError = max(maxError, error)
        }

        print("Bartlett symmetric max error: \(String(format: "%.4e", maxError))")
        XCTAssertLessThan(maxError, ValidationTolerances.window,
            "Bartlett symmetric max error \(maxError) exceeds tolerance \(ValidationTolerances.window)")
    }

    // MARK: - Window Properties Tests

    /// Verify window starts at expected value (0 for most windows except hamming)
    func testWindowStartValues() {
        let length = 256

        // Hann starts at 0
        let hann = Windows.generate(.hann, length: length, periodic: true)
        XCTAssertEqual(hann[0], 0.0, accuracy: 1e-6, "Hann should start at 0")

        // Hamming starts at 0.08 (not 0)
        let hamming = Windows.generate(.hamming, length: length, periodic: true)
        XCTAssertEqual(hamming[0], 0.08, accuracy: 1e-4, "Hamming should start at ~0.08")

        // Blackman starts at 0
        let blackman = Windows.generate(.blackman, length: length, periodic: true)
        XCTAssertEqual(blackman[0], 0.0, accuracy: 1e-6, "Blackman should start at 0")

        // Bartlett starts at 0
        let bartlett = Windows.generate(.bartlett, length: length, periodic: true)
        XCTAssertEqual(bartlett[0], 0.0, accuracy: 1e-6, "Bartlett should start at 0")
    }

    /// Verify window peak is at center and equals 1.0
    func testWindowPeakValues() {
        let length = 256
        let center = length / 2

        // All windows should peak at 1.0 at the center
        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett]

        for type in windowTypes {
            let window = Windows.generate(type, length: length, periodic: true)

            // Find max value
            let maxVal = window.max()!
            XCTAssertEqual(maxVal, 1.0, accuracy: 0.02,
                "\(type) window should have peak near 1.0, got \(maxVal)")

            // Peak should be near center
            let peakIndex = window.firstIndex(of: maxVal)!
            XCTAssertTrue(abs(peakIndex - center) <= 1,
                "\(type) window peak should be at center")
        }
    }

    /// Verify symmetric windows are actually symmetric
    func testWindowSymmetry() {
        let length = 256

        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett]

        for type in windowTypes {
            let window = Windows.generate(type, length: length, periodic: false)

            // Check symmetry: w[i] == w[n-1-i]
            for i in 0..<length / 2 {
                let j = length - 1 - i
                XCTAssertEqual(window[i], window[j], accuracy: 1e-6,
                    "\(type) symmetric window should satisfy w[\(i)] == w[\(j)]")
            }
        }
    }

    // MARK: - Edge Cases

    /// Test window with length 1
    func testWindowLength1() {
        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett, .rectangular]

        for type in windowTypes {
            let window = Windows.generate(type, length: 1, periodic: true)
            XCTAssertEqual(window.count, 1, "\(type) window should have length 1")
            XCTAssertEqual(window[0], 1.0, "\(type) window[0] should be 1.0 for length 1")
        }
    }

    /// Test window with length 0
    func testWindowLength0() {
        let windowTypes: [WindowType] = [.hann, .hamming, .blackman, .bartlett, .rectangular]

        for type in windowTypes {
            let window = Windows.generate(type, length: 0, periodic: true)
            XCTAssertTrue(window.isEmpty, "\(type) window should be empty for length 0")
        }
    }

    /// Test rectangular window
    func testRectangularWindow() {
        let length = 256
        let window = Windows.generate(.rectangular, length: length, periodic: true)

        XCTAssertEqual(window.count, length)

        // All values should be 1.0
        for value in window {
            XCTAssertEqual(value, 1.0, accuracy: 1e-10, "Rectangular window should be all 1.0")
        }
    }
}
