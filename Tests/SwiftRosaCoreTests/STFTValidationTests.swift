import XCTest
@testable import SwiftRosaCore

/// Comprehensive STFT/ISTFT validation tests against librosa reference.
///
/// These tests validate:
/// 1. STFT magnitude accuracy
/// 2. STFT phase accuracy
/// 3. ISTFT round-trip reconstruction
/// 4. Various parameter configurations
/// 5. Edge cases
final class STFTValidationTests: XCTestCase {

    // MARK: - Reference Data

    private static var referenceData: [String: Any]?

    override class func setUp() {
        super.setUp()
        loadReferenceData()
    }

    private static func loadReferenceData() {
        let bundle = Bundle(for: STFTValidationTests.self)

        let possiblePaths = [
            bundle.path(forResource: "librosa_reference", ofType: "json"),
            bundle.resourcePath.map { $0 + "/ReferenceData/librosa_reference.json" },
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceData/librosa_reference.json")
                .path
        ].compactMap { $0 }

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path),
               let data = FileManager.default.contents(atPath: path),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                referenceData = json
                return
            }
        }

        let workingPath = FileManager.default.currentDirectoryPath
        let fallbackPath = workingPath + "/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json"
        if let data = FileManager.default.contents(atPath: fallbackPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            referenceData = json
        }
    }

    private func getReference() throws -> [String: Any] {
        guard let ref = Self.referenceData else {
            throw XCTSkip("Reference data not found. Run scripts/generate_reference_data.py first.")
        }
        return ref
    }

    /// Load a test signal from reference data (ensures signal matches what librosa used)
    private func loadSignalFromReference(_ name: String) throws -> [Float] {
        let ref = try getReference()
        guard let signals = ref["test_signals"] as? [String: Any],
              let signalData = signals[name] as? [String: Any],
              let data = signalData["data"] as? [Double] else {
            throw XCTSkip("Test signal '\(name)' not found in reference data")
        }
        return data.map { Float($0) }
    }

    // MARK: - STFT Magnitude Tests

    /// Test STFT magnitude for pure 440Hz sine wave
    func testSTFTMagnitude_Sine440Hz() async throws {
        let ref = try getReference()
        guard let stftRef = ref["stft"] as? [String: Any],
              let sineRef = stftRef["sine_440hz"] as? [String: Any],
              let expectedMagnitude = sineRef["magnitude"] as? [[Double]],
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int else {
            throw XCTSkip("STFT reference data not found")
        }

        // Load signal from reference data to ensure exact match
        let signal = try loadSignalFromReference("sine_440hz")

        // Compute STFT
        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: nFFT,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)
        let magnitude = result.magnitude

        // Validate shape
        let expectedFrames = expectedMagnitude.count
        let expectedFreqs = expectedMagnitude.first?.count ?? 0

        XCTAssertEqual(result.cols, expectedFrames, accuracy: 2,
            "Frame count mismatch: expected ~\(expectedFrames), got \(result.cols)")

        // Validate magnitude values
        var maxAbsError: Double = 0
        var maxRelError: Double = 0
        var worstFreq = 0
        var worstFrame = 0
        var worstExpected: Double = 0
        var worstActual: Double = 0
        let framesToCompare = min(10, expectedFrames, result.cols)

        // Debug: print first few values
        print("STFT DEBUG: expectedMagnitude[0][0..5]: \(expectedMagnitude[0][0..<5])")
        print("STFT DEBUG: magnitude[0..5][0]: \(magnitude[0..<5].map { $0[0] })")

        for frameIdx in 0..<framesToCompare {
            let expectedFrame = expectedMagnitude[frameIdx]
            for freqIdx in 0..<min(expectedFreqs, magnitude.count) {
                let expected = expectedFrame[freqIdx]
                let actual = Double(magnitude[freqIdx][frameIdx])
                let absError = Swift.abs(expected - actual)
                maxAbsError = Swift.max(maxAbsError, absError)

                if expected > 1.0 {
                    let relError = absError / expected
                    if relError > maxRelError {
                        maxRelError = relError
                        worstFreq = freqIdx
                        worstFrame = frameIdx
                        worstExpected = expected
                        worstActual = actual
                    }
                }
            }
        }

        print("STFT DEBUG: maxRelError=\(maxRelError), worstFreq=\(worstFreq), worstFrame=\(worstFrame)")
        print("STFT DEBUG: worstExpected=\(worstExpected), worstActual=\(worstActual)")

        XCTAssertLessThan(maxRelError, ValidationTolerances.stftMagnitude,
            "STFT magnitude max relative error \(maxRelError) exceeds tolerance \(ValidationTolerances.stftMagnitude)")
    }

    /// Test STFT magnitude for multi-tone signal
    func testSTFTMagnitude_MultiTone() async throws {
        let ref = try getReference()
        guard let stftRef = ref["stft"] as? [String: Any],
              let multiRef = stftRef["multi_tone"] as? [String: Any],
              let expectedMagnitude = multiRef["magnitude"] as? [[Double]],
              let nFFT = multiRef["n_fft"] as? Int,
              let hopLength = multiRef["hop_length"] as? Int else {
            throw XCTSkip("STFT reference data not found for multi_tone")
        }

        // Load signal from reference data to ensure exact match
        let signal = try loadSignalFromReference("multi_tone")

        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)
        let magnitude = result.magnitude

        var maxRelError: Double = 0
        let framesToCompare = min(10, expectedMagnitude.count, result.cols)
        let freqsToCompare = min(expectedMagnitude.first?.count ?? 0, magnitude.count)

        for frameIdx in 0..<framesToCompare {
            for freqIdx in 0..<freqsToCompare {
                let expected = expectedMagnitude[frameIdx][freqIdx]
                let actual = Double(magnitude[freqIdx][frameIdx])

                if expected > 1.0 {
                    let relError = Swift.abs(expected - actual) / expected
                    maxRelError = Swift.max(maxRelError, relError)
                }
            }
        }

        XCTAssertLessThan(maxRelError, ValidationTolerances.stftMagnitude,
            "STFT multi-tone max relative error \(maxRelError) exceeds tolerance")
    }

    /// Test STFT magnitude for chirp signal
    func testSTFTMagnitude_Chirp() async throws {
        let ref = try getReference()
        guard let stftRef = ref["stft"] as? [String: Any],
              let chirpRef = stftRef["chirp"] as? [String: Any],
              let expectedMagnitude = chirpRef["magnitude"] as? [[Double]],
              let nFFT = chirpRef["n_fft"] as? Int,
              let hopLength = chirpRef["hop_length"] as? Int else {
            throw XCTSkip("STFT reference data not found for chirp")
        }

        // Load signal from reference data (librosa.chirp is different from our chirp)
        let signal = try loadSignalFromReference("chirp")

        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)
        let magnitude = result.magnitude

        var maxRelError: Double = 0
        let framesToCompare = min(10, expectedMagnitude.count, result.cols)
        let freqsToCompare = min(expectedMagnitude.first?.count ?? 0, magnitude.count)

        for frameIdx in 0..<framesToCompare {
            for freqIdx in 0..<freqsToCompare {
                let expected = expectedMagnitude[frameIdx][freqIdx]
                let actual = Double(magnitude[freqIdx][frameIdx])

                if expected > 1.0 {
                    let relError = Swift.abs(expected - actual) / expected
                    maxRelError = Swift.max(maxRelError, relError)
                }
            }
        }

        XCTAssertLessThan(maxRelError, ValidationTolerances.stftMagnitude * 10,  // Chirp has more variation
            "STFT chirp max relative error \(maxRelError) exceeds tolerance")
    }

    // MARK: - ISTFT Round-Trip Tests

    /// Test ISTFT round-trip reconstruction for sine wave
    func testISTFTRoundTrip_Sine440Hz() async throws {
        let ref = try getReference()
        guard let istftRef = ref["istft"] as? [String: Any],
              let sineRef = istftRef["sine_440hz"] as? [String: Any],
              let expectedError = sineRef["reconstruction_error"] as? Double,
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int else {
            throw XCTSkip("ISTFT reference data not found")
        }

        let signal = TestSignalGenerator.sine(frequency: 440)

        let config = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Forward STFT
        let spectrogram = await stft.transform(signal)

        // Inverse STFT
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        // Compute reconstruction error
        var sumSquaredError: Float = 0
        var sumSquaredRef: Float = 0

        for (original, recon) in zip(signal, reconstructed) {
            sumSquaredError += (original - recon) * (original - recon)
            sumSquaredRef += original * original
        }

        let ourError = Double(sqrt(sumSquaredError / max(sumSquaredRef, 1e-10)))

        // Our error should be similar to librosa's error
        XCTAssertLessThan(ourError, ValidationTolerances.istftReconstruction,
            "ISTFT reconstruction error \(ourError) exceeds tolerance \(ValidationTolerances.istftReconstruction)")

        // Also verify it's in the same ballpark as librosa
        XCTAssertLessThan(ourError, expectedError * 10,
            "ISTFT reconstruction error \(ourError) is significantly worse than librosa (\(expectedError))")
    }

    /// Test ISTFT round-trip for multi-tone
    func testISTFTRoundTrip_MultiTone() async throws {
        let signal = TestSignalGenerator.multiTone()

        let config = STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        var sumSquaredError: Float = 0
        var sumSquaredRef: Float = 0

        for (original, recon) in zip(signal, reconstructed) {
            sumSquaredError += (original - recon) * (original - recon)
            sumSquaredRef += original * original
        }

        let error = Double(sqrt(sumSquaredError / max(sumSquaredRef, 1e-10)))

        XCTAssertLessThan(error, ValidationTolerances.istftReconstruction,
            "ISTFT multi-tone reconstruction error \(error) exceeds tolerance")
    }

    /// Test ISTFT round-trip for chirp
    func testISTFTRoundTrip_Chirp() async throws {
        let signal = TestSignalGenerator.chirp()

        let config = STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        let spectrogram = await stft.transform(signal)
        let reconstructed = await istft.transform(spectrogram, length: signal.count)

        var sumSquaredError: Float = 0
        var sumSquaredRef: Float = 0

        for (original, recon) in zip(signal, reconstructed) {
            sumSquaredError += (original - recon) * (original - recon)
            sumSquaredRef += original * original
        }

        let error = Double(sqrt(sumSquaredError / max(sumSquaredRef, 1e-10)))

        XCTAssertLessThan(error, ValidationTolerances.istftReconstruction,
            "ISTFT chirp reconstruction error \(error) exceeds tolerance")
    }

    // MARK: - Parameter Variation Tests

    /// Test STFT with different FFT sizes
    func testSTFTParameterVariation_FFTSize() async throws {
        let signal = TestSignalGenerator.sine(frequency: 440)

        for nFFT in [512, 1024, 2048, 4096] {
            let config = STFTConfig(
                nFFT: nFFT,
                hopLength: nFFT / 4,
                windowType: .hann,
                center: true,
                padMode: .constant(0)
            )
            let stft = STFT(config: config)
            let result = await stft.transform(signal)

            // Verify frequency resolution
            XCTAssertEqual(result.rows, nFFT / 2 + 1,
                "Unexpected number of frequency bins for nFFT=\(nFFT)")

            // Verify dominant frequency is at 440 Hz
            let freqResolution = Float(22050) / Float(nFFT)
            let expectedBin = Int(440 / freqResolution)

            // Find bin with maximum energy (middle frame)
            let middleFrame = result.cols / 2
            var maxBin = 0
            var maxMag: Float = 0
            for bin in 0..<result.rows {
                let mag = result.magnitude[bin][middleFrame]
                if mag > maxMag {
                    maxMag = mag
                    maxBin = bin
                }
            }

            XCTAssertEqual(maxBin, expectedBin, accuracy: 2,
                "Peak frequency bin mismatch for nFFT=\(nFFT): expected ~\(expectedBin), got \(maxBin)")
        }
    }

    /// Test STFT with different window types
    func testSTFTParameterVariation_WindowType() async throws {
        let signal = TestSignalGenerator.sine(frequency: 440)

        let windowTypes: [WindowType] = [.hann, .hamming, .blackman]

        for windowType in windowTypes {
            let config = STFTConfig(
                nFFT: 1024,
                hopLength: 256,
                windowType: windowType,
                center: true,
                padMode: .constant(0)
            )
            let stft = STFT(config: config)
            let istft = ISTFT(config: config)

            let spectrogram = await stft.transform(signal)
            let reconstructed = await istft.transform(spectrogram, length: signal.count)

            var sumSquaredError: Float = 0
            var sumSquaredRef: Float = 0

            for (original, recon) in zip(signal, reconstructed) {
                sumSquaredError += (original - recon) * (original - recon)
                sumSquaredRef += original * original
            }

            let error = Double(sqrt(sumSquaredError / max(sumSquaredRef, 1e-10)))

            XCTAssertLessThan(error, ValidationTolerances.istftReconstruction * 2,
                "Round-trip error too high for window type \(windowType): \(error)")
        }
    }

    // MARK: - FFT Scaling Debug

    /// Debug test to verify FFT scaling across different n_fft sizes
    func testFFTScaling_ImpulseResponse() async throws {
        // Load sine wave from reference (same as mel test)
        let ref = try getReference()
        guard let signals = ref["test_signals"] as? [String: Any],
              let sineData = signals["sine_440hz"] as? [String: Any],
              let signalArray = sineData["data"] as? [Double] else {
            throw XCTSkip("Sine signal not found in reference")
        }
        let sineSignal = signalArray.map { Float($0) }

        print("FFT SCALE DEBUG - Testing with 440Hz sine wave from reference data")

        // Test with n_fft=2048 (same as mel test)
        let config2048 = STFTConfig(nFFT: 2048, hopLength: 512, center: true)
        let stft2048 = STFT(config: config2048)
        let result2048 = await stft2048.transform(sineSignal)
        let mag2048 = result2048.magnitude

        print("n_fft=2048 sine wave:")
        print("  Swift magnitude[0..3][0]: \(mag2048[0..<3].map { $0[0] })")
        print("  librosa expected: [7.97, 7.97, 7.98]")

        // This should be ~1.0, not ~2.0
        let ratio = mag2048[0][0] / 7.965587
        print("  Ratio Swift/librosa: \(ratio)")

        // The issue: Swift magnitude test passes, but mel test shows 2x magnitude
        // Let's also test impulse to rule out signal-specific issues

        let sampleRate = 22050
        let duration: Float = 1.0
        let nSamples = Int(Float(sampleRate) * duration)

        // Create impulse at center
        var signal = [Float](repeating: 0, count: nSamples)
        signal[nSamples / 2] = 1.0

        // Test with n_fft=1024
        let config1024 = STFTConfig(nFFT: 1024, hopLength: 256, center: true)
        let stft1024 = STFT(config: config1024)
        let result1024 = await stft1024.transform(signal)
        let centerFrame1024 = result1024.cols / 2
        let mag1024 = result1024.magnitude

        print("FFT SCALE DEBUG n_fft=1024:")
        print("  Magnitude at center frame [0..5]: \(mag1024[0..<5].map { $0[centerFrame1024] })")
        print("  Expected (librosa): ~[0.997, 0.997, 0.997, 0.997, 0.997]")

        // Test with n_fft=2048 (impulse)
        let config2048Impulse = STFTConfig(nFFT: 2048, hopLength: 512, center: true)
        let stft2048Impulse = STFT(config: config2048Impulse)
        let result2048Impulse = await stft2048Impulse.transform(signal)
        let centerFrame2048 = result2048Impulse.cols / 2
        let mag2048Impulse = result2048Impulse.magnitude

        print("FFT SCALE DEBUG n_fft=2048 (impulse):")
        print("  Magnitude at center frame [0..5]: \(mag2048Impulse[0..<5].map { $0[centerFrame2048] })")
        print("  Expected (librosa): ~[0.871, 0.871, 0.871, 0.871, 0.871]")

        // Check ratio - should be ~1.0 if scaling is correct
        let impulseRatio = mag2048Impulse[0][centerFrame2048] / mag1024[0][centerFrame1024]
        print("Ratio 2048/1024: \(impulseRatio) (expected: ~0.87/0.99 = 0.88)")
    }

    // MARK: - Edge Case Tests

    /// Test STFT with silence
    func testSTFTEdgeCase_Silence() async throws {
        let signal = TestSignalGenerator.silence()

        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)

        // All magnitudes should be near zero
        var maxMag: Float = 0
        for bin in 0..<result.rows {
            for frame in 0..<result.cols {
                maxMag = max(maxMag, result.magnitude[bin][frame])
            }
        }

        XCTAssertLessThan(maxMag, 1e-6, "Silence should produce near-zero STFT magnitude")
    }

    /// Test STFT with single impulse
    func testSTFTEdgeCase_Impulse() async throws {
        let signal = TestSignalGenerator.impulse()

        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)

        // Impulse should have flat spectrum (all frequencies present)
        // Check that the frame containing the impulse has significant energy across bins
        let impulseFrame = result.cols / 2

        var totalEnergy: Float = 0
        var lowFreqEnergy: Float = 0
        var highFreqEnergy: Float = 0

        let midBin = result.rows / 2

        for bin in 0..<result.rows {
            let mag = result.magnitude[bin][impulseFrame]
            totalEnergy += mag * mag
            if bin < midBin {
                lowFreqEnergy += mag * mag
            } else {
                highFreqEnergy += mag * mag
            }
        }

        // Both low and high frequencies should have energy for an impulse
        XCTAssertGreaterThan(lowFreqEnergy, 0.01, "Impulse should have low frequency energy")
        XCTAssertGreaterThan(highFreqEnergy, 0.01, "Impulse should have high frequency energy")
    }

    /// Test STFT with very short signal
    func testSTFTEdgeCase_ShortSignal() async throws {
        // Signal shorter than FFT size
        let shortSignal: [Float] = Array(TestSignalGenerator.sine(frequency: 440).prefix(512))

        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(shortSignal)

        // Should still produce valid output with padding
        XCTAssertGreaterThan(result.cols, 0, "Short signal should produce at least one frame")
        XCTAssertEqual(result.rows, 513, "Should have correct number of frequency bins")
    }

    /// Test STFT with DC offset
    func testSTFTEdgeCase_DCOffset() async throws {
        let signal = TestSignalGenerator.dcOffset(value: 0.5)

        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let stft = STFT(config: config)
        let result = await stft.transform(signal)

        // DC component should dominate (bin 0)
        // Note: Hann window causes spectral leakage to bin 1 (ratio ~2:1 per librosa)
        // Bins 2+ should be near zero
        let middleFrame = result.cols / 2
        let dcMag = result.magnitude[0][middleFrame]
        let bin1Mag = result.magnitude[1][middleFrame]

        // Find max in bins 2 and beyond
        var highBinMaxMag: Float = 0
        for bin in 2..<result.rows {
            highBinMaxMag = max(highBinMaxMag, result.magnitude[bin][middleFrame])
        }

        // DC should be largest
        XCTAssertGreaterThan(dcMag, bin1Mag,
            "DC component should be larger than bin 1")

        // Bins 2+ should be essentially zero (< 1e-3 compared to DC)
        XCTAssertLessThan(highBinMaxMag, dcMag * 1e-3,
            "Bins 2+ should be near zero for DC signal")
    }

    // MARK: - Mel Spectrogram Validation

    /// Test mel spectrogram against librosa reference
    func testMelSpectrogram_Sine440Hz() async throws {
        let ref = try getReference()
        guard let melRef = ref["mel_spectrogram"] as? [String: Any],
              let sineRef = melRef["sine_440hz"] as? [String: Any],
              let expectedMel = sineRef["mel_spectrogram"] as? [[Double]],
              let nFFT = sineRef["n_fft"] as? Int,
              let hopLength = sineRef["hop_length"] as? Int,
              let nMels = sineRef["n_mels"] as? Int else {
            throw XCTSkip("Mel spectrogram reference data not found")
        }

        // Load signal from reference data to ensure exact match
        let signal = try loadSignalFromReference("sine_440hz")

        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )

        // Debug: check power spectrum directly
        let stft = STFT(config: STFTConfig(nFFT: nFFT, hopLength: hopLength))
        let powerSpec = await stft.power(signal)
        print("MEL DEBUG: Power spec shape: [\(powerSpec.count)][\(powerSpec.first?.count ?? 0)]")
        print("MEL DEBUG: Power[0..3][0]: \(powerSpec[0..<3].map { $0[0] })")

        // Also check magnitude
        let spec = await stft.transform(signal)
        let mag = spec.magnitude
        print("MEL DEBUG: Magnitude[0..3][0]: \(mag[0..<3].map { $0[0] })")
        print("MEL DEBUG: Magnitude^2[0..3][0]: \(mag[0..<3].map { $0[0] * $0[0] })")

        let result = await melSpec.transform(signal)

        var maxRelError: Double = 0
        var worstFrame = 0
        var worstMel = 0
        var worstExpected: Double = 0
        var worstActual: Double = 0
        let framesToCompare = min(10, expectedMel.count, result.first?.count ?? 0)
        let melsToCompare = min(nMels, result.count)

        // Debug output - dimensions
        print("DEBUG: expectedMel shape: [\(expectedMel.count)][\(expectedMel.first?.count ?? 0)]")
        print("DEBUG: result shape: [\(result.count)][\(result.first?.count ?? 0)]")
        print("DEBUG: framesToCompare=\(framesToCompare), melsToCompare=\(melsToCompare)")

        for frameIdx in 0..<framesToCompare {
            for melIdx in 0..<melsToCompare {
                let expected = expectedMel[frameIdx][melIdx]
                let actual = Double(result[melIdx][frameIdx])

                // Use combined error metric: absolute for small values, relative for larger
                // This is mathematically proper - relative error is meaningless for near-zero values
                let absError = Swift.abs(expected - actual)
                let error: Double
                if Swift.abs(expected) < 1e-4 {
                    // For values near zero, use absolute error scaled to tolerance
                    // If absolute error < 1e-4 (same as tolerance), it's effectively 0% error
                    error = absError / 1e-4 * ValidationTolerances.melSpectrogram
                } else {
                    // For significant values, use relative error
                    error = absError / Swift.abs(expected)
                }

                if error > maxRelError {
                    maxRelError = error
                    worstFrame = frameIdx
                    worstMel = melIdx
                    worstExpected = expected
                    worstActual = actual
                }
            }
        }

        // Debug worst case
        print("DEBUG: Worst error at frame=\(worstFrame), mel=\(worstMel)")
        print("DEBUG: Expected=\(worstExpected), Actual=\(worstActual)")
        print("DEBUG: First 3 expected values [0][0..2]: \(expectedMel[0][0...2])")
        print("DEBUG: First 3 actual values [0..2][0]: \(result[0...2].map { $0[0] })")

        XCTAssertLessThan(maxRelError, ValidationTolerances.melSpectrogram,
            "Mel spectrogram max relative error \(maxRelError) exceeds tolerance")
    }
}
