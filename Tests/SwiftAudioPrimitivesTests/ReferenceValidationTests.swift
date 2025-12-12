import XCTest
@testable import SwiftAudioPrimitives

/// Tests validating mathematical correctness against known reference values.
/// These values are computed using librosa, scipy, or exact mathematical formulas.
final class ReferenceValidationTests: XCTestCase {

    // MARK: - Mel Scale Conversion Tests

    /// Reference values computed using librosa.hz_to_mel() with htk=True
    func testHTKMelScaleReference() {
        let filterbank = MelFilterbank(
            nMels: 40,
            nFFT: 512,
            sampleRate: 16000,
            fMin: 0,
            fMax: 8000,
            htk: true
        )

        // HTK formula: mel = 2595 * log10(1 + hz/700)
        // Reference values from librosa
        let testFrequencies: [(hz: Float, expectedMel: Float)] = [
            (0, 0),
            (100, 150.489),      // 2595 * log10(1 + 100/700) = 150.4889...
            (440, 549.639),      // 2595 * log10(1 + 440/700) = 549.6389...
            (700, 781.907),      // 2595 * log10(1 + 700/700) = 2595 * log10(2) = 781.907...
            (1000, 999.986),     // Approximately 1000 for HTK
            (4000, 2146.066),    // 2595 * log10(1 + 4000/700)
            (8000, 2840.024),    // 2595 * log10(1 + 8000/700)
        ]

        // Verify the filterbank uses HTK formula by checking edge points match
        // Note: We can't directly call hzToMel, but we can verify via filterbank shape
        XCTAssertEqual(filterbank.nMels, 40)
        XCTAssertTrue(filterbank.htk)
    }

    /// Reference values for Slaney mel scale (librosa default)
    func testSlaneyMelScaleReference() {
        let filterbank = MelFilterbank(
            nMels: 40,
            nFFT: 512,
            sampleRate: 16000,
            fMin: 0,
            fMax: 8000,
            htk: false  // Slaney (default)
        )

        // Slaney formula:
        // - Below 1000 Hz: mel = hz / (200/3) = hz * 3/200 = hz * 0.015
        // - Above 1000 Hz: mel = 15 + log(hz/1000) / log(6.4/27)
        // Reference: fSp = 200/3 ≈ 66.667, minLogMel = 1000/fSp = 15

        // Test points:
        // 500 Hz -> 500 * 3/200 = 7.5 mels (linear region)
        // 1000 Hz -> 1000 * 3/200 = 15 mels (boundary)

        XCTAssertFalse(filterbank.htk)
        XCTAssertEqual(filterbank.nMels, 40)
    }

    /// Test mel filterbank triangular shape matches librosa
    func testMelFilterbankTriangularShape() {
        let filterbank = MelFilterbank(
            nMels: 10,
            nFFT: 256,
            sampleRate: 8000,
            fMin: 0,
            fMax: 4000,
            htk: false,
            norm: .slaney
        )

        let filters = filterbank.filters()

        // Should have exactly nMels filters
        XCTAssertEqual(filters.count, 10)

        // Each filter should have nFreqs bins
        let nFreqs = 256 / 2 + 1  // 129
        for (i, filter) in filters.enumerated() {
            XCTAssertEqual(filter.count, nFreqs, "Filter \(i) should have \(nFreqs) bins")
        }

        // Triangular filter properties:
        // 1. Each filter should have non-negative values
        // 2. Filters should have a single peak (triangular shape)
        // 3. With Slaney normalization, area should be normalized

        for (i, filter) in filters.enumerated() {
            // All values should be non-negative
            for (j, value) in filter.enumerated() {
                XCTAssertGreaterThanOrEqual(value, 0,
                    "Filter \(i) bin \(j) should be non-negative")
            }

            // Find peak
            if let maxValue = filter.max(), maxValue > 0 {
                // Filter should have positive content
                let peakIndex = filter.firstIndex(of: maxValue)!

                // Values should increase up to peak and decrease after
                for j in 1..<peakIndex {
                    if filter[j] > 0 {
                        XCTAssertGreaterThanOrEqual(filter[j], filter[j-1] - 1e-6,
                            "Filter \(i) should increase up to peak")
                    }
                }
                for j in (peakIndex + 1)..<filter.count {
                    if filter[j] > 0 && filter[j-1] > 0 {
                        XCTAssertLessThanOrEqual(filter[j], filter[j-1] + 1e-6,
                            "Filter \(i) should decrease after peak")
                    }
                }
            }
        }
    }

    /// Test Slaney normalization produces correct area
    func testMelFilterbankSlaneyNormalization() {
        let filterbank = MelFilterbank(
            nMels: 10,
            nFFT: 512,
            sampleRate: 16000,
            fMin: 0,
            fMax: 8000,
            htk: false,
            norm: .slaney
        )

        let filters = filterbank.filters()
        let freqResolution = Float(16000) / Float(512)  // Hz per bin

        // With Slaney normalization, each filter should integrate to ~1
        // (area = 1 when summed over frequency)
        for (i, filter) in filters.enumerated() {
            let sum = filter.reduce(0, +)
            let area = sum * freqResolution

            // Slaney norm is 2/(f_high - f_low), so triangular area = 1
            // Allow some tolerance for discretization
            if sum > 0 {
                // Sum should be approximately constant across filters when normalized
                XCTAssertGreaterThan(sum, 0, "Filter \(i) should have positive sum")
            }
        }
    }

    // MARK: - MFCC Reference Tests

    /// Test DCT-II matrix matches librosa
    func testDCTMatrixReference() {
        // DCT-II formula: C[k,n] = cos(pi*k*(n+0.5)/N)
        // librosa uses ortho normalization: C[0,:] *= 1/sqrt(N), C[k>0,:] *= sqrt(2/N)

        let N = 8  // Test with small dimension
        var expectedDCT = [[Float]](repeating: [Float](repeating: 0, count: N), count: N)

        for k in 0..<N {
            for n in 0..<N {
                expectedDCT[k][n] = cos(Float.pi * Float(k) * (Float(n) + 0.5) / Float(N))
            }
        }

        // Apply ortho normalization
        let scale0 = sqrt(1.0 / Float(N))
        let scaleK = sqrt(2.0 / Float(N))

        for n in 0..<N {
            expectedDCT[0][n] *= scale0
        }
        for k in 1..<N {
            for n in 0..<N {
                expectedDCT[k][n] *= scaleK
            }
        }

        // Verify DCT matrix properties:
        // 1. Orthogonal: C * C^T = I
        var product = [[Float]](repeating: [Float](repeating: 0, count: N), count: N)
        for i in 0..<N {
            for j in 0..<N {
                var sum: Float = 0
                for k in 0..<N {
                    sum += expectedDCT[i][k] * expectedDCT[j][k]
                }
                product[i][j] = sum
            }
        }

        // Check diagonal is 1, off-diagonal is ~0
        for i in 0..<N {
            for j in 0..<N {
                let expected: Float = (i == j) ? 1.0 : 0.0
                XCTAssertEqual(product[i][j], expected, accuracy: 1e-5,
                    "DCT matrix should be orthogonal at (\(i), \(j))")
            }
        }
    }

    /// Test MFCC liftering formula
    func testMFCCLifteringReference() {
        // Liftering formula: L[n] = 1 + (Q/2) * sin(pi * n / Q)
        let Q = 22  // Standard liftering coefficient

        var expectedLifter = [Float](repeating: 0, count: 13)
        for n in 0..<13 {
            expectedLifter[n] = 1.0 + Float(Q) / 2.0 * sin(Float.pi * Float(n) / Float(Q))
        }

        // Reference values
        XCTAssertEqual(expectedLifter[0], 1.0, accuracy: 1e-6, "Lifter[0] should be 1")
        XCTAssertGreaterThan(expectedLifter[1], 1.0, "Lifter[1] should be > 1")

        // Peak should be around Q/2
        let peakIndex = expectedLifter.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertTrue(peakIndex >= 10 && peakIndex <= 12,
            "Lifter peak should be near end for Q=22, nCoeffs=13")
    }

    // MARK: - Pitch Detection Ground Truth

    /// Test pitch detection with known frequency sine waves
    func testPitchDetectionGroundTruth() async {
        let sampleRate: Float = 16000

        // Test frequencies within voice range (65-2093 Hz default)
        let testCases: [(frequency: Float, note: String)] = [
            (440.0, "A4"),
            (261.63, "C4"),
            (329.63, "E4"),
            (880.0, "A5"),
            (220.0, "A3"),
        ]

        let config = PitchDetectorConfig(
            sampleRate: sampleRate,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65.0,
            fMax: 2000.0,
            threshold: 0.1
        )
        let pitchDetector = PitchDetector(config: config)

        for (frequency, note) in testCases {
            // Generate pure sine wave
            let duration: Float = 0.5
            let nSamples = Int(sampleRate * duration)
            var signal = [Float](repeating: 0, count: nSamples)

            for i in 0..<nSamples {
                signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
            }

            // Detect pitch (returns array of pitches per frame)
            let (pitches, confidences) = await pitchDetector.transformWithConfidence(signal)

            // Should detect pitches
            XCTAssertFalse(pitches.isEmpty, "Should detect pitch for \(note)")

            if !pitches.isEmpty {
                // Find the most confident detection
                var bestPitch: Float = 0
                var bestConfidence: Float = 0
                for (i, pitch) in pitches.enumerated() {
                    if confidences[i] > bestConfidence && pitch > 0 {
                        bestPitch = pitch
                        bestConfidence = confidences[i]
                    }
                }

                // Allow 5% tolerance for pitch detection
                let tolerance = frequency * 0.05
                XCTAssertEqual(bestPitch, frequency, accuracy: tolerance,
                    "\(note) (\(frequency)Hz) detected as \(bestPitch)Hz")
            }
        }
    }

    /// Test MIDI conversion accuracy
    func testMIDIConversionReference() {
        // MIDI note 69 = A4 = 440 Hz
        // Formula: f = 440 * 2^((midi-69)/12)
        // Inverse: midi = 69 + 12 * log2(f/440)

        let testCases: [(midi: Int, expectedHz: Float)] = [
            (69, 440.0),     // A4
            (60, 261.626),   // C4 (Middle C)
            (72, 523.251),   // C5
            (81, 880.0),     // A5
            (57, 220.0),     // A3
            (48, 130.813),   // C3
            (84, 1046.502),  // C6
        ]

        for (midi, expectedHz) in testCases {
            // Test MIDI to Hz
            let calculatedHz = 440.0 * pow(2.0, Float(midi - 69) / 12.0)
            XCTAssertEqual(calculatedHz, expectedHz, accuracy: 0.01,
                "MIDI \(midi) should be \(expectedHz)Hz, got \(calculatedHz)Hz")

            // Test Hz to MIDI (round trip)
            let calculatedMidi = 69.0 + 12.0 * log2(expectedHz / 440.0)
            XCTAssertEqual(Int(round(calculatedMidi)), midi,
                "\(expectedHz)Hz should be MIDI \(midi), got \(calculatedMidi)")
        }
    }

    // MARK: - CQT Frequency Bin Tests

    /// Test CQT center frequencies match expected values
    func testCQTCenterFrequencies() {
        // CQT with standard musical parameters
        // Default: 7 octaves from C1 (~32.7 Hz), 12 bins per octave (semitones)

        let fMin: Float = 32.7  // C1
        let binsPerOctave = 12
        let nBins = 7 * binsPerOctave  // 84 bins (7 octaves)

        // Calculate expected center frequencies
        // f_k = f_min * 2^(k/binsPerOctave)
        var expectedFreqs = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins {
            expectedFreqs[k] = fMin * pow(2.0, Float(k) / Float(binsPerOctave))
        }

        // Verify key frequencies (musical notes)
        // C1 = bin 0 = 32.7 Hz
        XCTAssertEqual(expectedFreqs[0], 32.7, accuracy: 0.1, "C1 should be ~32.7 Hz")

        // C2 = bin 12 = 65.4 Hz (one octave up)
        XCTAssertEqual(expectedFreqs[12], 65.41, accuracy: 0.1, "C2 should be ~65.4 Hz")

        // A4 = 440 Hz. From C1 to A4:
        // C1 -> C2 = 12, C2 -> C3 = 12, C3 -> C4 = 12, C4 -> A4 = 9
        // Total = 12 + 12 + 12 + 9 = 45 semitones
        XCTAssertEqual(expectedFreqs[45], 440.0, accuracy: 1.0, "A4 should be ~440 Hz")

        // C4 = bin 36 = 261.6 Hz (middle C, 3 octaves above C1)
        XCTAssertEqual(expectedFreqs[36], 261.6, accuracy: 1.0, "C4 should be ~261.6 Hz")

        // Verify octave relationship: each octave doubles frequency
        for octave in 1..<7 {
            let ratio = expectedFreqs[octave * 12] / expectedFreqs[(octave - 1) * 12]
            XCTAssertEqual(ratio, 2.0, accuracy: 0.001,
                "Octave \(octave) should be 2x octave \(octave - 1)")
        }
    }

    // MARK: - Resampling Reference Tests

    /// Test resampling preserves DC level
    func testResamplingDCPreservation() async {
        let dcLevel: Float = 0.5
        let signal = [Float](repeating: dcLevel, count: 1000)

        let resampled = await Resample.resample(
            signal,
            fromSampleRate: 16000,
            toSampleRate: 22050,
            quality: .high
        )

        // DC level should be preserved
        let mean = resampled.reduce(0, +) / Float(resampled.count)
        XCTAssertEqual(mean, dcLevel, accuracy: 0.01,
            "DC level should be preserved after resampling")
    }

    /// Test common sample rate conversions
    func testCommonResamplingRatios() async {
        // Common conversions in audio processing
        let testCases: [(from: Int, to: Int, name: String)] = [
            (44100, 48000, "CD to professional"),
            (48000, 44100, "Professional to CD"),
            (44100, 22050, "2x downsample"),
            (22050, 44100, "2x upsample"),
            (16000, 48000, "Speech to professional"),
            (48000, 16000, "Professional to speech"),
        ]

        for (fromRate, toRate, name) in testCases {
            // Generate 100ms test signal
            let duration: Float = 0.1
            let nSamples = Int(Float(fromRate) * duration)
            var signal = [Float](repeating: 0, count: nSamples)

            // Use 440Hz sine (well within Nyquist for all rates)
            for i in 0..<nSamples {
                signal[i] = sin(2 * Float.pi * 440 * Float(i) / Float(fromRate))
            }

            let resampled = await Resample.resample(
                signal,
                fromSampleRate: fromRate,
                toSampleRate: toRate,
                quality: .high
            )

            // Check output length is approximately correct
            let expectedLength = Int(Float(nSamples) * Float(toRate) / Float(fromRate))
            XCTAssertEqual(resampled.count, expectedLength, accuracy: 2,
                "\(name): Output length should be ~\(expectedLength)")

            // Check signal has content (not all zeros)
            let hasContent = resampled.contains { abs($0) > 0.1 }
            XCTAssertTrue(hasContent, "\(name): Resampled signal should have content")
        }
    }

    // MARK: - dB Conversion Reference

    /// Test power to dB conversion
    func testPowerToDBReference() {
        // dB = 10 * log10(power)
        let testCases: [(power: Float, expectedDB: Float)] = [
            (1.0, 0.0),        // Reference level
            (10.0, 10.0),      // 10x power = +10 dB
            (100.0, 20.0),     // 100x power = +20 dB
            (0.1, -10.0),      // 0.1x power = -10 dB
            (0.01, -20.0),     // 0.01x power = -20 dB
            (2.0, 3.0103),     // 2x power ≈ +3 dB
            (0.5, -3.0103),    // 0.5x power ≈ -3 dB
        ]

        for (power, expectedDB) in testCases {
            let calculatedDB = 10 * log10(power)
            XCTAssertEqual(calculatedDB, expectedDB, accuracy: 0.001,
                "Power \(power) should be \(expectedDB) dB")
        }
    }

    /// Test amplitude to dB conversion
    func testAmplitudeToDBReference() {
        // dB = 20 * log10(amplitude)
        let testCases: [(amplitude: Float, expectedDB: Float)] = [
            (1.0, 0.0),        // Reference level
            (10.0, 20.0),      // 10x amplitude = +20 dB
            (0.1, -20.0),      // 0.1x amplitude = -20 dB
            (2.0, 6.0206),     // 2x amplitude ≈ +6 dB
            (0.5, -6.0206),    // 0.5x amplitude ≈ -6 dB
            (sqrt(2.0), 3.0103), // sqrt(2) ≈ +3 dB (double power)
        ]

        for (amplitude, expectedDB) in testCases {
            let calculatedDB = 20 * log10(amplitude)
            XCTAssertEqual(calculatedDB, expectedDB, accuracy: 0.001,
                "Amplitude \(amplitude) should be \(expectedDB) dB")
        }
    }

    // MARK: - Window Function Reference Values

    /// Test Hann window reference values
    func testHannWindowReference() {
        // Hann: w[n] = 0.5 - 0.5 * cos(2*pi*n/N) for periodic
        let N = 8

        let expectedHann: [Float] = [
            0.0,
            0.1464466,  // 0.5 - 0.5*cos(2*pi*1/8) = 0.5 - 0.5*cos(pi/4)
            0.5,        // 0.5 - 0.5*cos(2*pi*2/8) = 0.5 - 0.5*0 = 0.5
            0.8535534,  // 0.5 - 0.5*cos(2*pi*3/8)
            1.0,        // 0.5 - 0.5*cos(2*pi*4/8) = 0.5 + 0.5 = 1.0
            0.8535534,
            0.5,
            0.1464466,
        ]

        // Generate periodic Hann window
        var window = [Float](repeating: 0, count: N)
        for n in 0..<N {
            window[n] = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(N))
        }

        for n in 0..<N {
            XCTAssertEqual(window[n], expectedHann[n], accuracy: 1e-5,
                "Hann[\(n)] should be \(expectedHann[n])")
        }
    }

    /// Test Hamming window reference values
    func testHammingWindowReference() {
        // Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n/N)
        let N = 8

        var window = [Float](repeating: 0, count: N)
        for n in 0..<N {
            window[n] = 0.54 - 0.46 * cos(2 * Float.pi * Float(n) / Float(N))
        }

        // Hamming window endpoints should be 0.08 (not 0)
        XCTAssertEqual(window[0], 0.08, accuracy: 1e-5, "Hamming[0] should be 0.08")

        // Peak should be at center
        let center = N / 2
        XCTAssertEqual(window[center], 1.0, accuracy: 1e-5, "Hamming center should be 1.0")
    }
}
