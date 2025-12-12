import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Tests for newly implemented transforms and decomposition features.
final class NewTransformsTests: XCTestCase {

    // MARK: - Test Utilities

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    private func generateChord(frequencies: [Float], sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            var sample: Float = 0
            for freq in frequencies {
                sample += sin(2.0 * Float.pi * freq * Float(i) / sampleRate)
            }
            return sample / Float(frequencies.count)
        }
    }

    private func generateNoise(length: Int, seed: Int = 42) -> [Float] {
        srand48(seed)
        return (0..<length).map { _ in Float(drand48() * 2 - 1) }
    }

    // MARK: - Tonnetz Tests

    func testTonnetzBasic() async {
        let tonnetz = Tonnetz()

        // Generate C major chord (C4, E4, G4)
        let signal = generateChord(frequencies: [261.63, 329.63, 392.00], sampleRate: 22050, duration: 1.0)

        let features = await tonnetz.transform(signal)

        // Should have 6 tonnetz dimensions
        XCTAssertEqual(features.count, 6, "Tonnetz should have 6 dimensions")
        XCTAssertGreaterThan(features[0].count, 0, "Should have output frames")

        // All values should be finite
        for row in features {
            for val in row {
                XCTAssertFalse(val.isNaN)
                XCTAssertFalse(val.isInfinite)
            }
        }
    }

    func testTonnetzFromChroma() {
        let tonnetz = Tonnetz()

        // Create simple chroma (C major)
        var chroma: [[Float]] = [[Float]](repeating: [Float](repeating: 0, count: 10), count: 12)
        chroma[0] = [Float](repeating: 1.0, count: 10)  // C
        chroma[4] = [Float](repeating: 1.0, count: 10)  // E
        chroma[7] = [Float](repeating: 1.0, count: 10)  // G

        let features = tonnetz.transform(chroma: chroma)

        XCTAssertEqual(features.count, 6)
        XCTAssertEqual(features[0].count, 10)
    }

    // MARK: - Pseudo-CQT Tests

    func testPseudoCQTBasic() async {
        let pcqt = PseudoCQT(config: PseudoCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        let spectrum = await pcqt.transform(signal)

        // Check output shape
        XCTAssertEqual(spectrum.real.count, 84, "Should have 84 frequency bins")
        XCTAssertGreaterThan(spectrum.real[0].count, 0, "Should have output frames")
    }

    func testPseudoCQTMagnitude() async {
        let pcqt = PseudoCQT()

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)
        let magnitude = await pcqt.magnitude(signal)

        // Verify we got output
        XCTAssertEqual(magnitude.count, 84, "Should have 84 bins")
        XCTAssertGreaterThan(magnitude[0].count, 0, "Should have frames")

        // Find peak bin
        var maxBin = 0
        var maxAvg: Float = 0
        for bin in 0..<magnitude.count {
            let avg = magnitude[bin].reduce(0, +) / Float(magnitude[bin].count)
            if avg > maxAvg {
                maxAvg = avg
                maxBin = bin
            }
        }

        // Verify the peak is near A4 (440 Hz)
        // With fMin=32.70 (C1) and 12 bins/octave:
        // Bin 44 = 415.26 Hz (G#4), Bin 45 = 439.96 Hz (A4)
        // Due to spectral leakage and bin mapping, peak may be at 44 or 45
        // Accept either as valid
        XCTAssertTrue(maxBin == 44 || maxBin == 45,
                      "Peak at 440 Hz should be at bin 44 or 45, got \(maxBin)")

        // Verify peak bin has significantly more energy than neighbors
        let peakEnergy = magnitude[maxBin].reduce(0, +) / Float(magnitude[maxBin].count)
        let lowerEnergy = maxBin > 0 ? magnitude[maxBin - 1].reduce(0, +) / Float(magnitude[maxBin - 1].count) : 0
        let upperEnergy = maxBin < 83 ? magnitude[maxBin + 1].reduce(0, +) / Float(magnitude[maxBin + 1].count) : 0

        XCTAssertGreaterThan(peakEnergy, lowerEnergy, "Peak should have more energy than lower neighbor")
        XCTAssertGreaterThan(peakEnergy, upperEnergy, "Peak should have more energy than upper neighbor")
    }

    // MARK: - Harmonic CQT Tests

    func testHarmonicCQTBasic() async {
        let hcqt = HarmonicCQT(config: HarmonicCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 36,  // 3 octaves
            nHarmonics: 3
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        // transform returns [ComplexMatrix] - one ComplexMatrix per harmonic
        let spectrum = await hcqt.transform(signal)

        // Should have nHarmonics ComplexMatrix objects
        XCTAssertEqual(spectrum.count, 3, "Should have 3 harmonic layers")
        // Each ComplexMatrix.real has shape (nBins, nFrames)
        XCTAssertEqual(spectrum[0].real.count, 36, "Each layer should have 36 bins")
    }

    func testHarmonicCQTHarmonics() async {
        let hcqt = HarmonicCQT(config: HarmonicCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 110,  // A2
            nBins: 36,
            nHarmonics: 4
        ))

        // Generate signal with harmonics (sawtooth-like)
        let fundamental: Float = 220  // A3
        var signal = [Float](repeating: 0, count: Int(22050 * 0.5))
        for h in 1...4 {
            let freq = fundamental * Float(h)
            let amplitude = 1.0 / Float(h)
            for i in 0..<signal.count {
                signal[i] += amplitude * sin(2.0 * Float.pi * freq * Float(i) / 22050)
            }
        }

        // transform returns [ComplexMatrix]
        let spectrum = await hcqt.transform(signal)

        // Each harmonic layer should show energy at different bins
        XCTAssertEqual(spectrum.count, 4)

        // All values should be finite - spectrum is [ComplexMatrix]
        for layer in spectrum {
            // layer is ComplexMatrix with .real and .imag arrays
            for row in layer.real {
                for val in row {
                    XCTAssertFalse(val.isNaN)
                    XCTAssertFalse(val.isInfinite)
                }
            }
            for row in layer.imag {
                for val in row {
                    XCTAssertFalse(val.isNaN)
                    XCTAssertFalse(val.isInfinite)
                }
            }
        }
    }

    // MARK: - VQT Tests

    func testVQTBasic() async {
        let vqt = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            gamma: 0  // Pure CQT behavior
        ))

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)

        let spectrum = await vqt.transform(signal)

        XCTAssertEqual(spectrum.real.count, 84)
        XCTAssertGreaterThan(spectrum.real[0].count, 0)
    }

    func testVQTWithGamma() async {
        // VQT with gamma should have better time resolution at low frequencies
        let vqtWithGamma = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            gamma: 20
        ))

        let signal = generateSineWave(frequency: 100, sampleRate: 22050, duration: 0.5)

        let magnitude = await vqtWithGamma.magnitude(signal)

        XCTAssertEqual(magnitude.count, 84)

        for row in magnitude {
            for val in row {
                XCTAssertFalse(val.isNaN)
                XCTAssertGreaterThanOrEqual(val, 0, "Magnitude should be non-negative")
            }
        }
    }

    func testVQTQFactors() {
        let vqtConfig = VQTConfig(
            sampleRate: 22050,
            fMin: 32.70,
            nBins: 84,
            gamma: 20
        )

        let vqt = VQT(config: vqtConfig)
        let qFactors = vqt.qFactors

        XCTAssertEqual(qFactors.count, 84)

        // Q should increase with frequency due to gamma
        XCTAssertLessThan(qFactors[0], qFactors[83],
                          "Q should increase with frequency when gamma > 0")
    }

    // MARK: - NMF Tests

    func testNMFBasic() throws {
        let nmf = NMF(config: NMFConfig(
            nComponents: 4,
            maxIterations: 50,
            tolerance: 1e-3
        ))

        // Create simple non-negative matrix
        var V: [[Float]] = []
        for i in 0..<32 {
            var row = [Float]()
            for j in 0..<20 {
                row.append(Float(i + j) / 50.0 + 0.1)
            }
            V.append(row)
        }

        let result = try nmf.decompose(V)

        // Check shapes
        XCTAssertEqual(result.W.count, 32, "W should have nFeatures rows")
        XCTAssertEqual(result.W[0].count, 4, "W should have nComponents columns")
        XCTAssertEqual(result.H.count, 4, "H should have nComponents rows")
        XCTAssertEqual(result.H[0].count, 20, "H should have nFrames columns")

        // Check non-negativity
        for row in result.W {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0, "W should be non-negative")
            }
        }
        for row in result.H {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0, "H should be non-negative")
            }
        }
    }

    func testNMFReconstruction() throws {
        let nmf = NMF(config: NMFConfig(
            nComponents: 8,
            maxIterations: 100,
            tolerance: 1e-4
        ))

        // Create random non-negative matrix
        var V: [[Float]] = []
        srand48(42)
        for _ in 0..<64 {
            var row = [Float]()
            for _ in 0..<50 {
                row.append(Float(drand48()) + 0.1)
            }
            V.append(row)
        }

        let result = try nmf.decompose(V)

        // Reconstruct
        let VHat = nmf.reconstruct(W: result.W, H: result.H)

        // Check reconstruction quality
        var totalError: Float = 0
        var totalValue: Float = 0
        for i in 0..<V.count {
            for j in 0..<V[0].count {
                let diff = V[i][j] - VHat[i][j]
                totalError += diff * diff
                totalValue += V[i][j] * V[i][j]
            }
        }

        let relativeError = sqrt(totalError) / sqrt(totalValue)
        XCTAssertLessThan(relativeError, 0.5, "Reconstruction error should be reasonable")
    }

    func testNMFSingleComponent() throws {
        let nmf = NMF(config: NMFConfig(nComponents: 4, maxIterations: 50))

        var V: [[Float]] = []
        for i in 0..<32 {
            var row = [Float]()
            for j in 0..<20 {
                row.append(Float(i + j) / 50.0 + 0.1)
            }
            V.append(row)
        }

        let result = try nmf.decompose(V)

        // Reconstruct single component
        let component0 = nmf.reconstruct(component: 0, W: result.W, H: result.H)

        XCTAssertEqual(component0.count, 32)
        XCTAssertEqual(component0[0].count, 20)
    }

    // MARK: - NN Filter Tests

    func testNNFilterBasic() {
        let nnFilter = NNFilter(config: NNFilterConfig(
            k: 5,
            width: 10,
            aggregation: .median
        ))

        // Create simple spectrogram with an outlier
        var features: [[Float]] = []
        for _ in 0..<32 {
            var row = [Float](repeating: 1.0, count: 20)
            row[10] = 10.0  // Outlier
            features.append(row)
        }

        let filtered = nnFilter.filter(features)

        XCTAssertEqual(filtered.count, 32)
        XCTAssertEqual(filtered[0].count, 20)

        // Outlier should be smoothed
        XCTAssertLessThan(filtered[0][10], 5.0, "Median filter should reduce outlier")
    }

    func testNNFilterSoftMask() {
        let nnFilter = NNFilter(config: NNFilterConfig(k: 5, width: 10))

        var features: [[Float]] = []
        for _ in 0..<32 {
            var row = [Float](repeating: 1.0, count: 20)
            row[10] = 10.0  // Outlier
            features.append(row)
        }

        let mask = nnFilter.softMask(features)

        XCTAssertEqual(mask.count, 32)

        // Mask values should be between 0 and 1
        for row in mask {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0)
                XCTAssertLessThanOrEqual(val, 1)
            }
        }

        // Outlier position should have lower mask value
        let avgMask = mask[0].reduce(0, +) / Float(mask[0].count)
        XCTAssertLessThan(mask[0][10], avgMask, "Outlier should have lower mask value")
    }

    func testNNFilterDecompose() {
        let nnFilter = NNFilter(config: NNFilterConfig(k: 5, width: 15))

        // Create spectrogram with both repeating and transient elements
        var features: [[Float]] = []
        for _ in 0..<32 {
            var row = [Float](repeating: 1.0, count: 30)
            // Add transient at frame 15
            row[15] = 5.0
            features.append(row)
        }

        let (repeating, nonRepeating) = nnFilter.decompose(features)

        XCTAssertEqual(repeating.count, 32)
        XCTAssertEqual(nonRepeating.count, 32)

        // Non-repeating should capture the transient
        let transientEnergy = nonRepeating[0][15]
        let avgNonRepeating = nonRepeating[0].reduce(0, +) / Float(nonRepeating[0].count)
        XCTAssertGreaterThan(transientEnergy, avgNonRepeating,
                             "Transient should appear in non-repeating component")
    }

    func testNNFilterMetrics() {
        // Test different distance metrics
        let euclidean = NNFilter(config: NNFilterConfig(k: 3, width: 5, metric: .euclidean))
        let cosine = NNFilter(config: NNFilterConfig(k: 3, width: 5, metric: .cosine))
        let manhattan = NNFilter(config: NNFilterConfig(k: 3, width: 5, metric: .manhattan))

        var features: [[Float]] = []
        for _ in 0..<16 {
            features.append((0..<10).map { _ in Float.random(in: 0...1) })
        }

        let resultEuc = euclidean.filter(features)
        let resultCos = cosine.filter(features)
        let resultMan = manhattan.filter(features)

        // All should produce valid output
        XCTAssertEqual(resultEuc.count, 16)
        XCTAssertEqual(resultCos.count, 16)
        XCTAssertEqual(resultMan.count, 16)
    }
}
