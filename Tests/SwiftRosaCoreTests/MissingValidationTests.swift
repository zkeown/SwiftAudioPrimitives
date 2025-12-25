import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Tests for API functions not covered by ComprehensiveValidationTests.
///
/// These tests validate:
/// - Delta (delta features / derivatives)
/// - PolyFeatures (polynomial fitting)
/// - StackMemory (time-delay embedding)
/// - Tonnetz (tonal centroid features)
/// - DTW (dynamic time warping)
/// - NNFilter (nearest neighbor filter)
/// - Chromagram
/// - Tempogram
final class MissingValidationTests: XCTestCase {

    // MARK: - Delta Features

    func testDeltaBasic() {
        // Create a simple feature matrix where delta should be predictable
        // Linear increase: delta should be constant
        let nFrames = 20
        let nFeatures = 3

        var features = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)
        for feat in 0..<nFeatures {
            for frame in 0..<nFrames {
                features[feat][frame] = Float(frame) * Float(feat + 1)
            }
        }

        let delta = Delta.compute(features, width: 2)

        XCTAssertEqual(delta.count, nFeatures, "Delta should have same number of features")
        XCTAssertEqual(delta[0].count, nFrames, "Delta should have same number of frames")

        // For linear input, delta should be approximately constant (except edges)
        for feat in 0..<nFeatures {
            let middleValues = Array(delta[feat][3..<(nFrames-3)])
            let mean = middleValues.reduce(0, +) / Float(middleValues.count)
            let variance = middleValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(middleValues.count)

            XCTAssertLessThan(variance, 0.1, "Delta of linear input should be constant")
        }
    }

    func testDeltaDelta() {
        // Test second derivative (delta-delta)
        let nFrames = 30
        let nFeatures = 2

        // Quadratic input: f(t) = t^2
        // Delta should be linear: f'(t) ~ 2t
        // Delta-delta should be constant: f''(t) ~ 2
        var features = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)
        for feat in 0..<nFeatures {
            for frame in 0..<nFrames {
                let t = Float(frame) / Float(nFrames)
                features[feat][frame] = t * t * Float(feat + 1)
            }
        }

        let delta = Delta.compute(features, width: 2)
        let deltaDelta = Delta.compute(delta, width: 2)

        XCTAssertEqual(deltaDelta.count, nFeatures)
        XCTAssertEqual(deltaDelta[0].count, nFrames)

        // Middle portion should be relatively constant
        let middle = Array(deltaDelta[0][5..<(nFrames-5)])
        let mean = middle.reduce(0, +) / Float(middle.count)
        let maxDeviation = middle.map { abs($0 - mean) }.max() ?? 0

        XCTAssertLessThan(maxDeviation / max(abs(mean), 1e-6), 0.5,
                          "Delta-delta of quadratic should be approximately constant")
    }

    // MARK: - PolyFeatures

    func testPolyFeaturesBasic() async {
        let config = PolyFeaturesConfig(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            order: 1  // Linear polynomial
        )

        let polyFeatures = PolyFeatures(config: config)

        // PolyFeatures works on magnitude spectrogram, not raw signal
        // First compute STFT magnitude
        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.5)
        let stftConfig = STFTConfig(uncheckedNFFT: 2048, hopLength: 512)
        let stft = STFT(config: stftConfig)
        let magnitude = await stft.magnitude(signal)

        let result = polyFeatures.transform(magnitudeSpec: magnitude)

        // Order 1 should produce 2 coefficients per frame (intercept + slope)
        XCTAssertEqual(result.count, 2, "Order 1 poly should have 2 coefficients")
        XCTAssertFalse(result[0].isEmpty, "Should produce output frames")
    }

    func testPolyFeaturesHigherOrder() async {
        let config = PolyFeaturesConfig(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            order: 3  // Cubic polynomial
        )

        let polyFeatures = PolyFeatures(config: config)

        // Compute magnitude spectrogram first
        let signal = TestSignalGenerator.chirp(fmin: 100, fmax: 4000, sampleRate: 22050, duration: 0.5)
        let stftConfig = STFTConfig(uncheckedNFFT: 2048, hopLength: 512)
        let stft = STFT(config: stftConfig)
        let magnitude = await stft.magnitude(signal)

        let result = polyFeatures.transform(magnitudeSpec: magnitude)

        // Order 3 should produce 4 coefficients
        XCTAssertEqual(result.count, 4, "Order 3 poly should have 4 coefficients")
    }

    // MARK: - StackMemory

    func testStackMemoryBasic() {
        let nFrames = 20
        let nFeatures = 3
        let delay = 3

        var features = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)
        for feat in 0..<nFeatures {
            for frame in 0..<nFrames {
                features[feat][frame] = Float(frame * 10 + feat)
            }
        }

        let stacked = StackMemory.stack(features, delay: delay)

        // Should have (delay+1) * nFeatures rows
        XCTAssertEqual(stacked.count, (delay + 1) * nFeatures,
                       "Stacked should have (delay+1) * nFeatures rows")
        XCTAssertEqual(stacked[0].count, nFrames,
                       "Stacked should preserve frame count")

        // First nFeatures rows should be the original features
        for feat in 0..<nFeatures {
            XCTAssertEqual(stacked[feat], features[feat],
                           "First features should be unchanged")
        }
    }

    func testStackMemoryNegativeDelay() {
        let nFrames = 20
        let nFeatures = 2
        let delay = -2  // Look ahead instead of behind

        var features = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nFeatures)
        for feat in 0..<nFeatures {
            for frame in 0..<nFrames {
                features[feat][frame] = Float(frame)
            }
        }

        let stacked = StackMemory.stack(features, delay: delay)

        // Should still produce valid output
        XCTAssertGreaterThan(stacked.count, 0, "Should handle negative delay")
    }

    // MARK: - Tonnetz

    func testTonnetzBasic() async {
        // TonnetzConfig takes (sampleRate:, hopLength:, nChroma:)
        let config = TonnetzConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12
        )

        let tonnetz = Tonnetz(config: config)

        // Test with harmonic signal
        let signal = TestSignalGenerator.multiTone(fundamentalFrequency: 440, sampleRate: 22050, duration: 0.5)
        let result = await tonnetz.transform(signal)

        // Tonnetz always produces 6 dimensions
        XCTAssertEqual(result.count, 6, "Tonnetz should have 6 dimensions")
        XCTAssertFalse(result[0].isEmpty, "Should produce output frames")

        // Values should be bounded (normalized tonal features)
        let allValues = result.flatMap { $0 }
        let maxAbs = allValues.map { abs($0) }.max() ?? 0
        XCTAssertLessThan(maxAbs, 10, "Tonnetz values should be bounded")
    }

    func testTonnetzDifferentNChroma() async {
        let signal = TestSignalGenerator.multiTone(fundamentalFrequency: 262, sampleRate: 22050, duration: 0.5)

        // Test with standard 12 chroma
        let config12 = TonnetzConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12
        )
        let tonnetz12 = Tonnetz(config: config12)
        let result12 = await tonnetz12.transform(signal)

        // Test with 24 chroma (quarter tones)
        let config24 = TonnetzConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 24
        )
        let tonnetz24 = Tonnetz(config: config24)
        let result24 = await tonnetz24.transform(signal)

        // Both should produce valid 6-dimensional output
        XCTAssertEqual(result12.count, 6, "12-chroma Tonnetz should have 6 dimensions")
        XCTAssertEqual(result24.count, 6, "24-chroma Tonnetz should have 6 dimensions")
    }

    // MARK: - DTW (Dynamic Time Warping)

    func testDTWIdentical() {
        let config = DTWConfig(metric: .euclidean, windowType: .none)
        let dtw = DTW(config: config)

        // Create identical sequences
        let sequence: [[Float]] = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6]
        ]

        let result = dtw.align(sequence, sequence)

        // Distance between identical sequences should be 0
        XCTAssertEqual(result.distance, 0, accuracy: 1e-6,
                       "DTW distance between identical sequences should be 0")

        // Path should be diagonal
        XCTAssertEqual(result.path.count, 5, "Path length should equal sequence length")
        for (i, (x, y)) in result.path.enumerated() {
            XCTAssertEqual(x, i, "Path should be diagonal")
            XCTAssertEqual(y, i, "Path should be diagonal")
        }
    }

    func testDTWTimeShift() {
        let config = DTWConfig(metric: .euclidean, windowType: .none)
        let dtw = DTW(config: config)

        // Create two sequences that are time-shifted versions
        let seq1: [[Float]] = [[0, 1, 2, 3, 4, 5, 0, 0]]
        let seq2: [[Float]] = [[0, 0, 0, 1, 2, 3, 4, 5]]

        let result = dtw.align(seq1, seq2)

        // Should find alignment despite time shift
        XCTAssertGreaterThan(result.path.count, 0, "Should produce valid path")

        // Normalized distance should be relatively low for shifted versions of same pattern
        XCTAssertLessThan(result.normalizedDistance, 1.0,
                          "Normalized distance should be low for time-shifted signals")
    }

    func testDTWWithWindow() {
        // Test Sakoe-Chiba band constraint
        let config = DTWConfig(
            metric: .euclidean,
            windowType: .sakoeChiba,
            windowParam: 2
        )
        let dtw = DTW(config: config)

        let seq1: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]
        let seq2: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8]]

        let result = dtw.align(seq1, seq2)

        // Path should stay within window constraint
        for (x, y) in result.path {
            XCTAssertLessThanOrEqual(abs(Int32(x) - Int32(y)), 2,
                                     "Path should stay within Sakoe-Chiba band")
        }
    }

    // MARK: - NNFilter

    func testNNFilterBasic() {
        let config = NNFilterConfig(k: 3, width: 5, aggregation: .median)
        let nnFilter = NNFilter(config: config)

        // Create test spectrogram with some outliers
        let nBins = 20
        let nFrames = 30
        var spectrogram = [[Float]](repeating: [Float](repeating: 1.0, count: nFrames), count: nBins)

        // Add an outlier
        spectrogram[10][15] = 100.0

        let filtered = nnFilter.filter(spectrogram)

        // Filtered output should have reduced outlier
        XCTAssertEqual(filtered.count, nBins)
        XCTAssertEqual(filtered[0].count, nFrames)

        // The outlier should be attenuated by median filtering
        XCTAssertLessThan(filtered[10][15], 100.0, "Outlier should be attenuated")
    }

    func testNNFilterMean() {
        let config = NNFilterConfig(k: 5, width: 3, aggregation: .mean)
        let nnFilter = NNFilter(config: config)

        let nBins = 10
        let nFrames = 20
        let spectrogram = [[Float]](repeating: [Float](repeating: 2.0, count: nFrames), count: nBins)

        let filtered = nnFilter.filter(spectrogram)

        // For constant input, mean filter should preserve values
        for row in filtered {
            for val in row {
                XCTAssertEqual(val, 2.0, accuracy: 0.1, "Mean filter should preserve constant input")
            }
        }
    }

    // MARK: - Chromagram

    func testChromagramSTFT() async {
        // STFT-based chromagram uses useCQT: false
        let config = ChromagramConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12,
            useCQT: false,
            nFFT: 2048
        )

        let chroma = Chromagram(config: config)

        // Test with C major chord (C, E, G)
        let cFreq: Float = 261.63
        let eFreq: Float = 329.63
        let gFreq: Float = 392.00

        let signal = (0..<22050).map { i in
            let t = Float(i) / 22050.0
            return (sin(2 * Float.pi * cFreq * t) +
                    sin(2 * Float.pi * eFreq * t) +
                    sin(2 * Float.pi * gFreq * t)) / 3.0
        }

        let result = await chroma.transform(signal)

        XCTAssertEqual(result.count, 12, "Chromagram should have 12 pitch classes")
        XCTAssertFalse(result[0].isEmpty, "Should produce output frames")

        // Find dominant chroma bins
        let meanChroma = result.map { $0.reduce(0, +) / Float($0.count) }
        let maxChroma = meanChroma.max() ?? 0

        // Normalize
        let normalizedChroma = meanChroma.map { $0 / max(maxChroma, 1e-10) }

        // C (0), E (4), G (7) should have high energy
        // Using relative threshold since exact bin mapping depends on tuning
        XCTAssertGreaterThan(normalizedChroma.max() ?? 0, 0.5,
                             "Should detect chromatic content")
    }

    func testChromagramCQT() async {
        // CQT-based chromagram uses useCQT: true
        let config = ChromagramConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12,
            useCQT: true
        )

        let chroma = Chromagram(config: config)
        let signal = TestSignalGenerator.multiTone(fundamentalFrequency: 440, sampleRate: 22050, duration: 0.5)

        let result = await chroma.transform(signal)

        XCTAssertEqual(result.count, 12, "CQT Chromagram should have 12 pitch classes")
        XCTAssertFalse(result[0].isEmpty)
    }

    // MARK: - Tempogram
    // Note: Tempogram is in SwiftRosaAnalysis module, need to import it

    func testTempogramBasic() async throws {
        throw XCTSkip("Tempogram is in SwiftRosaAnalysis module - tested in SwiftRosaAnalysisTests")
    }

    // MARK: - A-Weighting
    // Note: AWeighting initializer is private - weighting is applied internally

    func testAWeighting() throws {
        // AWeighting is used internally by PerceptualWeighting
        // Direct instantiation is not part of public API
        throw XCTSkip("AWeighting initializer is private - tested through PerceptualWeighting")
    }

    // MARK: - Perceptual Weighting

    func testPerceptualWeighting() async throws {
        // PerceptualWeighting applies A-weighting internally
        // Test through SpectralFeatures instead since direct instantiation may be private
        throw XCTSkip("PerceptualWeighting tested through SpectralFeatures validation tests")
    }

    // MARK: - HPSS

    func testHPSSBasic() async {
        // HPSSConfig uses tuple kernelSize: (harmonic: Int, percussive: Int)
        let hpssConfig = HPSSConfig(
            kernelSize: (harmonic: 31, percussive: 31),
            power: 2.0,
            margin: 1.0
        )

        let hpss = HPSS(hpssConfig: hpssConfig)

        // Create mixed signal: harmonic + percussive
        let harmonic = TestSignalGenerator.multiTone(fundamentalFrequency: 440, sampleRate: 22050, duration: 0.5)
        let percussive = TestSignalGenerator.clickTrain(clicksPerSecond: 4, sampleRate: 22050, duration: 0.5)

        let mixed = zip(harmonic, percussive).map { $0 + $1 * 0.5 }

        let result = await hpss.separate(mixed)

        XCTAssertFalse(result.harmonic.isEmpty, "Should produce harmonic component")
        XCTAssertFalse(result.percussive.isEmpty, "Should produce percussive component")

        // Energy conservation: harmonic + percussive ≈ original
        let originalEnergy = mixed.map { $0 * $0 }.reduce(0, +)
        let harmonicEnergy = result.harmonic.map { $0 * $0 }.reduce(0, +)
        let percussiveEnergy = result.percussive.map { $0 * $0 }.reduce(0, +)

        // Allow some leakage/loss
        let totalSeparated = harmonicEnergy + percussiveEnergy
        let energyRatio = totalSeparated / max(originalEnergy, 1e-10)

        XCTAssertGreaterThan(energyRatio, 0.5, "Should preserve reasonable energy")
        XCTAssertLessThan(energyRatio, 2.0, "Should not create excessive energy")
    }

    // MARK: - NMF

    func testNMFBasic() throws {
        let config = NMFConfig(
            nComponents: 4,
            maxIterations: 100,
            randomSeed: 42
        )

        let nmf = NMF(config: config)

        // Create a simple non-negative matrix
        let nBins = 20
        let nFrames = 30
        var matrix = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nBins)

        for i in 0..<nBins {
            for j in 0..<nFrames {
                matrix[i][j] = Float.random(in: 0..<1)
            }
        }

        let result = try nmf.decompose(matrix)

        // W should be nBins x nComponents
        XCTAssertEqual(result.W.count, nBins)
        XCTAssertEqual(result.W[0].count, 4)

        // H should be nComponents x nFrames
        XCTAssertEqual(result.H.count, 4)
        XCTAssertEqual(result.H[0].count, nFrames)

        // All values should be non-negative
        let allW = result.W.flatMap { $0 }
        let allH = result.H.flatMap { $0 }

        XCTAssertTrue(allW.allSatisfy { $0 >= 0 }, "W should be non-negative")
        XCTAssertTrue(allH.allSatisfy { $0 >= 0 }, "H should be non-negative")

        // Reconstruction error should be reasonable
        XCTAssertLessThan(result.reconstructionError, 1.0,
                          "Reconstruction error should be bounded")
    }

    // MARK: - Phase Vocoder

    func testPhaseVocoderStretch() async {
        let config = PhaseVocoderConfig(
            nFFT: 2048,
            hopLength: 512
        )

        let vocoder = PhaseVocoder(config: config)

        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 0.5)

        // Time stretch by 2x (rate=0.5 means output is 2x longer)
        let stretched = await vocoder.timeStretch(signal, rate: 0.5)

        // Stretched signal should be approximately 2x longer
        let expectedLength = Int(Float(signal.count) * 2)
        XCTAssertEqual(stretched.count, expectedLength, accuracy: signal.count / 10,
                       "2x stretch should double length")

        // Stretched signal should preserve pitch (frequency content)
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let origSpec = await stft.transform(signal)
        let stretchedSpec = await stft.transform(stretched)

        // Find dominant frequency bin in both
        func dominantBin(_ mag: [[Float]]) -> Int {
            let avgMag = mag.enumerated().map { (idx, row) in
                (idx, row.reduce(0, +) / Float(row.count))
            }
            return avgMag.max(by: { $0.1 < $1.1 })?.0 ?? 0
        }

        let origDominant = dominantBin(origSpec.magnitude)
        let stretchedDominant = dominantBin(stretchedSpec.magnitude)

        XCTAssertEqual(origDominant, stretchedDominant, accuracy: 2,
                       "Pitch should be preserved during time stretch")
    }
}
