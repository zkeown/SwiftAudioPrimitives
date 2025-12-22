import XCTest
import Foundation
@testable import SwiftRosaCore
@testable import SwiftRosaEffects
@testable import SwiftRosaAnalysis

/// Validation tests using real audio from MUSDB18-HQ dataset.
///
/// These tests compare SwiftRosa output against librosa reference data
/// generated from actual music recordings, providing more realistic
/// validation than synthetic signals alone.
///
/// ## Setup
///
/// Before running these tests, generate reference data:
/// ```bash
/// # Install requirements
/// pip install librosa musdb numpy soundfile
///
/// # Generate reference data (requires MUSDB18-HQ dataset)
/// python scripts/generate_real_audio_reference.py --musdb-path /path/to/musdb18hq
/// ```
///
/// See Tests/RealAudioValidation/README.md for detailed setup instructions.
final class RealAudioValidationTests: XCTestCase {

    // MARK: - Reference Data

    private static var referenceData: [String: Any]?
    private static var referenceLoadError: Error?

    override class func setUp() {
        super.setUp()
        loadReferenceData()
    }

    private static func loadReferenceData() {
        let bundle = Bundle(for: RealAudioValidationTests.self)

        // Try multiple locations
        let possiblePaths = [
            bundle.path(forResource: "musdb18_reference", ofType: "json", inDirectory: "ReferenceData"),
            bundle.resourcePath.map { "\($0)/ReferenceData/musdb18_reference.json" },
            "Tests/RealAudioValidation/ReferenceData/musdb18_reference.json",
        ].compactMap { $0 }

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                do {
                    let data = try Data(contentsOf: URL(fileURLWithPath: path))
                    referenceData = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    print("Loaded real audio reference data from: \(path)")
                    return
                } catch {
                    referenceLoadError = error
                }
            }
        }

        referenceLoadError = NSError(
            domain: "RealAudioValidation",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Reference data not found. Run generate_real_audio_reference.py first."]
        )
    }

    private func getReference() throws -> [String: Any] {
        if let error = Self.referenceLoadError {
            throw XCTSkip("Reference data not available: \(error.localizedDescription)")
        }
        guard let ref = Self.referenceData else {
            throw XCTSkip("Reference data not loaded")
        }
        return ref
    }

    private func getTracks() throws -> [String: Any] {
        let ref = try getReference()
        guard let tracks = ref["tracks"] as? [String: Any] else {
            throw XCTSkip("No tracks in reference data")
        }
        return tracks
    }

    private func loadAudio(_ trackId: String) throws -> [Float] {
        let tracks = try getTracks()
        guard let track = tracks[trackId] as? [String: Any],
              let audioData = track["audio"] as? [Double] else {
            throw XCTSkip("Audio not found for track: \(trackId)")
        }
        return audioData.map { Float($0) }
    }

    private func getFeatures(_ trackId: String) throws -> [String: Any] {
        let tracks = try getTracks()
        guard let track = tracks[trackId] as? [String: Any],
              let features = track["features"] as? [String: Any] else {
            throw XCTSkip("Features not found for track: \(trackId)")
        }
        return features
    }

    // MARK: - Tolerance Definitions for Real Audio

    /// Tolerances may be slightly higher for real audio due to
    /// algorithmic edge cases not present in synthetic signals
    struct RealAudioTolerances {
        static let stftMagnitude: Double = 1e-4  // Slightly higher than synthetic
        static let melSpectrogram: Double = 5e-3
        static let mfcc: Double = 1e-3
        static let mfccCorrelation: Float = 0.98
        static let cqtMagnitude: Double = 0.05  // Higher due to algorithmic differences
        static let cqtCorrelation: Float = 0.90
        static let chroma: Double = 0.05
        static let chromaCorrelation: Float = 0.90
        static let spectralCentroid: Double = 1e-4
        static let spectralBandwidth: Double = 0.20
        static let spectralRolloff: Double = 1e-4
        static let spectralFlatness: Double = 1e-4
        static let rms: Double = 1e-4
        static let zcr: Double = 1e-4
        static let onsetFrameTolerance: Int = 5
        static let tempoTolerance: Double = 0.10  // 10% tempo tolerance
    }

    // MARK: - Validation Helpers

    /// Compute Pearson correlation coefficient between two arrays
    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, a.count > 1 else { return 0 }

        let n = Float(a.count)
        let meanA = a.reduce(0, +) / n
        let meanB = b.reduce(0, +) / n

        var varA: Float = 0
        var varB: Float = 0
        var cov: Float = 0

        for (va, vb) in zip(a, b) {
            let diffA = va - meanA
            let diffB = vb - meanB
            varA += diffA * diffA
            varB += diffB * diffB
            cov += diffA * diffB
        }

        let stdA = sqrt(varA / n)
        let stdB = sqrt(varB / n)

        guard stdA > 1e-10 && stdB > 1e-10 else { return 0 }
        return cov / (n * stdA * stdB)
    }

    // MARK: - Summary Tracking

    private static var testResults: [(feature: String, track: String, passed: Bool, error: Double)] = []

    private func recordResult(_ feature: String, track: String, passed: Bool, error: Double) {
        Self.testResults.append((feature, track, passed, error))
    }

    override class func tearDown() {
        super.tearDown()
        printSummary()
    }

    private static func printSummary() {
        guard !testResults.isEmpty else { return }

        print("\n" + String(repeating: "=", count: 70))
        print("REAL AUDIO VALIDATION SUMMARY")
        print(String(repeating: "=", count: 70))

        // Group by feature
        var featureResults: [String: [(track: String, passed: Bool, error: Double)]] = [:]
        for result in testResults {
            featureResults[result.feature, default: []].append((result.track, result.passed, result.error))
        }

        var totalPassed = 0
        var totalTests = 0

        for (feature, results) in featureResults.sorted(by: { $0.key < $1.key }) {
            let passed = results.filter { $0.passed }.count
            let total = results.count
            totalPassed += passed
            totalTests += total

            let status = passed == total ? "PASS" : "WARN"
            let avgError = results.map { $0.error }.reduce(0, +) / Double(total)

            print("\(status) \(feature): \(passed)/\(total) (avg error: \(String(format: "%.2e", avgError)))")
        }

        print(String(repeating: "-", count: 70))
        print("Overall: \(totalPassed)/\(totalTests) tests passed (\(String(format: "%.1f", Double(totalPassed)/Double(totalTests)*100))%)")
        print(String(repeating: "=", count: 70))
    }

    // MARK: - Test Cases

    func testSTFTMagnitude_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let stftRef = features["stft"] as? [String: Any],
                  let expectedMag = stftRef["magnitude"] as? [[Double]],
                  let nFFT = stftRef["n_fft"] as? Int,
                  let hopLength = stftRef["hop_length"] as? Int else {
                continue
            }

            let stft = STFT(config: STFTConfig(nFFT: nFFT, hopLength: hopLength))
            let result = await stft.magnitude(audio)

            // Compare shapes
            let expectedRows = expectedMag.count
            let expectedCols = expectedMag.first?.count ?? 0

            XCTAssertEqual(result.count, expectedRows, "STFT row count mismatch for \(trackId)")
            if let firstRow = result.first {
                XCTAssertEqual(firstRow.count, expectedCols, "STFT column count mismatch for \(trackId)")
            }

            // Use correlation-based comparison per frequency bin
            // Both librosa and SwiftRosa use shape (nFreqs, nFrames)
            var totalCorrelation: Float = 0
            var binCount = 0

            let binsToCompare = min(expectedMag.count, result.count)
            let framesToCompare = min(expectedMag.first?.count ?? 0, result.first?.count ?? 0)

            for binIdx in 0..<binsToCompare {
                var expectedBin = [Float]()
                var actualBin = [Float]()

                for frameIdx in 0..<framesToCompare {
                    expectedBin.append(Float(expectedMag[binIdx][frameIdx]))
                    actualBin.append(result[binIdx][frameIdx])
                }

                let corr = correlation(expectedBin, actualBin)
                if !corr.isNaN {
                    totalCorrelation += corr
                    binCount += 1
                }
            }

            let avgCorrelation = binCount > 0 ? totalCorrelation / Float(binCount) : 0
            let passed = avgCorrelation >= 0.999  // Very high correlation expected for STFT
            recordResult("stft", track: trackId, passed: passed, error: Double(1 - avgCorrelation))

            XCTAssertGreaterThanOrEqual(avgCorrelation, 0.999,
                "STFT magnitude correlation \(avgCorrelation) too low for \(trackId)")
        }
    }

    func testMelSpectrogram_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let melRef = features["mel_spectrogram"] as? [String: Any],
                  let expectedMel = melRef["values"] as? [[Double]],
                  let nMels = melRef["n_mels"] as? Int else {
                continue
            }

            let mel = MelSpectrogram(
                sampleRate: 22050,
                nFFT: 2048,
                hopLength: 512,
                nMels: nMels
            )
            let result = await mel.transform(audio)

            // Use correlation-based comparison
            // Both librosa and SwiftRosa use shape (nBins, nFrames)
            var totalCorrelation: Float = 0
            var binCount = 0

            let binsToCompare = min(expectedMel.count, result.count)
            let framesToCompare = min(expectedMel.first?.count ?? 0, result.first?.count ?? 0)

            for binIdx in 0..<binsToCompare {
                var expectedBin = [Float]()
                var actualBin = [Float]()

                for frameIdx in 0..<framesToCompare {
                    expectedBin.append(Float(expectedMel[binIdx][frameIdx]))
                    actualBin.append(result[binIdx][frameIdx])
                }

                let corr = correlation(expectedBin, actualBin)
                if !corr.isNaN {
                    totalCorrelation += corr
                    binCount += 1
                }
            }

            let avgCorrelation = binCount > 0 ? totalCorrelation / Float(binCount) : 0
            let passed = avgCorrelation >= 0.95
            recordResult("mel_spectrogram", track: trackId, passed: passed, error: Double(1 - avgCorrelation))

            XCTAssertGreaterThanOrEqual(avgCorrelation, 0.95,
                "Mel spectrogram correlation \(avgCorrelation) too low for \(trackId)")
        }
    }

    func testMFCC_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let mfccRef = features["mfcc"] as? [String: Any],
                  let expectedMFCC = mfccRef["values"] as? [[Double]],
                  let nMFCC = mfccRef["n_mfcc"] as? Int else {
                continue
            }

            let mfcc = MFCC(
                sampleRate: 22050,
                nFFT: 2048,
                hopLength: 512,
                nMFCC: nMFCC
            )
            let result = await mfcc.transform(audio)

            // Correlation-based comparison
            var totalCorrelation: Float = 0
            var coeffCount = 0

            for coeffIdx in 0..<min(result.count, expectedMFCC.count) {
                let expected = expectedMFCC[coeffIdx].map { Float($0) }
                let actual = result[coeffIdx]

                let corr = correlation(expected, actual)
                if !corr.isNaN {
                    totalCorrelation += corr
                    coeffCount += 1
                }
            }

            let avgCorrelation = coeffCount > 0 ? totalCorrelation / Float(coeffCount) : 0
            let passed = avgCorrelation >= RealAudioTolerances.mfccCorrelation
            recordResult("mfcc", track: trackId, passed: passed, error: Double(1 - avgCorrelation))

            XCTAssertGreaterThanOrEqual(avgCorrelation, RealAudioTolerances.mfccCorrelation,
                "MFCC correlation \(avgCorrelation) too low for \(trackId)")
        }
    }

    func testCQT_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let cqtRef = features["cqt"] as? [String: Any],
                  let expectedMag = cqtRef["magnitude"] as? [[Double]],
                  let fmin = cqtRef["fmin"] as? Double,
                  let nBins = cqtRef["n_bins"] as? Int,
                  let binsPerOctave = cqtRef["bins_per_octave"] as? Int else {
                continue
            }

            let cqt = CQT(config: CQTConfig(
                sampleRate: 22050,
                hopLength: 512,
                fMin: Float(fmin),
                nBins: nBins,
                binsPerOctave: binsPerOctave
            ))
            let result = await cqt.magnitude(audio)

            // Correlation-based comparison
            // Both librosa and SwiftRosa use shape (nBins, nFrames)
            var totalCorrelation: Float = 0
            var binCount = 0

            let binsToCompare = min(expectedMag.count, result.count)
            let framesToCompare = min(expectedMag.first?.count ?? 0, result.first?.count ?? 0)

            for binIdx in 0..<binsToCompare {
                var expectedBin = [Float]()
                var actualBin = [Float]()

                for frameIdx in 0..<framesToCompare {
                    expectedBin.append(Float(expectedMag[binIdx][frameIdx]))
                    actualBin.append(result[binIdx][frameIdx])
                }

                let corr = correlation(expectedBin, actualBin)
                if !corr.isNaN {
                    totalCorrelation += corr
                    binCount += 1
                }
            }

            let avgCorrelation = binCount > 0 ? totalCorrelation / Float(binCount) : 0
            let passed = avgCorrelation >= RealAudioTolerances.cqtCorrelation
            recordResult("cqt", track: trackId, passed: passed, error: Double(1 - avgCorrelation))

            XCTAssertGreaterThanOrEqual(avgCorrelation, RealAudioTolerances.cqtCorrelation,
                "CQT correlation \(avgCorrelation) too low for \(trackId)")
        }
    }

    func testSpectralCentroid_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let centroidRef = features["spectral_centroid"] as? [String: Any],
                  let expectedCentroid = centroidRef["values"] as? [Double] else {
                continue
            }

            let sf = SpectralFeatures(config: SpectralFeaturesConfig(
                sampleRate: 22050,
                nFFT: 2048,
                hopLength: 512
            ))
            let result = await sf.centroid(audio)

            var maxRelError: Double = 0
            // Use minimum centroid of 100 Hz for relative error calculation
            // Below this, we're in noise floor territory where relative error is meaningless
            let minCentroidForRelError: Double = 100.0
            for i in 0..<min(result.count, expectedCentroid.count) {
                if expectedCentroid[i] > minCentroidForRelError {
                    let relError = abs(Double(result[i]) - expectedCentroid[i]) / expectedCentroid[i]
                    maxRelError = max(maxRelError, relError)
                }
            }

            let passed = maxRelError <= RealAudioTolerances.spectralCentroid
            recordResult("spectral_centroid", track: trackId, passed: passed, error: maxRelError)

            XCTAssertLessThanOrEqual(maxRelError, RealAudioTolerances.spectralCentroid,
                "Spectral centroid error \(maxRelError) exceeds tolerance for \(trackId)")
        }
    }

    func testSpectralBandwidth_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let bwRef = features["spectral_bandwidth"] as? [String: Any],
                  let expectedBW = bwRef["values"] as? [Double] else {
                continue
            }

            let sf = SpectralFeatures(config: SpectralFeaturesConfig(
                sampleRate: 22050,
                nFFT: 2048,
                hopLength: 512
            ))
            let result = await sf.bandwidth(audio)

            var maxRelError: Double = 0
            // Use minimum bandwidth of 100 Hz for relative error calculation
            // Below this, we're in noise floor territory where relative error is meaningless
            let minBWForRelError: Double = 100.0
            for i in 0..<min(result.count, expectedBW.count) {
                if expectedBW[i] > minBWForRelError {
                    let relError = abs(Double(result[i]) - expectedBW[i]) / expectedBW[i]
                    maxRelError = max(maxRelError, relError)
                }
            }

            let passed = maxRelError <= RealAudioTolerances.spectralBandwidth
            recordResult("spectral_bandwidth", track: trackId, passed: passed, error: maxRelError)

            XCTAssertLessThanOrEqual(maxRelError, RealAudioTolerances.spectralBandwidth,
                "Spectral bandwidth error \(maxRelError) exceeds tolerance for \(trackId)")
        }
    }

    func testRMS_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let rmsRef = features["rms"] as? [String: Any],
                  let expectedRMS = rmsRef["values"] as? [Double] else {
                continue
            }

            let rms = RMSEnergy(config: RMSEnergyConfig(frameLength: 2048, hopLength: 512))
            let result = rms.transform(audio)

            var maxRelError: Double = 0
            for i in 0..<min(result.count, expectedRMS.count) {
                if expectedRMS[i] > 1e-6 {
                    let relError = abs(Double(result[i]) - expectedRMS[i]) / expectedRMS[i]
                    maxRelError = max(maxRelError, relError)
                }
            }

            let passed = maxRelError <= RealAudioTolerances.rms
            recordResult("rms", track: trackId, passed: passed, error: maxRelError)

            XCTAssertLessThanOrEqual(maxRelError, RealAudioTolerances.rms,
                "RMS error \(maxRelError) exceeds tolerance for \(trackId)")
        }
    }

    func testZeroCrossingRate_RealAudio() async throws {
        let tracks = try getTracks()

        for (trackId, _) in tracks {
            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let zcrRef = features["zero_crossing_rate"] as? [String: Any],
                  let expectedZCR = zcrRef["values"] as? [Double] else {
                continue
            }

            let zcr = ZeroCrossingRate(frameLength: 2048, hopLength: 512)
            let result = zcr.transform(audio)

            var maxRelError: Double = 0
            for i in 0..<min(result.count, expectedZCR.count) {
                if expectedZCR[i] > 1e-6 {
                    let relError = abs(Double(result[i]) - expectedZCR[i]) / expectedZCR[i]
                    maxRelError = max(maxRelError, relError)
                }
            }

            let passed = maxRelError <= RealAudioTolerances.zcr
            recordResult("zcr", track: trackId, passed: passed, error: maxRelError)

            XCTAssertLessThanOrEqual(maxRelError, RealAudioTolerances.zcr,
                "ZCR error \(maxRelError) exceeds tolerance for \(trackId)")
        }
    }

    func testOnsetDetection_RealAudio() async throws {
        _ = try getTracks()  // Verify we have data

        // Only test on tracks with clear onsets
        let onsetTracks = ["rock_drums", "drums_isolated", "electronic_rock"]

        for trackId in onsetTracks {
            guard let _ = (try? getTracks())?[trackId] else { continue }

            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let onsetRef = features["onset_detect"] as? [String: Any],
                  let expectedFrames = onsetRef["frames"] as? [Int] else {
                continue
            }

            // Use librosa-compatible method with lower threshold to match librosa defaults
            let detector = OnsetDetector(config: OnsetConfig(
                sampleRate: 22050,
                hopLength: 512,
                method: .librosa,
                peakThreshold: 0.1  // Lower threshold matches librosa's sensitivity
            ))
            let detectedFrames = await detector.detect(audio)

            // Calculate F1 score with frame tolerance
            let tolerance = RealAudioTolerances.onsetFrameTolerance
            var truePositives = 0

            for detected in detectedFrames {
                if expectedFrames.contains(where: { abs($0 - detected) <= tolerance }) {
                    truePositives += 1
                }
            }

            let precision = detectedFrames.isEmpty ? 0 : Float(truePositives) / Float(detectedFrames.count)
            let recall = expectedFrames.isEmpty ? 0 : Float(truePositives) / Float(expectedFrames.count)
            let f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0

            let passed = f1 >= 0.5  // More lenient for real audio
            recordResult("onset_detect", track: trackId, passed: passed, error: Double(1 - f1))

            XCTAssertGreaterThanOrEqual(f1, 0.5,
                "Onset detection F1=\(f1) too low for \(trackId)")
        }
    }

    func testTempo_RealAudio() async throws {
        _ = try getTracks()  // Verify we have data

        // Only test on tracks with clear tempo
        let tempoTracks = ["rock_drums", "electronic_rock", "electronic_bass", "pop_vocals"]

        for trackId in tempoTracks {
            guard let _ = (try? getTracks())?[trackId] else { continue }

            let audio = try loadAudio(trackId)
            let features = try getFeatures(trackId)

            guard let tempoRef = features["tempo"] as? [String: Any],
                  let expectedTempo = tempoRef["bpm"] as? Double else {
                continue
            }

            let tracker = BeatTracker(config: BeatTrackerConfig(
                sampleRate: 22050,
                hopLength: 512
            ))
            let detectedTempo = Double(await tracker.estimateTempo(audio))

            // Allow for octave errors (double/half tempo)
            let relError = min(
                abs(detectedTempo - expectedTempo) / expectedTempo,
                abs(detectedTempo - expectedTempo * 2) / (expectedTempo * 2),
                abs(detectedTempo - expectedTempo / 2) / (expectedTempo / 2)
            )

            let passed = relError <= RealAudioTolerances.tempoTolerance
            recordResult("tempo", track: trackId, passed: passed, error: relError)

            XCTAssertLessThanOrEqual(relError, RealAudioTolerances.tempoTolerance,
                "Tempo error \(relError) exceeds tolerance for \(trackId) (expected: \(expectedTempo), got: \(detectedTempo))")
        }
    }
}
