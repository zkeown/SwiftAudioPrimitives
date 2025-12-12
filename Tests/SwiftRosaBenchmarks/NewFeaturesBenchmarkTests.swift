import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects
@testable import SwiftRosaAnalysis

/// Performance benchmarks for newly implemented features.
final class NewFeaturesBenchmarkTests: XCTestCase {

    // MARK: - Test Data Generation

    private func generateTestSignal(duration: Float, sampleRate: Float = 22050) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        srand48(42)
        return (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            // Mix of tones and noise
            return 0.5 * sin(2.0 * Float.pi * 440 * t) +
                   0.3 * sin(2.0 * Float.pi * 880 * t) +
                   0.2 * Float(drand48() * 2 - 1)
        }
    }

    private func generateTestSpectrogram(nFreqs: Int, nFrames: Int) -> [[Float]] {
        srand48(42)
        return (0..<nFreqs).map { _ in
            (0..<nFrames).map { _ in Float(drand48()) + 0.1 }
        }
    }

    // MARK: - Feature Extraction Benchmarks

    func testRMSEnergyPerformance() {
        let rms = RMSEnergy()
        let signal = generateTestSignal(duration: 10.0)

        measure {
            _ = rms.transform(signal)
        }
    }

    func testDeltaPerformance() {
        let features = generateTestSpectrogram(nFreqs: 40, nFrames: 1000)

        measure {
            _ = Delta.compute(features, width: 9, order: 1)
        }
    }

    func testAWeightingPerformance() {
        // Create a power spectrogram
        let powerSpec = generateTestSpectrogram(nFreqs: 1025, nFrames: 500)

        // Measure applying A-weighting to spectrogram
        measure {
            _ = AWeighting.apply(to: powerSpec, nFFT: 2048, sampleRate: 22050)
        }
    }

    func testPCENPerformance() {
        let pcen = PCEN()
        let spectrogram = generateTestSpectrogram(nFreqs: 128, nFrames: 500)

        // PCEN is synchronous
        measure {
            _ = pcen.transform(spectrogram)
        }
    }

    func testPolyFeaturesPerformance() {
        let poly = PolyFeatures(config: PolyFeaturesConfig(order: 2))
        let spectrogram = generateTestSpectrogram(nFreqs: 1025, nFrames: 500)

        measure {
            _ = poly.transform(magnitudeSpec: spectrogram)
        }
    }

    func testStackMemoryPerformance() {
        let features = generateTestSpectrogram(nFreqs: 40, nFrames: 1000)

        measure {
            _ = StackMemory.stack(features, nSteps: 4, delay: 2)
        }
    }

    func testSyncFeaturesPerformance() {
        let features = generateTestSpectrogram(nFreqs: 128, nFrames: 1000)
        let boundaries = Array(stride(from: 0, to: 1000, by: 50))

        measure {
            _ = SyncFeatures.sync(features, boundaries: boundaries, aggregation: .mean)
        }
    }

    // MARK: - Generator Benchmarks

    func testToneGeneratorPerformance() {
        measure {
            _ = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 10.0)
        }
    }

    func testChirpGeneratorPerformance() {
        measure {
            _ = ChirpGenerator.chirp(fMin: 100, fMax: 10000, sampleRate: 22050, duration: 10.0, type: .exponential)
        }
    }

    func testClicksGeneratorPerformance() {
        let times = (0..<1000).map { Float($0) * 0.01 }

        measure {
            _ = ClickGenerator.clicks(times: times, sampleRate: 22050, length: 220500)
        }
    }

    // MARK: - Transform Benchmarks

    func testTonnetzPerformance() async {
        let tonnetz = Tonnetz()
        let signal = generateTestSignal(duration: 5.0)

        // Warm up
        _ = await tonnetz.transform(signal)

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = await tonnetz.transform(signal)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Tonnetz 5s: \(elapsed / 3 * 1000) ms per run")
    }

    func testPseudoCQTPerformance() async {
        let pcqt = PseudoCQT(config: PseudoCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 84
        ))
        let signal = generateTestSignal(duration: 5.0)

        // Warm up
        _ = await pcqt.magnitude(signal)

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = await pcqt.magnitude(signal)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Pseudo-CQT 5s (84 bins): \(elapsed / 3 * 1000) ms per run")
    }

    func testVQTPerformance() async {
        let vqt = VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 84,
            gamma: 20
        ))
        let signal = generateTestSignal(duration: 5.0)

        // Warm up
        _ = await vqt.magnitude(signal)

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = await vqt.magnitude(signal)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("VQT 5s (84 bins, gamma=20): \(elapsed / 3 * 1000) ms per run")
    }

    func testHarmonicCQTPerformance() async {
        let hcqt = HarmonicCQT(config: HarmonicCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 60,
            nHarmonics: 4
        ))
        let signal = generateTestSignal(duration: 5.0)

        // Warm up
        _ = await hcqt.transform(signal)

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = await hcqt.transform(signal)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Harmonic CQT 5s (60 bins, 4 harmonics): \(elapsed / 3 * 1000) ms per run")
    }

    func testTempogramPerformance() async {
        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 384,
            type: .autocorrelation
        ))
        let signal = generateTestSignal(duration: 10.0)

        // Warm up
        _ = await tempogram.transform(signal)

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = await tempogram.transform(signal)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("Tempogram 10s (autocorrelation): \(elapsed / 3 * 1000) ms per run")
    }

    // MARK: - Decomposition Benchmarks

    func testNMFPerformance() throws {
        let nmf = NMF(config: NMFConfig(
            nComponents: 8,
            maxIterations: 50,
            tolerance: 1e-3
        ))

        let V = generateTestSpectrogram(nFreqs: 128, nFrames: 200)

        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try nmf.decompose(V)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("NMF (128x200, 8 components, 50 iter): \(elapsed * 1000) ms")
        print("  Converged in \(result.iterations) iterations")
        print("  Final error: \(result.reconstructionError)")
    }

    func testNMFScaling() throws {
        // Test how NMF scales with matrix size
        let nmf = NMF(config: NMFConfig(nComponents: 8, maxIterations: 30))

        let sizes = [(64, 100), (128, 200), (256, 400)]

        for (nFreqs, nFrames) in sizes {
            let V = generateTestSpectrogram(nFreqs: nFreqs, nFrames: nFrames)

            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try nmf.decompose(V)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            print("NMF \(nFreqs)x\(nFrames): \(elapsed * 1000) ms")
        }
    }

    func testNNFilterPerformance() {
        let nnFilter = NNFilter(config: NNFilterConfig(
            k: 5,
            width: 31,
            aggregation: .median
        ))

        let features = generateTestSpectrogram(nFreqs: 128, nFrames: 500)

        let startTime = CFAbsoluteTimeGetCurrent()
        _ = nnFilter.filter(features)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("NN Filter (128x500, k=5, width=31): \(elapsed * 1000) ms")
    }

    func testNNFilterDecomposePerformance() {
        let nnFilter = NNFilter(config: NNFilterConfig(k: 7, width: 31))

        let features = generateTestSpectrogram(nFreqs: 128, nFrames: 500)

        let startTime = CFAbsoluteTimeGetCurrent()
        _ = nnFilter.decompose(features)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        print("NN Filter Decompose (128x500): \(elapsed * 1000) ms")
    }

    // MARK: - MR Frequencies Benchmark

    func testMRFrequenciesPerformance() {
        measure {
            let config = MRFrequenciesConfig(
                sampleRate: 22050,
                fMin: 32.70,
                nBins: 168,  // 14 octaves at 12 bins per octave
                binsPerOctave: 12,
                scale: .vqt,
                gamma: 20
            )
            _ = MRFrequencies.compute(config: config)
        }
    }

    // MARK: - Summary

    func testPrintSummary() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("NEW FEATURES BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 60))

        let signal10s = generateTestSignal(duration: 10.0)
        let signal5s = generateTestSignal(duration: 5.0)
        let spec128x500 = generateTestSpectrogram(nFreqs: 128, nFrames: 500)

        // RMS
        var start = CFAbsoluteTimeGetCurrent()
        let rms = RMSEnergy()
        _ = rms.transform(signal10s)
        print("RMS Energy (10s):          \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // Delta
        let features40x1000 = generateTestSpectrogram(nFreqs: 40, nFrames: 1000)
        start = CFAbsoluteTimeGetCurrent()
        _ = Delta.compute(features40x1000)
        print("Delta (40x1000):           \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // A-Weighting - apply to spectrogram
        let spec1025x500 = generateTestSpectrogram(nFreqs: 1025, nFrames: 500)
        start = CFAbsoluteTimeGetCurrent()
        _ = AWeighting.apply(to: spec1025x500, nFFT: 2048, sampleRate: 22050)
        print("A-Weighting (1025x500):    \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // PCEN - synchronous
        let pcen = PCEN()
        start = CFAbsoluteTimeGetCurrent()
        _ = pcen.transform(spec128x500)
        print("PCEN (128x500):            \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // Poly Features
        let poly = PolyFeatures(config: PolyFeaturesConfig(order: 2))
        start = CFAbsoluteTimeGetCurrent()
        _ = poly.transform(magnitudeSpec: spec1025x500)
        print("PolyFeatures (1025x500):   \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // Generators
        start = CFAbsoluteTimeGetCurrent()
        _ = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 10.0)
        print("Tone Generator (10s):      \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        start = CFAbsoluteTimeGetCurrent()
        _ = ChirpGenerator.chirp(fMin: 100, fMax: 10000, sampleRate: 22050, duration: 10.0)
        print("Chirp Generator (10s):     \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // Transforms
        let tonnetz = Tonnetz()
        start = CFAbsoluteTimeGetCurrent()
        _ = await tonnetz.transform(signal5s)
        print("Tonnetz (5s):              \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        let pcqt = PseudoCQT()
        start = CFAbsoluteTimeGetCurrent()
        _ = await pcqt.magnitude(signal5s)
        print("Pseudo-CQT (5s):           \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        let vqt = VQT.erb
        start = CFAbsoluteTimeGetCurrent()
        _ = await vqt.magnitude(signal5s)
        print("VQT (5s, ERB):             \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        let tempogram = Tempogram.standard
        start = CFAbsoluteTimeGetCurrent()
        _ = await tempogram.transform(signal10s)
        print("Tempogram (10s):           \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        // Decomposition
        let nmf = NMF(config: NMFConfig(nComponents: 8, maxIterations: 50))
        start = CFAbsoluteTimeGetCurrent()
        _ = try nmf.decompose(spec128x500)
        print("NMF (128x500, 8 comp):     \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        let nnFilter = NNFilter.standard
        start = CFAbsoluteTimeGetCurrent()
        _ = nnFilter.filter(spec128x500)
        print("NN Filter (128x500):       \(String(format: "%8.2f", (CFAbsoluteTimeGetCurrent() - start) * 1000)) ms")

        print(String(repeating: "=", count: 60))
        print("")
    }
}
