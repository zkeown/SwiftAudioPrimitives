import XCTest
import Foundation

@testable import SwiftRosaCore
@testable import SwiftRosaEffects
@testable import SwiftRosaAnalysis
@testable import SwiftRosaML

// MARK: - Librosa Timing Data Structures

/// Timing data loaded from librosa benchmark JSON.
struct LibrosaTimings: Codable {
    let metadata: LibrosaMetadata
    let benchmarks: [String: LibrosaBenchmark]
}

struct LibrosaMetadata: Codable {
    let generator: String
    let generated_at: String
    let librosa_version: String
    let numpy_version: String
    let scipy_version: String
    let sample_rate: Int
    let hardware: LibrosaHardware
}

struct LibrosaHardware: Codable {
    let platform: String
    let processor: String
    let machine: String
    let python_version: String
    let cpu_brand: String?
    let mac_model: String?
}

struct LibrosaBenchmark: Codable {
    let operation: String
    let parameters: [String: AnyCodableValue]
    let signal_length_seconds: Double
    let iterations: Int
    let mean_ms: Double
    let std_ms: Double
    let min_ms: Double
    let max_ms: Double
    let samples_per_second: Double
    let real_time_factor: Double
}

/// Wrapper for heterogeneous JSON values in parameters.
enum AnyCodableValue: Codable {
    case int(Int)
    case double(Double)
    case string(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) {
            self = .int(intVal)
        } else if let doubleVal = try? container.decode(Double.self) {
            self = .double(doubleVal)
        } else if let stringVal = try? container.decode(String.self) {
            self = .string(stringVal)
        } else {
            throw DecodingError.typeMismatch(
                AnyCodableValue.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported type")
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .int(let val): try container.encode(val)
        case .double(let val): try container.encode(val)
        case .string(let val): try container.encode(val)
        }
    }

    var intValue: Int? {
        switch self {
        case .int(let v): return v
        case .double(let v): return Int(v)
        case .string(let v): return Int(v)
        }
    }
}

// MARK: - Comparison Result

/// Result of comparing SwiftRosa vs librosa performance.
struct ComparisonResult {
    let operation: String
    let swiftRosaMs: Double
    let librosaMs: Double
    let speedup: Double  // > 1 means SwiftRosa is faster
    let passed: Bool     // SwiftRosa >= librosa speed

    var summary: String {
        let status = passed ? "PASS" : "FAIL"
        return String(format: "[\(status)] %@: Swift=%.2fms, librosa=%.2fms, speedup=%.2fx",
                      operation, swiftRosaMs, librosaMs, speedup)
    }
}

// MARK: - LibrosaComparisonTests

/// Comprehensive performance comparison tests: SwiftRosa vs librosa.
///
/// These tests load timing baselines from librosa and compare against
/// SwiftRosa implementations. The goal is for SwiftRosa to match or
/// exceed librosa performance on all operations.
final class LibrosaComparisonTests: XCTestCase {

    // MARK: - Properties

    static var librosaTimings: LibrosaTimings?
    static let sampleRate: Float = 22050

    // Pre-computed test signals
    static let signal1s = generateTestSignalStatic(duration: 1.0)
    static let signal10s = generateTestSignalStatic(duration: 10.0)
    static let signal60s = generateTestSignalStatic(duration: 60.0)

    var comparisonResults: [ComparisonResult] = []

    // MARK: - Setup

    override class func setUp() {
        super.setUp()
        loadLibrosaTimings()
    }

    private static func loadLibrosaTimings() {
        let bundle = Bundle(for: LibrosaComparisonTests.self)

        // Try bundle resource first
        if let url = bundle.url(forResource: "librosa_timings", withExtension: "json", subdirectory: "ReferenceTimings") {
            loadTimings(from: url)
            return
        }

        // Try relative path from test bundle
        let testBundlePath = bundle.bundlePath
        let possiblePaths = [
            // Direct path in test bundle
            URL(fileURLWithPath: testBundlePath).appendingPathComponent("ReferenceTimings/librosa_timings.json"),
            // Path relative to source file (#file)
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceTimings/librosa_timings.json"),
            // Path from current working directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Tests/SwiftRosaBenchmarks/ReferenceTimings/librosa_timings.json"),
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                loadTimings(from: path)
                return
            }
        }

        print("WARNING: Could not find librosa_timings.json. Run benchmark_librosa.py first.")
        print("Searched paths:")
        for path in possiblePaths {
            print("  - \(path.path)")
        }
    }

    private static func loadTimings(from url: URL) {
        do {
            let data = try Data(contentsOf: url)
            librosaTimings = try JSONDecoder().decode(LibrosaTimings.self, from: data)
            print("Loaded librosa timings: \(librosaTimings?.benchmarks.count ?? 0) benchmarks")
            print("Hardware: \(librosaTimings?.metadata.hardware.cpu_brand ?? "Unknown")")
        } catch {
            print("ERROR loading librosa timings: \(error)")
        }
    }

    private static func generateTestSignalStatic(duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            let t = Float(i) / sampleRate
            return 0.5 * sin(2 * Float.pi * 440 * t) +
                   0.3 * sin(2 * Float.pi * 880 * t) +
                   0.2 * sin(2 * Float.pi * 1760 * t)
        }
    }

    // MARK: - Helpers

    private func getLibrosaTiming(_ name: String) -> LibrosaBenchmark? {
        return Self.librosaTimings?.benchmarks[name]
    }

    private func measureSwiftRosa(
        iterations: Int = 10,
        warmup: Int = 2,
        block: () async throws -> Void
    ) async throws -> Double {
        // Warmup
        for _ in 0..<warmup {
            try await block()
        }

        // Measure
        var timings: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try await block()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000  // ms
            timings.append(elapsed)
        }

        return timings.reduce(0, +) / Double(timings.count)
    }

    private func measureSwiftRosaSync(
        iterations: Int = 10,
        warmup: Int = 2,
        block: () throws -> Void
    ) throws -> Double {
        // Warmup
        for _ in 0..<warmup {
            try block()
        }

        // Measure
        var timings: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try block()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000  // ms
            timings.append(elapsed)
        }

        return timings.reduce(0, +) / Double(timings.count)
    }

    private func compare(
        operation: String,
        swiftRosaMs: Double,
        allowSlower: Double = 1.0  // Allow SwiftRosa to be up to this factor slower (1.0 = must be equal or faster)
    ) -> ComparisonResult {
        guard let librosa = getLibrosaTiming(operation) else {
            print("WARNING: No librosa timing for \(operation)")
            return ComparisonResult(
                operation: operation,
                swiftRosaMs: swiftRosaMs,
                librosaMs: 0,
                speedup: 0,
                passed: true  // Pass if no baseline
            )
        }

        let speedup = librosa.mean_ms / swiftRosaMs
        let passed = speedup >= (1.0 / allowSlower)

        let result = ComparisonResult(
            operation: operation,
            swiftRosaMs: swiftRosaMs,
            librosaMs: librosa.mean_ms,
            speedup: speedup,
            passed: passed
        )

        print(result.summary)
        return result
    }

    // MARK: - STFT Tests

    func testSTFT_1s_nfft2048() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let signal = Self.signal1s

        let swiftMs = try await measureSwiftRosa {
            _ = await stft.transform(signal)
        }

        let result = compare(operation: "stft_1s_nfft2048", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa STFT should be >= librosa speed")
    }

    func testSTFT_10s_nfft1024() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 1024))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await stft.transform(signal)
        }

        let result = compare(operation: "stft_10s_nfft1024", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa STFT should be >= librosa speed")
    }

    func testSTFT_10s_nfft2048() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await stft.transform(signal)
        }

        let result = compare(operation: "stft_10s_nfft2048", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa STFT should be >= librosa speed")
    }

    func testSTFT_10s_nfft4096() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 4096))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await stft.transform(signal)
        }

        let result = compare(operation: "stft_10s_nfft4096", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa STFT should be >= librosa speed")
    }

    func testSTFT_60s_nfft2048() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let signal = Self.signal60s

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await stft.transform(signal)
        }

        let result = compare(operation: "stft_60s_nfft2048", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa STFT should be >= librosa speed")
    }

    // MARK: - ISTFT Tests

    func testISTFT_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let istft = ISTFT(config: STFTConfig(nFFT: 2048))
        let signal = Self.signal10s

        // Pre-compute STFT
        let spectrogram = await stft.transform(signal)

        let swiftMs = try await measureSwiftRosa {
            _ = await istft.transform(spectrogram, length: signal.count)
        }

        let result = compare(operation: "istft_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa ISTFT should be >= librosa speed")
    }

    func testISTFT_60s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let istft = ISTFT(config: STFTConfig(nFFT: 2048))
        let signal = Self.signal60s

        let spectrogram = await stft.transform(signal)

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await istft.transform(spectrogram, length: signal.count)
        }

        let result = compare(operation: "istft_60s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa ISTFT should be >= librosa speed")
    }

    // MARK: - Mel Spectrogram Tests

    func testMelSpectrogram_10s_128mels() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let melSpec = MelSpectrogram(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await melSpec.transform(signal)
        }

        let result = compare(operation: "mel_10s_nmels128", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa Mel should be >= librosa speed")
    }

    func testMelSpectrogram_60s_128mels() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let melSpec = MelSpectrogram(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )
        let signal = Self.signal60s

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await melSpec.transform(signal)
        }

        let result = compare(operation: "mel_60s_nmels128", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa Mel should be >= librosa speed")
    }

    // MARK: - MFCC Tests

    func testMFCC_10s_13coeffs() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let mfcc = MFCC(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 13
        )
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await mfcc.transform(signal)
        }

        let result = compare(operation: "mfcc_10s_nmfcc13", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa MFCC should be >= librosa speed")
    }

    func testMFCC_60s_13coeffs() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let mfcc = MFCC(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 13
        )
        let signal = Self.signal60s

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await mfcc.transform(signal)
        }

        let result = compare(operation: "mfcc_60s_nmfcc13", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa MFCC should be >= librosa speed")
    }

    // MARK: - CQT Tests

    func testCQT_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let cqt = CQT(config: CQTConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512,
            nBins: 84,
            binsPerOctave: 12
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await cqt.transform(signal)
        }

        let result = compare(operation: "cqt_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa CQT should be >= librosa speed")
    }

    func testCQT_60s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let cqt = CQT(config: CQTConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512,
            nBins: 84,
            binsPerOctave: 12
        ))
        let signal = Self.signal60s

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await cqt.transform(signal)
        }

        let result = compare(operation: "cqt_60s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa CQT should be >= librosa speed")
    }

    // MARK: - Spectral Features Tests

    func testSpectralCentroid_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await spectralFeatures.centroid(signal)
        }

        let result = compare(operation: "spectral_centroid_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa spectral centroid should be >= librosa speed")
    }

    func testSpectralBandwidth_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await spectralFeatures.bandwidth(signal)
        }

        let result = compare(operation: "spectral_bandwidth_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa spectral bandwidth should be >= librosa speed")
    }

    func testSpectralRolloff_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await spectralFeatures.rolloff(signal)
        }

        let result = compare(operation: "spectral_rolloff_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa spectral rolloff should be >= librosa speed")
    }

    func testSpectralFlatness_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await spectralFeatures.flatness(signal)
        }

        let result = compare(operation: "spectral_flatness_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa spectral flatness should be >= librosa speed")
    }

    func testSpectralContrast_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await spectralFeatures.contrast(signal)
        }

        let result = compare(operation: "spectral_contrast_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa spectral contrast should be >= librosa speed")
    }

    // MARK: - Onset Detection Tests

    func testOnsetStrength_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await onsetDetector.onsetStrength(signal)
        }

        let result = compare(operation: "onset_strength_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa onset strength should be >= librosa speed")
    }

    func testOnsetDetect_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await onsetDetector.detect(signal)
        }

        let result = compare(operation: "onset_detect_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa onset detect should be >= librosa speed")
    }

    // MARK: - Resampling Tests

    func testResample_10s_22050to16000() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let signal = Self.signal10s

        let swiftMs = try measureSwiftRosaSync {
            _ = Resample.resampleSync(signal, fromSampleRate: 22050, toSampleRate: 16000)
        }

        let result = compare(operation: "resample_10s_22050to16000", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa resample should be >= librosa speed")
    }

    func testResample_10s_22050to44100() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let signal = Self.signal10s

        let swiftMs = try measureSwiftRosaSync {
            _ = Resample.resampleSync(signal, fromSampleRate: 22050, toSampleRate: 44100)
        }

        let result = compare(operation: "resample_10s_22050to44100", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa resample should be >= librosa speed")
    }

    // MARK: - HPSS Tests

    func testHPSS_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let hpss = HPSS()
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa(iterations: 5, warmup: 1) {
            _ = await hpss.separate(signal)
        }

        let result = compare(operation: "hpss_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa HPSS should be >= librosa speed")
    }

    // MARK: - Chromagram Tests

    func testChromaSTFT_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let chromagram = Chromagram(config: ChromagramConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512,
            useCQT: false,
            nFFT: 2048
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await chromagram.transform(signal)
        }

        let result = compare(operation: "chroma_stft_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa chroma STFT should be >= librosa speed")
    }

    func testChromaCQT_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let chromagram = Chromagram(config: ChromagramConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512,
            useCQT: true
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await chromagram.transform(signal)
        }

        let result = compare(operation: "chroma_cqt_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa chroma CQT should be >= librosa speed")
    }

    // MARK: - RMS Tests

    func testRMS_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let rms = RMSEnergy(config: RMSEnergyConfig(frameLength: 2048, hopLength: 512))
        let signal = Self.signal10s

        let swiftMs = try measureSwiftRosaSync {
            _ = rms.transform(signal)
        }

        let result = compare(operation: "rms_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa RMS should be >= librosa speed")
    }

    // MARK: - ZCR Tests

    func testZCR_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let zcr = ZeroCrossingRate(frameLength: 2048, hopLength: 512)
        let signal = Self.signal10s

        let swiftMs = try measureSwiftRosaSync {
            _ = zcr.transform(signal)
        }

        let result = compare(operation: "zcr_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa ZCR should be >= librosa speed")
    }

    // MARK: - Delta Tests

    func testDelta_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let mfcc = MFCC(sampleRate: Self.sampleRate, nMFCC: 13)
        let signal = Self.signal10s

        // Pre-compute MFCCs
        let mfccs = await mfcc.transform(signal)

        let swiftMs = try measureSwiftRosaSync {
            _ = Delta.compute(mfccs, width: 5, order: 1)
        }

        let result = compare(operation: "delta_10s_width5", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa delta should be >= librosa speed")
    }

    // MARK: - Tonnetz Tests

    func testTonnetz_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let tonnetz = Tonnetz(config: TonnetzConfig(sampleRate: Self.sampleRate))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await tonnetz.transform(signal)
        }

        let result = compare(operation: "tonnetz_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa tonnetz should be >= librosa speed")
    }

    // MARK: - Tempogram Tests

    func testTempogram_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: Self.sampleRate,
            hopLength: 512
        ))
        let signal = Self.signal10s

        let swiftMs = try await measureSwiftRosa {
            _ = await tempogram.transform(signal)
        }

        let result = compare(operation: "tempogram_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa tempogram should be >= librosa speed")
    }

    // MARK: - PCEN Tests

    func testPCEN_10s() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        let pcen = PCEN()
        let melSpec = MelSpectrogram(sampleRate: Self.sampleRate, nMels: 128)
        let signal = Self.signal10s

        // Pre-compute mel spectrogram
        let mel = await melSpec.transform(signal)

        let swiftMs = try measureSwiftRosaSync {
            _ = pcen.transform(mel)
        }

        let result = compare(operation: "pcen_10s", swiftRosaMs: swiftMs)
        XCTAssertTrue(result.passed, "SwiftRosa PCEN should be >= librosa speed")
    }

    // MARK: - Summary Test

    func testPrintComparisonSummary() async throws {
        try XCTSkipIf(Self.librosaTimings == nil, "Librosa timings not loaded")

        print("\n" + String(repeating: "=", count: 70))
        print("SwiftRosa vs librosa Performance Summary")
        print(String(repeating: "=", count: 70))

        guard let timings = Self.librosaTimings else { return }

        print("\nlibrosa version: \(timings.metadata.librosa_version)")
        print("Hardware: \(timings.metadata.hardware.cpu_brand ?? "Unknown")")
        print("Total librosa benchmarks: \(timings.benchmarks.count)")

        print("\n" + String(repeating: "-", count: 70))
        print("Key Operations (10s signal):")
        print(String(repeating: "-", count: 70))

        let keyOps = [
            "stft_10s_nfft2048",
            "istft_10s",
            "mel_10s_nmels128",
            "mfcc_10s_nmfcc13",
            "cqt_10s",
            "onset_strength_10s",
            "resample_10s_22050to16000",
            "hpss_10s",
            "chroma_stft_10s",
        ]

        for op in keyOps {
            if let timing = timings.benchmarks[op] {
                print(String(format: "  %-35s %8.2fms  (%.1fx RT)",
                             op, timing.mean_ms, timing.real_time_factor))
            }
        }

        print(String(repeating: "=", count: 70))
    }
}
