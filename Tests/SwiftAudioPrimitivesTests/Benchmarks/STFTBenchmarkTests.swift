import XCTest

@testable import SwiftAudioPrimitives

/// Benchmark tests for STFT and related operations.
///
/// These tests measure performance and memory usage to establish baselines
/// and detect regressions after optimizations.
final class STFTBenchmarkTests: XCTestCase {
    // MARK: - Test Configuration

    /// Standard signal lengths for benchmarking (samples at 22050 Hz)
    static let signalLengths = [
        22050,      // 1 second
        220500,     // 10 seconds
        1323000,    // 60 seconds
    ]

    static let sampleRate: Float = 22050

    // MARK: - STFT Performance Tests

    func testSTFTPerformance_1s() async throws {
        let signal = generateTestSignal(length: 22050)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .thorough)

        let result = try await runner.run(
            name: "STFT_1s",
            parameters: ["nFFT": "2048", "hopLength": "512"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal)
        }

        // Should be significantly faster than real-time for 1s
        XCTAssertGreaterThan(
            result.throughput?.realTimeFactor ?? 0, 10.0,
            "STFT should be >10x real-time for 1s audio"
        )

        print("STFT 1s: \(result.summary)")
    }

    func testSTFTPerformance_10s() async throws {
        let signal = generateTestSignal(length: 220500)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .thorough)

        let result = try await runner.run(
            name: "STFT_10s",
            parameters: ["nFFT": "2048", "hopLength": "512"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal)
        }

        // Should still be faster than real-time
        XCTAssertGreaterThan(
            result.throughput?.realTimeFactor ?? 0, 1.0,
            "STFT should be faster than real-time for 10s audio"
        )

        print("STFT 10s: \(result.summary)")
    }

    func testSTFTPerformance_60s() async throws {
        let signal = generateTestSignal(length: 1323000)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: BenchmarkConfiguration(
            warmupIterations: 2,
            measureIterations: 5,
            captureMemory: true
        ))

        let result = try await runner.run(
            name: "STFT_60s",
            parameters: ["nFFT": "2048", "hopLength": "512"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal)
        }

        print("STFT 60s: \(result.summary)")
    }

    // MARK: - FFT Size Comparison

    func testSTFTPerformance_FFTSizes() async throws {
        let signal = generateTestSignal(length: 220500) // 10s
        let runner = BenchmarkRunner(config: .thorough)

        let fftSizes = [1024, 2048, 4096]

        for nFFT in fftSizes {
            let stft = STFT(config: STFTConfig(nFFT: nFFT))

            let result = try await runner.run(
                name: "STFT_nFFT\(nFFT)",
                parameters: ["nFFT": "\(nFFT)", "signalLength": "220500"],
                samplesProcessed: signal.count,
                sampleRate: Self.sampleRate
            ) {
                _ = await stft.transform(signal)
            }

            print("STFT nFFT=\(nFFT): \(result.wallTimeStats.formattedMean)")
        }
    }

    // MARK: - ISTFT Performance

    func testISTFTPerformance() async throws {
        let signal = generateTestSignal(length: 220500)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let istft = ISTFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .thorough)

        // First compute STFT
        let spectrogram = await stft.transform(signal)

        let result = try await runner.run(
            name: "ISTFT_10s",
            parameters: ["nFFT": "2048"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await istft.transform(spectrogram, length: signal.count)
        }

        print("ISTFT 10s: \(result.summary)")
    }

    // MARK: - Round-Trip Performance

    func testRoundTripPerformance() async throws {
        let signal = generateTestSignal(length: 220500)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let istft = ISTFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .thorough)

        let result = try await runner.run(
            name: "RoundTrip_10s",
            parameters: ["nFFT": "2048"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            let spec = await stft.transform(signal)
            _ = await istft.transform(spec, length: signal.count)
        }

        // Round-trip should still be faster than real-time
        XCTAssertGreaterThan(
            result.throughput?.realTimeFactor ?? 0, 0.5,
            "Round-trip should be at least 0.5x real-time for 10s audio"
        )

        print("Round-trip 10s: \(result.summary)")
    }

    // MARK: - Mel Spectrogram Performance

    func testMelSpectrogramPerformance() async throws {
        let signal = generateTestSignal(length: 220500)
        let melSpec = MelSpectrogram(
            sampleRate: Self.sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )
        let runner = BenchmarkRunner(config: .thorough)

        let result = try await runner.run(
            name: "MelSpectrogram_10s",
            parameters: ["nMels": "128", "nFFT": "2048"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await melSpec.transform(signal)
        }

        print("Mel Spectrogram 10s: \(result.summary)")
    }

    // MARK: - Comprehensive Benchmark Suite

    func testComprehensiveBenchmarkSuite() async throws {
        let runner = BenchmarkRunner(config: .thorough)
        let printer = BenchmarkPrinter()

        // Generate test signals
        let signals: [(name: String, length: Int)] = [
            ("1s", 22050),
            ("10s", 220500),
            ("60s", 1323000),
        ]

        for (name, length) in signals {
            let signal = generateTestSignal(length: length)
            let stft = STFT(config: STFTConfig(nFFT: 2048))

            _ = try await runner.run(
                name: "STFT_\(name)",
                parameters: ["signalLength": "\(length)"],
                samplesProcessed: length,
                sampleRate: Self.sampleRate
            ) {
                _ = await stft.transform(signal)
            }
        }

        // Print summary
        printer.printSummary(await runner.allResults())

        // Generate report
        let reportGen = ReportGenerator()
        let markdown = reportGen.generateMarkdown(results: await runner.allResults())
        print("\n--- Markdown Report ---\n\(markdown)")
    }

    // MARK: - Helper Methods

    private func generateTestSignal(length: Int) -> [Float] {
        // Generate a multi-frequency test signal
        (0..<length).map { i in
            let t = Float(i) / Self.sampleRate
            return 0.5 * sin(2 * Float.pi * 440 * t) +
                   0.3 * sin(2 * Float.pi * 880 * t) +
                   0.2 * sin(2 * Float.pi * 1760 * t)
        }
    }
}
