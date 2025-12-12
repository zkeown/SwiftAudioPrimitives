import XCTest

@testable import SwiftAudioPrimitives

/// Benchmark tests focused on memory usage and allocation patterns.
///
/// These tests measure memory footprint to establish baselines
/// for optimization work.
final class MemoryBenchmarkTests: XCTestCase {
    static let sampleRate: Float = 22050

    // MARK: - STFT Memory Tests

    func testSTFTMemoryUsage_10s() async throws {
        let signal = generateTestSignal(length: 220500)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .memoryFocused)

        let result = try await runner.run(
            name: "STFT_Memory_10s",
            parameters: ["nFFT": "2048"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal)
        }

        if let delta = result.memoryDelta {
            print("STFT 10s Memory:")
            print("  Peak: \(delta.formattedPeak)")
            print("  Delta: \(delta.formattedAllocated)")
        }

        // Memory baseline assertion - adjust after optimization
        // Currently expecting significant memory usage due to [[Float]] structure
        if let peak = result.peakMemory {
            // For 10s audio, peak should be reasonable (< 500MB)
            XCTAssertLessThan(
                peak, 500 * 1024 * 1024,
                "STFT peak memory should be < 500MB for 10s audio"
            )
        }
    }

    func testSTFTMemoryUsage_60s() async throws {
        let signal = generateTestSignal(length: 1323000)
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let runner = BenchmarkRunner(config: .memoryFocused)

        let result = try await runner.run(
            name: "STFT_Memory_60s",
            parameters: ["nFFT": "2048"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal)
        }

        if let delta = result.memoryDelta {
            print("STFT 60s Memory:")
            print("  Peak: \(delta.formattedPeak)")
            print("  Delta: \(delta.formattedAllocated)")
        }
    }

    // MARK: - GriffinLim Memory Tests

    func testGriffinLimMemoryUsage() async throws {
        // Generate a magnitude spectrogram for reconstruction
        let signal = generateTestSignal(length: 110250) // 5s
        let stft = STFT(config: STFTConfig(nFFT: 2048))
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        let griffinLim = GriffinLim(
            config: STFTConfig(nFFT: 2048),
            nIter: 32,
            momentum: 0.99
        )

        let runner = BenchmarkRunner(config: BenchmarkConfiguration(
            warmupIterations: 1,
            measureIterations: 3,
            captureMemory: true,
            gcBetweenIterations: true
        ))

        let result = try await runner.run(
            name: "GriffinLim_5s_32iter",
            parameters: ["nIter": "32", "momentum": "0.99"],
            samplesProcessed: signal.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await griffinLim.reconstruct(magnitude, length: signal.count)
        }

        if let delta = result.memoryDelta {
            print("GriffinLim 5s (32 iter) Memory:")
            print("  Peak: \(delta.formattedPeak)")
            print("  Delta: \(delta.formattedAllocated)")
        }

        print("GriffinLim timing: \(result.wallTimeStats.formattedMean)")
    }

    // MARK: - Streaming Memory Tests

    func testStreamingSTFTMemoryGrowth() async throws {
        let streaming = StreamingSTFT(config: STFTConfig(nFFT: 2048, hopLength: 512))
        let memoryTracker = MemoryTracker()

        await memoryTracker.begin()

        // Process 60 seconds of audio in 1-second chunks
        let chunkSize = 22050 // 1 second
        let totalChunks = 60

        var totalFrames = 0
        for i in 0..<totalChunks {
            let chunk = generateTestSignal(length: chunkSize, offset: i * chunkSize)
            await streaming.push(chunk)
            let frames = await streaming.popFrames()
            totalFrames += frames.count

            // Sample memory every 10 seconds
            if i % 10 == 0 {
                await memoryTracker.sample()
            }
        }

        // Flush remaining
        let remaining = await streaming.flush()
        totalFrames += remaining.count

        let delta = await memoryTracker.end()

        print("Streaming STFT (60s) Memory:")
        print("  Peak: \(delta.formattedPeak)")
        print("  Delta: \(delta.formattedAllocated)")
        print("  Frames processed: \(totalFrames)")

        // Memory should not grow unboundedly for streaming
        // Peak should be bounded by buffer size + frame processing overhead
        XCTAssertLessThan(
            delta.peakFootprint, 100 * 1024 * 1024,
            "Streaming peak memory should be < 100MB for 60s"
        )
    }

    // MARK: - ComplexMatrix Memory Tests

    func testComplexMatrixMemoryLayout() async throws {
        let runner = BenchmarkRunner(config: .memoryFocused)

        // Typical spectrogram dimensions
        let rows = 1025 // nFreqs for nFFT=2048
        let cols = 1000 // ~10s at hop=512

        let result = try await runner.run(
            name: "ComplexMatrix_Creation",
            parameters: ["rows": "\(rows)", "cols": "\(cols)"]
        ) {
            // Create a ComplexMatrix with realistic size
            var real = [[Float]]()
            var imag = [[Float]]()
            real.reserveCapacity(rows)
            imag.reserveCapacity(rows)

            for i in 0..<rows {
                real.append([Float](repeating: Float(i), count: cols))
                imag.append([Float](repeating: Float(i), count: cols))
            }

            let matrix = ComplexMatrix(real: real, imag: imag)
            // Force evaluation
            _ = matrix.rows
        }

        if let delta = result.memoryDelta {
            print("ComplexMatrix \(rows)x\(cols) Memory:")
            print("  Peak: \(delta.formattedPeak)")

            // Calculate theoretical minimum
            let theoreticalMin = rows * cols * 2 * MemoryLayout<Float>.size
            print("  Theoretical minimum: \(ByteCountFormatter.string(fromByteCount: Int64(theoreticalMin), countStyle: .memory))")
        }
    }

    func testComplexMatrixMagnitudeMemory() async throws {
        // Create a pre-made ComplexMatrix
        let rows = 1025
        let cols = 1000

        var real = [[Float]]()
        var imag = [[Float]]()
        for i in 0..<rows {
            real.append([Float](repeating: Float(i) * 0.001, count: cols))
            imag.append([Float](repeating: Float(i) * 0.001, count: cols))
        }
        let matrix = ComplexMatrix(real: real, imag: imag)

        let runner = BenchmarkRunner(config: .memoryFocused)

        let result = try await runner.run(
            name: "ComplexMatrix_Magnitude",
            parameters: ["rows": "\(rows)", "cols": "\(cols)"]
        ) {
            // Compute magnitude
            _ = matrix.magnitude
        }

        if let delta = result.memoryDelta {
            print("ComplexMatrix Magnitude Memory:")
            print("  Peak: \(delta.formattedPeak)")
            print("  Delta: \(delta.formattedAllocated)")
        }

        print("Magnitude timing: \(result.wallTimeStats.formattedMean)")
    }

    // MARK: - Memory Comparison Before/After

    func testMemoryBaselineCapture() async throws {
        let runner = BenchmarkRunner(config: .memoryFocused)
        var results: [BenchmarkResult] = []

        // STFT 10s
        let signal10s = generateTestSignal(length: 220500)
        let stft = STFT(config: STFTConfig(nFFT: 2048))

        let stftResult = try await runner.run(
            name: "STFT_10s_Baseline",
            samplesProcessed: signal10s.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await stft.transform(signal10s)
        }
        results.append(stftResult)

        // Mel Spectrogram
        let melSpec = MelSpectrogram(sampleRate: Self.sampleRate, nMels: 128)
        let melResult = try await runner.run(
            name: "MelSpec_10s_Baseline",
            samplesProcessed: signal10s.count,
            sampleRate: Self.sampleRate
        ) {
            _ = await melSpec.transform(signal10s)
        }
        results.append(melResult)

        // Print baseline summary
        print("\n=== Memory Baselines ===")
        for result in results {
            print("\(result.name):")
            print("  Time: \(result.wallTimeStats.formattedMean)")
            if let peak = result.peakMemory {
                print("  Peak Memory: \(ByteCountFormatter.string(fromByteCount: Int64(peak), countStyle: .memory))")
            }
        }
    }

    // MARK: - Helper Methods

    private func generateTestSignal(length: Int, offset: Int = 0) -> [Float] {
        (0..<length).map { i in
            let t = Float(i + offset) / Self.sampleRate
            return 0.5 * sin(2 * Float.pi * 440 * t) +
                   0.3 * sin(2 * Float.pi * 880 * t)
        }
    }
}
