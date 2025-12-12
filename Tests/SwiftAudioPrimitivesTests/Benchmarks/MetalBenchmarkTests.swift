import XCTest

@testable import SwiftAudioPrimitives

/// Benchmark tests comparing Metal GPU vs CPU performance.
final class MetalBenchmarkTests: XCTestCase {
    static let sampleRate: Float = 22050

    // MARK: - Metal Availability

    func testMetalAvailability() throws {
        XCTAssertTrue(MetalEngine.isAvailable, "Metal should be available on this device")
    }

    // MARK: - Overlap-Add GPU vs CPU

    func testOverlapAddGPUvsCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .thorough)
        let engine = try MetalEngine()

        // Generate test frames (100 frames of 2048 samples each)
        let frameCount = 100
        let frameLength = 2048
        let hopLength = 512
        let outputLength = frameCount * hopLength + frameLength

        let frames: [[Float]] = (0..<frameCount).map { i in
            (0..<frameLength).map { j in
                sin(Float(i * frameLength + j) * 0.01)
            }
        }

        let window = Windows.generate(.hann, length: frameLength, periodic: true)

        // CPU baseline (using ISTFT's overlap-add logic pattern)
        let cpuResult = try await runner.run(
            name: "OverlapAdd_CPU",
            parameters: ["frames": "\(frameCount)", "frameLength": "\(frameLength)"]
        ) {
            var output = [Float](repeating: 0, count: outputLength)
            for (i, frame) in frames.enumerated() {
                let startIdx = i * hopLength
                for j in 0..<frameLength {
                    if startIdx + j < outputLength {
                        output[startIdx + j] += frame[j] * window[j]
                    }
                }
            }
            _ = output.count // Force evaluation
        }

        // Metal GPU
        let gpuResult = try await runner.run(
            name: "OverlapAdd_GPU",
            parameters: ["frames": "\(frameCount)", "frameLength": "\(frameLength)"]
        ) {
            _ = try await engine.overlapAdd(
                frames: frames,
                window: window,
                hopLength: hopLength,
                outputLength: outputLength
            )
        }

        let speedup = cpuResult.wallTimeStats.mean / gpuResult.wallTimeStats.mean

        print("\n=== Overlap-Add GPU vs CPU ===")
        print("CPU: \(cpuResult.wallTimeStats.formattedMean)")
        print("GPU: \(gpuResult.wallTimeStats.formattedMean)")
        print("Speedup: \(String(format: "%.2fx", speedup))")

        // Note: GPU may not be faster for small data sizes due to dispatch overhead
    }

    // MARK: - Mel Filterbank GPU vs CPU

    func testMelFilterbankGPUvsCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .thorough)
        let engine = try MetalEngine()

        // Generate test spectrogram (1025 freqs x 1000 frames)
        let nFreqs = 1025
        let nFrames = 1000
        let nMels = 128

        let spectrogram: [[Float]] = (0..<nFreqs).map { i in
            (0..<nFrames).map { j in Float(i * nFrames + j) * 0.001 }
        }

        // Generate filterbank matrix
        let filterbank = MelFilterbank(
            nMels: nMels,
            nFFT: 2048,
            sampleRate: Self.sampleRate
        ).filters()

        // CPU baseline using vDSP
        let cpuResult = try await runner.run(
            name: "MelFilterbank_CPU",
            parameters: ["nMels": "\(nMels)", "nFreqs": "\(nFreqs)", "nFrames": "\(nFrames)"]
        ) {
            var result = [[Float]]()
            result.reserveCapacity(nMels)

            for m in 0..<nMels {
                var row = [Float](repeating: 0, count: nFrames)
                for f in 0..<nFreqs {
                    for t in 0..<nFrames {
                        row[t] += filterbank[m][f] * spectrogram[f][t]
                    }
                }
                result.append(row)
            }
            _ = result.count
        }

        // Metal GPU
        let gpuResult = try await runner.run(
            name: "MelFilterbank_GPU",
            parameters: ["nMels": "\(nMels)", "nFreqs": "\(nFreqs)", "nFrames": "\(nFrames)"]
        ) {
            _ = try await engine.applyMelFilterbank(
                spectrogram: spectrogram,
                filterbank: filterbank
            )
        }

        let speedup = cpuResult.wallTimeStats.mean / gpuResult.wallTimeStats.mean

        print("\n=== Mel Filterbank GPU vs CPU ===")
        print("CPU: \(cpuResult.wallTimeStats.formattedMean)")
        print("GPU: \(gpuResult.wallTimeStats.formattedMean)")
        print("Speedup: \(String(format: "%.2fx", speedup))")

        // GPU should be faster for this matrix size
        if nFreqs * nFrames > 100_000 {
            XCTAssertGreaterThan(speedup, 0.5, "GPU should be competitive for large matrices")
        }
    }

    // MARK: - Complex Multiply GPU vs CPU

    func testComplexMultiplyGPUvsCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .thorough)
        let engine = try MetalEngine()

        // Generate test complex matrices (1025 x 1000)
        let rows = 1025
        let cols = 1000

        let aReal: [[Float]] = (0..<rows).map { i in
            (0..<cols).map { j in Float(i * cols + j) * 0.001 }
        }
        let aImag: [[Float]] = (0..<rows).map { i in
            (0..<cols).map { j in Float(i * cols + j) * 0.0005 }
        }
        let bReal = aReal
        let bImag = aImag

        // CPU baseline
        let cpuResult = try await runner.run(
            name: "ComplexMultiply_CPU",
            parameters: ["rows": "\(rows)", "cols": "\(cols)"]
        ) {
            var resultReal = [[Float]]()
            var resultImag = [[Float]]()
            resultReal.reserveCapacity(rows)
            resultImag.reserveCapacity(rows)

            for i in 0..<rows {
                var rowReal = [Float](repeating: 0, count: cols)
                var rowImag = [Float](repeating: 0, count: cols)

                for j in 0..<cols {
                    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                    rowReal[j] = aReal[i][j] * bReal[i][j] - aImag[i][j] * bImag[i][j]
                    rowImag[j] = aReal[i][j] * bImag[i][j] + aImag[i][j] * bReal[i][j]
                }

                resultReal.append(rowReal)
                resultImag.append(rowImag)
            }
            _ = resultReal.count
        }

        // Metal GPU
        let gpuResult = try await runner.run(
            name: "ComplexMultiply_GPU",
            parameters: ["rows": "\(rows)", "cols": "\(cols)"]
        ) {
            _ = try await engine.complexMultiply(
                aReal: aReal,
                aImag: aImag,
                bReal: bReal,
                bImag: bImag
            )
        }

        let speedup = cpuResult.wallTimeStats.mean / gpuResult.wallTimeStats.mean

        print("\n=== Complex Multiply GPU vs CPU ===")
        print("CPU: \(cpuResult.wallTimeStats.formattedMean)")
        print("GPU: \(gpuResult.wallTimeStats.formattedMean)")
        print("Speedup: \(String(format: "%.2fx", speedup))")
    }

    // MARK: - GPU Threshold Analysis

    func testGPUBenefitThreshold() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try MetalEngine()

        // Test different data sizes to find crossover point
        let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]

        print("\n=== GPU Benefit Threshold Analysis ===")
        print("Elements    | CPU          | GPU          | Speedup | GPU Faster?")
        print(String(repeating: "-", count: 70))

        for elementCount in sizes {
            let rows = 1025
            let cols = elementCount / rows

            guard cols > 0 else { continue }

            let spectrogram: [[Float]] = (0..<rows).map { _ in
                [Float](repeating: 0.5, count: cols)
            }

            let filterbank = MelFilterbank(
                nMels: 128,
                nFFT: 2048,
                sampleRate: Self.sampleRate
            ).filters()

            // CPU
            let cpuResult = try await runner.run(
                name: "Threshold_CPU_\(elementCount)"
            ) {
                var result = [[Float]]()
                for m in 0..<128 {
                    var row = [Float](repeating: 0, count: cols)
                    for f in 0..<min(rows, filterbank[m].count) {
                        for t in 0..<cols {
                            row[t] += filterbank[m][f] * spectrogram[f][t]
                        }
                    }
                    result.append(row)
                }
                _ = result
            }

            // GPU
            let gpuResult = try await runner.run(
                name: "Threshold_GPU_\(elementCount)"
            ) {
                _ = try await engine.applyMelFilterbank(
                    spectrogram: spectrogram,
                    filterbank: filterbank
                )
            }

            let speedup = cpuResult.wallTimeStats.mean / gpuResult.wallTimeStats.mean
            let isFaster = speedup > 1.0

            print(String(format: "%10d | %12s | %12s | %7.2fx | %@",
                         elementCount,
                         cpuResult.wallTimeStats.formattedMean,
                         gpuResult.wallTimeStats.formattedMean,
                         speedup,
                         isFaster ? "Yes" : "No"))
        }

        // Verify the heuristic in MetalEngine
        let shouldUse100k = MetalEngine.shouldUseGPU(elementCount: 100_000)
        let shouldUse10k = MetalEngine.shouldUseGPU(elementCount: 10_000)

        print("\nshouldUseGPU(100k): \(shouldUse100k)")
        print("shouldUseGPU(10k): \(shouldUse10k)")
    }

    // MARK: - Comprehensive Metal Benchmark

    func testMetalComprehensiveSuite() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .thorough)
        let printer = BenchmarkPrinter()

        // Run all Metal benchmarks
        let engine = try MetalEngine()

        // Test data
        let frames: [[Float]] = (0..<200).map { i in
            (0..<2048).map { j in sin(Float(i * 2048 + j) * 0.01) }
        }
        let window = Windows.generate(.hann, length: 2048, periodic: true)

        _ = try await runner.run(name: "Metal_OverlapAdd_200frames") {
            _ = try await engine.overlapAdd(
                frames: frames,
                window: window,
                hopLength: 512,
                outputLength: 200 * 512 + 2048
            )
        }

        printer.printSummary(await runner.allResults())
    }
}
