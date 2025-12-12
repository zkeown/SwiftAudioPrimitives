//
//  LSTMBenchmarkTests.swift
//  SwiftRosaBenchmarks
//
//  Performance benchmarks for LSTM operations.
//

import XCTest

@testable import SwiftRosaCore
@testable import SwiftRosaML
@testable import SwiftRosaNN

/// Benchmark tests for Metal-accelerated LSTM operations.
///
/// Tests cover:
/// - Single LSTM cell throughput
/// - Full sequence processing at various lengths
/// - Bidirectional LSTM performance
/// - Banquet-scale configurations (H=256, 12 BiLSTM pairs)
final class LSTMBenchmarkTests: XCTestCase {

    // MARK: - LSTM Cell Benchmarks

    func testLSTMCellThroughput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        // Banquet-style dimensions
        let hiddenSize = 256
        let inputSize = 256
        let batchSize = 1

        // Create weights
        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let cell = try await LSTMCell(weights: weights, config: config, engine: engine)

        // Test data
        let input = (0..<(batchSize * inputSize)).map { _ in Float(drand48() - 0.5) }
        var h = [Float](repeating: 0, count: batchSize * hiddenSize)
        var c = [Float](repeating: 0, count: batchSize * hiddenSize)

        let result = try await runner.run(
            name: "LSTMCell_H256",
            parameters: ["hiddenSize": "\(hiddenSize)", "inputSize": "\(inputSize)"]
        ) {
            let (hNew, cNew) = try await cell.forward(
                input: input,
                hPrev: h,
                cPrev: c,
                batchSize: batchSize
            )
            h = hNew
            c = cNew
        }

        print("\n=== LSTM Cell Throughput (H=256) ===")
        print("Mean: \(result.wallTimeStats.formattedMean)")
        print("StdDev: \(result.wallTimeStats.formattedStdDev)")
        print("Min: \(formatTime(result.wallTimeStats.min))")
        print("Max: \(formatTime(result.wallTimeStats.max))")

        // Calculate cells per second
        let cellsPerSecond = 1.0 / result.wallTimeStats.mean
        print("Throughput: \(String(format: "%.0f", cellsPerSecond)) cells/sec")
    }

    // MARK: - LSTM Layer Benchmarks

    func testLSTMLayerSequence100() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        let hiddenSize = 256
        let inputSize = 256
        let batchSize = 1
        let seqLen = 100

        // Create weights
        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let result = try await runner.run(
            name: "LSTMLayer_seq100",
            parameters: ["seqLen": "\(seqLen)", "hiddenSize": "\(hiddenSize)"]
        ) {
            _ = try await layer.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        let timestepsPerSec = Double(seqLen) / result.wallTimeStats.mean

        print("\n=== LSTM Layer Performance (H=256, seqLen=100) ===")
        print("Mean: \(result.wallTimeStats.formattedMean)")
        print("StdDev: \(result.wallTimeStats.formattedStdDev)")
        print("Min: \(formatTime(result.wallTimeStats.min))")
        print("Max: \(formatTime(result.wallTimeStats.max))")
        print("Throughput: \(String(format: "%.0f", timestepsPerSec)) timesteps/sec")
    }

    // MARK: - BiLSTM Benchmarks

    func testBiLSTMPerformance() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        let hiddenSize = 256
        let inputSize = 256
        let batchSize = 1
        let seqLen = 100

        // Create weights for both directions
        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let forwardWeights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        // Different weights for backward
        let weightIHBack = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHHBack = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let backwardWeights = LSTMLayerWeights(
            weightIH: weightIHBack,
            weightHH: weightHHBack,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            batchFirst: true
        )

        let bilstm = try await BiLSTMLayer(
            forwardWeights: forwardWeights,
            backwardWeights: backwardWeights,
            config: config,
            engine: engine
        )

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let result = try await runner.run(
            name: "BiLSTM_H256_seq100",
            parameters: ["hiddenSize": "\(hiddenSize)", "seqLen": "\(seqLen)"]
        ) {
            _ = try await bilstm.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        print("\n=== BiLSTM Performance (H=256, seqLen=100) ===")
        print("Mean: \(result.wallTimeStats.formattedMean)")
        print("StdDev: \(result.wallTimeStats.formattedStdDev)")
        print("Min: \(formatTime(result.wallTimeStats.min))")
        print("Max: \(formatTime(result.wallTimeStats.max))")

        let timestepsPerSec = Double(seqLen) / result.wallTimeStats.mean
        print("Throughput: \(String(format: "%.0f", timestepsPerSec)) timesteps/sec")
    }

    // MARK: - Banquet-Scale Benchmark

    func testBanquetScaleBiLSTMStack() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        // Banquet uses 12 pairs of BiLSTMs with H=256
        // Each BiLSTM processes time or frequency axis
        let hiddenSize = 256
        let inputSize = 512  // Output of previous layer (2*H for bidirectional)
        let batchSize = 1
        let seqLen = 100  // Typical frame count for short audio

        // Create a single BiLSTM for benchmarking
        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let forwardWeights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let weightIHBack = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHHBack = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let backwardWeights = LSTMLayerWeights(
            weightIH: weightIHBack,
            weightHH: weightHHBack,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            batchFirst: true
        )

        let bilstm = try await BiLSTMLayer(
            forwardWeights: forwardWeights,
            backwardWeights: backwardWeights,
            config: config,
            engine: engine
        )

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        // Benchmark single BiLSTM
        let singleResult = try await runner.run(
            name: "Banquet_SingleBiLSTM",
            parameters: ["hiddenSize": "\(hiddenSize)", "inputSize": "\(inputSize)"]
        ) {
            _ = try await bilstm.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        // Simulate full Banquet stack (12 pairs = 24 BiLSTMs)
        // Running sequentially to get realistic estimate
        let numBiLSTMs = 24

        let stackResult = try await runner.run(
            name: "Banquet_24BiLSTM_Sequential",
            parameters: ["numLayers": "\(numBiLSTMs)", "seqLen": "\(seqLen)"]
        ) {
            for _ in 0..<numBiLSTMs {
                _ = try await bilstm.forward(input, batchSize: batchSize, seqLen: seqLen)
            }
        }

        print("\n=== Banquet-Scale BiLSTM Stack ===")
        print("Configuration: H=256, input=512 (BiLSTM output), seqLen=100")
        print("")
        print("Single BiLSTM:")
        print("  Mean: \(singleResult.wallTimeStats.formattedMean)")
        print("  Min:  \(formatTime(singleResult.wallTimeStats.min))")
        print("  Max:  \(formatTime(singleResult.wallTimeStats.max))")
        print("")
        print("Full Stack (24 BiLSTMs sequential):")
        print("  Mean: \(stackResult.wallTimeStats.formattedMean)")
        print("  Min:  \(formatTime(stackResult.wallTimeStats.min))")
        print("  Max:  \(formatTime(stackResult.wallTimeStats.max))")
        print("")

        // Calculate real-time factor
        // Assuming 22050 Hz, 512 hop length -> ~43 frames/sec of audio
        let audioFramesPerSec = 22050.0 / 512.0
        let processingTimePerFrame = stackResult.wallTimeStats.mean / Double(seqLen)
        let rtf = processingTimePerFrame * audioFramesPerSec

        print("Real-Time Analysis:")
        print("  Processing time per frame: \(formatTime(processingTimePerFrame))")
        print("  Audio frames/sec (22kHz, hop=512): \(String(format: "%.1f", audioFramesPerSec))")
        print("  Real-Time Factor (RTF): \(String(format: "%.2fx", rtf))")
        print("  Status: \(rtf < 1.0 ? "REAL-TIME CAPABLE" : "NOT REAL-TIME")")
    }

    // MARK: - Hidden Size Scaling

    func testHiddenSize512() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        let hiddenSize = 512
        let inputSize = 512
        let seqLen = 100
        let batchSize = 1

        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let result = try await runner.run(
            name: "LSTM_H512",
            parameters: ["hiddenSize": "\(hiddenSize)"]
        ) {
            _ = try await layer.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        let timestepsPerSec = Double(seqLen) / result.wallTimeStats.mean

        // Estimate GFLOPS: LSTM has ~8*H^2 + 8*H*I multiplications per timestep
        let flopsPerTimestep = Double(16 * hiddenSize * (hiddenSize + inputSize))
        let totalFlops = flopsPerTimestep * Double(seqLen)
        let gflops = totalFlops / result.wallTimeStats.mean / 1e9

        print("\n=== Large LSTM (H=512, seqLen=100) ===")
        print("Mean: \(result.wallTimeStats.formattedMean)")
        print("Throughput: \(String(format: "%.0f", timestepsPerSec)) timesteps/sec")
        print("Estimated GFLOPS: \(String(format: "%.2f", gflops))")
    }

    // MARK: - Batch Size Scaling

    func testBatchSize4() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: .quick)
        let engine = try LSTMEngine()

        let hiddenSize = 256
        let inputSize = 256
        let seqLen = 100
        let batchSize = 4

        // Create weights
        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let result = try await runner.run(
            name: "LSTM_batch4",
            parameters: ["batchSize": "\(batchSize)"]
        ) {
            _ = try await layer.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        let samplesPerSec = Double(batchSize * seqLen) / result.wallTimeStats.mean

        print("\n=== Batched LSTM (batch=4, H=256, seqLen=100) ===")
        print("Mean: \(result.wallTimeStats.formattedMean)")
        print("Throughput: \(String(format: "%.0f", samplesPerSec)) samples/sec")
    }

    // MARK: - Memory Usage

    func testMemoryUsage() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let runner = BenchmarkRunner(config: BenchmarkConfiguration(
            warmupIterations: 1,
            measureIterations: 3,
            removeOutliers: false,
            captureMemory: true,
            gcBetweenIterations: true
        ))

        let engine = try LSTMEngine()

        let hiddenSize = 256
        let inputSize = 256
        let batchSize = 1
        let seqLen = 500  // Longer sequence for memory test

        srand48(42)
        let weightIH = (0..<(4 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(4 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let result = try await runner.run(
            name: "LSTM_Memory_seq500",
            parameters: ["seqLen": "\(seqLen)", "hiddenSize": "\(hiddenSize)"]
        ) {
            _ = try await layer.forward(input, batchSize: batchSize, seqLen: seqLen)
        }

        print("\n=== Memory Usage (H=256, seqLen=500) ===")
        print("Mean Time: \(result.wallTimeStats.formattedMean)")
        if let peak = result.peakMemory {
            print("Peak Memory: \(ByteCountFormatter.string(fromByteCount: Int64(peak), countStyle: .memory))")
        }
        if let delta = result.memoryDelta {
            print("Memory Delta: \(delta.formattedAllocated)")
        }

        // Calculate theoretical memory usage
        let weightBytes = (weightIH.count + weightHH.count + biasIH.count + biasHH.count) * 4
        let inputBytes = input.count * 4
        let outputBytes = batchSize * seqLen * hiddenSize * 4
        let stateBytes = 2 * batchSize * hiddenSize * 4  // h and c

        print("\nTheoretical Minimum:")
        print("  Weights: \(ByteCountFormatter.string(fromByteCount: Int64(weightBytes), countStyle: .memory))")
        print("  Input: \(ByteCountFormatter.string(fromByteCount: Int64(inputBytes), countStyle: .memory))")
        print("  Output: \(ByteCountFormatter.string(fromByteCount: Int64(outputBytes), countStyle: .memory))")
        print("  State: \(ByteCountFormatter.string(fromByteCount: Int64(stateBytes), countStyle: .memory))")
    }

    // MARK: - Helpers

    private func formatTime(_ seconds: Double) -> String {
        if seconds < 0.001 {
            return String(format: "%.2f µs", seconds * 1_000_000)
        } else if seconds < 1.0 {
            return String(format: "%.2f ms", seconds * 1000)
        } else {
            return String(format: "%.2f s", seconds)
        }
    }
}
