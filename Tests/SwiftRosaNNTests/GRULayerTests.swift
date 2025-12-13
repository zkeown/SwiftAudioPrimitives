//
//  GRULayerTests.swift
//  SwiftRosaNNTests
//
//  Unit tests for GRU sequence processing.
//

import XCTest

@testable import SwiftRosaNN

final class GRULayerTests: XCTestCase {
    private let tolerance: Float = 1e-4

    // MARK: - GRULayer Tests

    func testGRULayerCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4

        let weightIH = [Float](repeating: 0.1, count: 3 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 3 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 3 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 3 * hiddenSize)

        let weights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await GRULayer(weights: weights, config: config, engine: engine)

        let layerInputSize = await layer.inputSize
        let layerHiddenSize = await layer.hiddenSize
        let layerOutputSize = await layer.outputSize
        XCTAssertEqual(layerInputSize, inputSize)
        XCTAssertEqual(layerHiddenSize, hiddenSize)
        XCTAssertEqual(layerOutputSize, hiddenSize)
    }

    func testGRULayerForwardShapes() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4
        let batchSize = 2
        let seqLen = 5

        let weightIH = [Float](repeating: 0.01, count: 3 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.01, count: 3 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.1, count: 3 * hiddenSize)
        let biasHH = [Float](repeating: 0.1, count: 3 * hiddenSize)

        let weights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await GRULayer(weights: weights, config: config, engine: engine)

        // Create input: [batchSize, seqLen, inputSize]
        srand48(42)
        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let (output, finalState) = try await layer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Check output shape: [batchSize, seqLen, hiddenSize]
        XCTAssertEqual(output.count, batchSize * seqLen * hiddenSize)

        // Check final state shape - GRU only has hidden state, no cell state
        XCTAssertEqual(finalState.batchSize, batchSize)
        XCTAssertEqual(finalState.hiddenSize, hiddenSize)
        XCTAssertEqual(finalState.hidden.count, batchSize * hiddenSize)
    }

    func testGRULayerSequenceConsistency() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 1
        let seqLen = 3

        srand48(123)
        let weightIH = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)

        // Process with GRULayer
        let layer = try await GRULayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }
        let (layerOutput, _) = try await layer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Process step by step with GRUCell
        let cell = try await GRUCell(weights: weights, config: config, engine: engine)

        var h = [Float](repeating: 0, count: batchSize * hiddenSize)
        var cellOutputs: [[Float]] = []

        for t in 0..<seqLen {
            let inputSlice = Array(input[(t * inputSize)..<((t + 1) * inputSize)])
            let hNew = try await cell.forward(
                input: inputSlice,
                hPrev: h,
                batchSize: batchSize
            )
            h = hNew
            cellOutputs.append(h)
        }

        // Compare outputs
        for t in 0..<seqLen {
            for i in 0..<hiddenSize {
                let layerVal = layerOutput[t * hiddenSize + i]
                let cellVal = cellOutputs[t][i]
                XCTAssertEqual(
                    layerVal,
                    cellVal,
                    accuracy: tolerance,
                    "Mismatch at t=\(t), h=\(i)"
                )
            }
        }
    }

    // MARK: - BiGRU Tests

    func testBiGRULayerCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4

        let weightIH = [Float](repeating: 0.1, count: 3 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 3 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 3 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 3 * hiddenSize)

        let forwardWeights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let backwardWeights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            batchFirst: true
        )

        let bigru = try await BiGRULayer(
            forwardWeights: forwardWeights,
            backwardWeights: backwardWeights,
            config: config,
            engine: engine
        )

        let actualInputSize = await bigru.inputSize
        let actualHiddenSize = await bigru.hiddenSize
        let actualOutputSize = await bigru.outputSize
        XCTAssertEqual(actualInputSize, inputSize)
        XCTAssertEqual(actualHiddenSize, hiddenSize)
        XCTAssertEqual(actualOutputSize, hiddenSize * 2)
    }

    func testBiGRULayerOutputShape() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4
        let batchSize = 2
        let seqLen = 5

        srand48(42)
        let weightIH = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHH = (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let forwardWeights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        // Different weights for backward
        let weightIHBack = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHHBack =
            (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let backwardWeights = GRULayerWeights(
            weightIH: weightIHBack,
            weightHH: weightHHBack,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            batchFirst: true
        )

        let bigru = try await BiGRULayer(
            forwardWeights: forwardWeights,
            backwardWeights: backwardWeights,
            config: config,
            engine: engine
        )

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let (output, finalStates) = try await bigru.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Output should be [batchSize, seqLen, 2*hiddenSize]
        XCTAssertEqual(output.count, batchSize * seqLen * hiddenSize * 2)

        // Both forward and backward states should be returned
        XCTAssertEqual(finalStates.forward.batchSize, batchSize)
        XCTAssertEqual(finalStates.backward.batchSize, batchSize)
    }

    func testBiGRULayerWithGRUWeights() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4
        let batchSize = 1
        let seqLen = 3

        srand48(456)
        let weightIHFwd = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHHFwd = (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIHFwd = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHHFwd = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let forwardWeights = GRULayerWeights(
            weightIH: weightIHFwd,
            weightHH: weightHHFwd,
            biasIH: biasIHFwd,
            biasHH: biasHHFwd,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let weightIHBwd = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let weightHHBwd = (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasIHBwd = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHHBwd = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let backwardWeights = GRULayerWeights(
            weightIH: weightIHBwd,
            weightHH: weightHHBwd,
            biasIH: biasIHBwd,
            biasHH: biasHHBwd,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            bidirectional: true,
            batchFirst: true
        )

        // Create GRUWeights container
        let gruWeights = GRUWeights(
            forward: [forwardWeights],
            backward: [backwardWeights],
            config: config
        )

        // Create BiGRU layer using GRUWeights
        let bigru = try await BiGRULayer(weights: gruWeights, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let (output, _) = try await bigru.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Verify output shape
        XCTAssertEqual(output.count, batchSize * seqLen * hiddenSize * 2)

        // Check that output values are reasonable
        for value in output {
            XCTAssertTrue(abs(value) < 10.0, "Output value should be bounded")
        }
    }

    // MARK: - GRU Banquet Config Tests

    func testGRUBanquetConfig() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        // Test Banquet-style configuration for GRU
        // Banquet uses H=256, bidirectional, input size varies by layer
        let config = GRUConfig.banquet(inputSize: 128)

        XCTAssertEqual(config.inputSize, 128)
        XCTAssertEqual(config.hiddenSize, 256)
        XCTAssertTrue(config.bidirectional)
        XCTAssertEqual(config.outputSize, 512)  // 2 * 256
        XCTAssertEqual(config.gateSize, 768)    // 3 * 256

        // Verify the weight shape expectations
        let expectedWeightIHSize = 3 * 256 * 128  // [768, 128]
        let expectedWeightHHSize = 3 * 256 * 256  // [768, 256]
        XCTAssertEqual(expectedWeightIHSize, 768 * 128)
        XCTAssertEqual(expectedWeightHHSize, 768 * 256)
    }

    // MARK: - GRU Output Non-Zero Tests

    func testGRUProducesNonZeroOutput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 16
        let inputSize = 8
        let batchSize = 1
        let seqLen = 10

        srand48(789)
        let weightIH = (0..<(3 * hiddenSize * inputSize)).map { _ in Float(drand48() - 0.5) * 0.2 }
        let weightHH = (0..<(3 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.2 }
        let biasIH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let biasHH = (0..<(3 * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let weights = GRULayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let layer = try await GRULayer(weights: weights, config: config, engine: engine)

        // Create non-zero input
        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }

        let (output, finalState) = try await layer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Check that output is non-zero
        let outputSum = output.reduce(0) { $0 + abs($1) }
        XCTAssertTrue(outputSum > 0, "GRU output should be non-zero")

        // Check that final state is non-zero
        let stateSum = finalState.hidden.reduce(0) { $0 + abs($1) }
        XCTAssertTrue(stateSum > 0, "GRU final state should be non-zero")
    }
}
