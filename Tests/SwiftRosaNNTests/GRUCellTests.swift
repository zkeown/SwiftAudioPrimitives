//
//  GRUCellTests.swift
//  SwiftRosaNNTests
//
//  Unit tests for GRU cell operations.
//

import XCTest

@testable import SwiftRosaNN

final class GRUCellTests: XCTestCase {
    private let tolerance: Float = 1e-4

    // MARK: - GRU Cell Tests

    func testGRUCellCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3

        // Create simple weights (all 0.1)
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

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let cell = try await GRUCell(weights: weights, config: config, engine: engine)

        let cellInputSize = await cell.inputSize
        let cellHiddenSize = await cell.hiddenSize
        XCTAssertEqual(cellInputSize, inputSize)
        XCTAssertEqual(cellHiddenSize, hiddenSize)
    }

    func testGRUCellForwardZeroInput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 2

        // Create weights with zero bias
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

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let cell = try await GRUCell(weights: weights, config: config, engine: engine)

        // Zero input and zero state
        let input = [Float](repeating: 0, count: batchSize * inputSize)
        let hPrev = [Float](repeating: 0, count: batchSize * hiddenSize)

        let hOut = try await cell.forward(
            input: input,
            hPrev: hPrev,
            batchSize: batchSize
        )

        // With zero input, zero state, zero bias:
        // gates_ih = 0, gates_hh = 0
        // r = sigmoid(0 + 0) = 0.5
        // z = sigmoid(0 + 0) = 0.5
        // n = tanh(0 + 0.5 * 0) = tanh(0) = 0
        // h_new = (1 - 0.5) * 0 + 0.5 * 0 = 0

        XCTAssertEqual(hOut.count, batchSize * hiddenSize)

        for i in 0..<hOut.count {
            XCTAssertEqual(hOut[i], 0, accuracy: tolerance, "h[\(i)] should be 0")
        }
    }

    func testGRUCellForwardNonZeroInput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 1

        // Create weights with small values
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

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let cell = try await GRUCell(weights: weights, config: config, engine: engine)

        // Non-zero input, zero state
        let input = [Float](repeating: 1.0, count: batchSize * inputSize)
        let hPrev = [Float](repeating: 0, count: batchSize * hiddenSize)

        let hOut = try await cell.forward(
            input: input,
            hPrev: hPrev,
            batchSize: batchSize
        )

        // With input = [1,1,1], weights = 0.1, h_prev = 0:
        // gates_ih[each] = 0.1 * 3 = 0.3
        // gates_hh[each] = 0 (since h_prev = 0)
        // r = sigmoid(0.3 + 0) = sigmoid(0.3) ≈ 0.5744
        // z = sigmoid(0.3 + 0) = sigmoid(0.3) ≈ 0.5744
        // n = tanh(0.3 + r * 0) = tanh(0.3) ≈ 0.2913
        // h_new = (1 - z) * n + z * h_prev = (1 - 0.5744) * 0.2913 + 0.5744 * 0 ≈ 0.124

        XCTAssertEqual(hOut.count, batchSize * hiddenSize)

        // Check that output is non-zero and reasonable
        for h in hOut {
            XCTAssertTrue(abs(h) > 0.01, "Hidden state should be non-zero")
            XCTAssertTrue(abs(h) < 1.0, "Hidden state should be bounded")
        }
    }

    func testGRUCellOutputShapes() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 16
        let inputSize = 8
        let batchSize = 4

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

        let config = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let cell = try await GRUCell(weights: weights, config: config, engine: engine)

        // Random input
        srand48(42)
        let input = (0..<(batchSize * inputSize)).map { _ in Float(drand48() - 0.5) }
        let hPrev = (0..<(batchSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let hOut = try await cell.forward(
            input: input,
            hPrev: hPrev,
            batchSize: batchSize
        )

        XCTAssertEqual(hOut.count, batchSize * hiddenSize, "Hidden output shape mismatch")
    }

    // MARK: - Config Tests

    func testGRUConfigBasic() {
        let config = GRUConfig(inputSize: 256, hiddenSize: 512)

        XCTAssertEqual(config.inputSize, 256)
        XCTAssertEqual(config.hiddenSize, 512)
        XCTAssertEqual(config.numLayers, 1)
        XCTAssertFalse(config.bidirectional)
        XCTAssertTrue(config.bias)
        XCTAssertTrue(config.batchFirst)
        XCTAssertEqual(config.outputSize, 512)
        XCTAssertEqual(config.numDirections, 1)
        XCTAssertEqual(config.gateSize, 3 * 512)
    }

    func testGRUConfigBidirectional() {
        let config = GRUConfig(inputSize: 128, hiddenSize: 256, bidirectional: true)

        XCTAssertTrue(config.bidirectional)
        XCTAssertEqual(config.outputSize, 512)  // 2 * 256
        XCTAssertEqual(config.numDirections, 2)
        XCTAssertEqual(config.gateSize, 3 * 256)
    }

    func testGRUConfigBanquet() {
        let config = GRUConfig.banquet(inputSize: 128)

        XCTAssertEqual(config.inputSize, 128)
        XCTAssertEqual(config.hiddenSize, 256)
        XCTAssertTrue(config.bidirectional)
        XCTAssertEqual(config.outputSize, 512)
        XCTAssertEqual(config.gateSize, 3 * 256)
    }

    // MARK: - State Tests

    func testGRUStateZeros() {
        let state = GRUState.zeros(
            numLayers: 2,
            numDirections: 2,
            batchSize: 4,
            hiddenSize: 128
        )

        let expectedSize = 2 * 2 * 4 * 128
        XCTAssertEqual(state.hidden.count, expectedSize)
        XCTAssertTrue(state.hidden.allSatisfy { $0 == 0 })
    }

    func testGRULayerStateZeros() {
        let state = GRULayerState.zeros(batchSize: 2, hiddenSize: 64)

        XCTAssertEqual(state.hidden.count, 2 * 64)
        XCTAssertEqual(state.batchSize, 2)
        XCTAssertEqual(state.hiddenSize, 64)
    }

    // MARK: - Weights Tests

    func testGRULayerWeightsCreation() {
        let inputSize = 8
        let hiddenSize = 16

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

        XCTAssertEqual(weights.inputSize, inputSize)
        XCTAssertEqual(weights.hiddenSize, hiddenSize)
        XCTAssertEqual(weights.gateSize, 3 * hiddenSize)
        XCTAssertEqual(weights.weightIH.count, 3 * hiddenSize * inputSize)
    }

    // MARK: - GRU vs LSTM Gate Count Tests

    func testGRUHas3Gates() {
        let hiddenSize = 256
        let inputSize = 128

        let gruConfig = GRUConfig(inputSize: inputSize, hiddenSize: hiddenSize)
        let lstmConfig = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize)

        // GRU has 3 gates (reset, update, new)
        XCTAssertEqual(gruConfig.gateSize, 3 * hiddenSize)

        // This is different from LSTM which would have 4 gates
        // LSTMConfig doesn't expose gateSize, but weights would be 4*H
        XCTAssertEqual(gruConfig.gateSize, 768)  // 3 * 256
        XCTAssertEqual(lstmConfig.outputSize, hiddenSize)  // Both have same output size when unidirectional
    }
}
