//
//  LSTMCellTests.swift
//  SwiftRosaNNTests
//
//  Unit tests for LSTM cell operations.
//

import XCTest

@testable import SwiftRosaNN

final class LSTMCellTests: XCTestCase {
    private let tolerance: Float = 1e-4

    // MARK: - LSTMEngine Tests

    func testEngineAvailability() {
        // Metal should be available on macOS
        #if os(macOS) || os(iOS)
            XCTAssertTrue(LSTMEngine.isAvailable, "LSTMEngine should be available on Apple platforms")
        #endif
    }

    func testEngineCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        XCTAssertNotNil(engine)
    }

    // MARK: - Single Cell Tests

    func testLSTMCellCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3

        // Create simple weights (all 0.1)
        let weightIH = [Float](repeating: 0.1, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 4 * hiddenSize)

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

        let cellInputSize = await cell.inputSize
        let cellHiddenSize = await cell.hiddenSize
        XCTAssertEqual(cellInputSize, inputSize)
        XCTAssertEqual(cellHiddenSize, hiddenSize)
    }

    func testLSTMCellForwardZeroInput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 2

        // Create weights with zero bias
        let weightIH = [Float](repeating: 0.1, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 4 * hiddenSize)

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

        // Zero input and zero state
        let input = [Float](repeating: 0, count: batchSize * inputSize)
        let hPrev = [Float](repeating: 0, count: batchSize * hiddenSize)
        let cPrev = [Float](repeating: 0, count: batchSize * hiddenSize)

        let (hOut, cOut) = try await cell.forward(
            input: input,
            hPrev: hPrev,
            cPrev: cPrev,
            batchSize: batchSize
        )

        // With zero input, zero state, zero bias:
        // gates = 0, so i = f = o = sigmoid(0) = 0.5, g = tanh(0) = 0
        // c_new = f * c_prev + i * g = 0.5 * 0 + 0.5 * 0 = 0
        // h_new = o * tanh(c_new) = 0.5 * 0 = 0

        XCTAssertEqual(hOut.count, batchSize * hiddenSize)
        XCTAssertEqual(cOut.count, batchSize * hiddenSize)

        for i in 0..<hOut.count {
            XCTAssertEqual(hOut[i], 0, accuracy: tolerance, "h[\(i)] should be 0")
            XCTAssertEqual(cOut[i], 0, accuracy: tolerance, "c[\(i)] should be 0")
        }
    }

    func testLSTMCellForwardNonZeroInput() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 1

        // Create weights with small values
        let weightIH = [Float](repeating: 0.1, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 4 * hiddenSize)

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

        // Non-zero input, zero state
        let input = [Float](repeating: 1.0, count: batchSize * inputSize)
        let hPrev = [Float](repeating: 0, count: batchSize * hiddenSize)
        let cPrev = [Float](repeating: 0, count: batchSize * hiddenSize)

        let (hOut, cOut) = try await cell.forward(
            input: input,
            hPrev: hPrev,
            cPrev: cPrev,
            batchSize: batchSize
        )

        // With input = [1,1,1], weights = 0.1:
        // W_ih @ x = 0.1 * 3 = 0.3 for each gate element
        // All gates have pre-activation 0.3
        // i = f = o = sigmoid(0.3) ≈ 0.5744
        // g = tanh(0.3) ≈ 0.2913
        // c_new = f * 0 + i * g ≈ 0.5744 * 0.2913 ≈ 0.1673
        // h_new = o * tanh(c_new) ≈ 0.5744 * tanh(0.1673) ≈ 0.5744 * 0.1658 ≈ 0.0952

        XCTAssertEqual(hOut.count, batchSize * hiddenSize)
        XCTAssertEqual(cOut.count, batchSize * hiddenSize)

        // Check that output is non-zero and reasonable
        for h in hOut {
            XCTAssertTrue(abs(h) > 0.01, "Hidden state should be non-zero")
            XCTAssertTrue(abs(h) < 1.0, "Hidden state should be bounded")
        }
        for c in cOut {
            XCTAssertTrue(abs(c) > 0.01, "Cell state should be non-zero")
        }
    }

    func testLSTMCellOutputShapes() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 16
        let inputSize = 8
        let batchSize = 4

        let weightIH = [Float](repeating: 0.01, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.01, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.1, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.1, count: 4 * hiddenSize)

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

        // Random input
        srand48(42)
        let input = (0..<(batchSize * inputSize)).map { _ in Float(drand48() - 0.5) }
        let hPrev = (0..<(batchSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }
        let cPrev = (0..<(batchSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

        let (hOut, cOut) = try await cell.forward(
            input: input,
            hPrev: hPrev,
            cPrev: cPrev,
            batchSize: batchSize
        )

        XCTAssertEqual(hOut.count, batchSize * hiddenSize, "Hidden output shape mismatch")
        XCTAssertEqual(cOut.count, batchSize * hiddenSize, "Cell output shape mismatch")
    }

    // MARK: - Config Tests

    func testLSTMConfigBasic() {
        let config = LSTMConfig(inputSize: 256, hiddenSize: 512)

        XCTAssertEqual(config.inputSize, 256)
        XCTAssertEqual(config.hiddenSize, 512)
        XCTAssertEqual(config.numLayers, 1)
        XCTAssertFalse(config.bidirectional)
        XCTAssertTrue(config.bias)
        XCTAssertTrue(config.batchFirst)
        XCTAssertEqual(config.outputSize, 512)
        XCTAssertEqual(config.numDirections, 1)
    }

    func testLSTMConfigBidirectional() {
        let config = LSTMConfig(inputSize: 128, hiddenSize: 256, bidirectional: true)

        XCTAssertTrue(config.bidirectional)
        XCTAssertEqual(config.outputSize, 512)  // 2 * 256
        XCTAssertEqual(config.numDirections, 2)
    }

    func testLSTMConfigBanquet() {
        let config = LSTMConfig.banquet(inputSize: 128)

        XCTAssertEqual(config.inputSize, 128)
        XCTAssertEqual(config.hiddenSize, 256)
        XCTAssertTrue(config.bidirectional)
        XCTAssertEqual(config.outputSize, 512)
    }

    // MARK: - State Tests

    func testLSTMStateZeros() {
        let state = LSTMState.zeros(
            numLayers: 2,
            numDirections: 2,
            batchSize: 4,
            hiddenSize: 128
        )

        let expectedSize = 2 * 2 * 4 * 128
        XCTAssertEqual(state.hidden.count, expectedSize)
        XCTAssertEqual(state.cell.count, expectedSize)
        XCTAssertTrue(state.hidden.allSatisfy { $0 == 0 })
        XCTAssertTrue(state.cell.allSatisfy { $0 == 0 })
    }

    func testLSTMLayerStateZeros() {
        let state = LSTMLayerState.zeros(batchSize: 2, hiddenSize: 64)

        XCTAssertEqual(state.hidden.count, 2 * 64)
        XCTAssertEqual(state.cell.count, 2 * 64)
        XCTAssertEqual(state.batchSize, 2)
        XCTAssertEqual(state.hiddenSize, 64)
    }

    // MARK: - Weights Tests

    func testLSTMLayerWeightsCreation() {
        let inputSize = 8
        let hiddenSize = 16

        let weightIH = [Float](repeating: 0.1, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 4 * hiddenSize)

        let weights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        XCTAssertEqual(weights.inputSize, inputSize)
        XCTAssertEqual(weights.hiddenSize, hiddenSize)
        XCTAssertEqual(weights.gateSize, 4 * hiddenSize)
        XCTAssertEqual(weights.weightIH.count, 4 * hiddenSize * inputSize)
    }
}
