//
//  LSTMLayerTests.swift
//  SwiftRosaNNTests
//
//  Unit tests for LSTM sequence processing.
//

import XCTest

@testable import SwiftRosaNN

final class LSTMLayerTests: XCTestCase {
    private let tolerance: Float = 1e-4

    // MARK: - LSTMLayer Tests

    func testLSTMLayerCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4

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

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let layerInputSize = await layer.inputSize
        let layerHiddenSize = await layer.hiddenSize
        let layerOutputSize = await layer.outputSize
        XCTAssertEqual(layerInputSize, inputSize)
        XCTAssertEqual(layerHiddenSize, hiddenSize)
        XCTAssertEqual(layerOutputSize, hiddenSize)
    }

    func testLSTMLayerForwardShapes() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4
        let batchSize = 2
        let seqLen = 5

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

        let config = LSTMConfig(inputSize: inputSize, hiddenSize: hiddenSize, batchFirst: true)
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

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

        // Check final state shape
        XCTAssertEqual(finalState.batchSize, batchSize)
        XCTAssertEqual(finalState.hiddenSize, hiddenSize)
        XCTAssertEqual(finalState.hidden.count, batchSize * hiddenSize)
        XCTAssertEqual(finalState.cell.count, batchSize * hiddenSize)
    }

    func testLSTMLayerSequenceConsistency() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 4
        let inputSize = 3
        let batchSize = 1
        let seqLen = 3

        srand48(123)
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

        // Process with LSTMLayer
        let layer = try await LSTMLayer(weights: weights, config: config, engine: engine)

        let input = (0..<(batchSize * seqLen * inputSize)).map { _ in Float(drand48() - 0.5) }
        let (layerOutput, _) = try await layer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen
        )

        // Process step by step with LSTMCell
        let cell = try await LSTMCell(weights: weights, config: config, engine: engine)

        var h = [Float](repeating: 0, count: batchSize * hiddenSize)
        var c = [Float](repeating: 0, count: batchSize * hiddenSize)
        var cellOutputs: [[Float]] = []

        for t in 0..<seqLen {
            let inputSlice = Array(input[(t * inputSize)..<((t + 1) * inputSize)])
            let (hNew, cNew) = try await cell.forward(
                input: inputSlice,
                hPrev: h,
                cPrev: c,
                batchSize: batchSize
            )
            h = hNew
            c = cNew
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

    // MARK: - BiLSTM Tests

    func testBiLSTMLayerCreation() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4

        let weightIH = [Float](repeating: 0.1, count: 4 * hiddenSize * inputSize)
        let weightHH = [Float](repeating: 0.1, count: 4 * hiddenSize * hiddenSize)
        let biasIH = [Float](repeating: 0.0, count: 4 * hiddenSize)
        let biasHH = [Float](repeating: 0.0, count: 4 * hiddenSize)

        let forwardWeights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )

        let backwardWeights = LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
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

        let actualInputSize = await bilstm.inputSize
        let actualHiddenSize = await bilstm.hiddenSize
        let actualOutputSize = await bilstm.outputSize
        XCTAssertEqual(actualInputSize, inputSize)
        XCTAssertEqual(actualHiddenSize, hiddenSize)
        XCTAssertEqual(actualOutputSize, hiddenSize * 2)
    }

    func testBiLSTMLayerOutputShape() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let hiddenSize = 8
        let inputSize = 4
        let batchSize = 2
        let seqLen = 5

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
        let weightHHBack =
            (0..<(4 * hiddenSize * hiddenSize)).map { _ in Float(drand48() - 0.5) * 0.1 }

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

        let (output, finalStates) = try await bilstm.forward(
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

    // MARK: - Reverse Sequence Tests

    func testReverseSequence() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let batchSize = 2
        let seqLen = 4
        let featureSize = 3

        // Create a simple input where we can verify reversal
        // [batch, seq, feature] - each timestep has distinct values
        var input = [Float](repeating: 0, count: batchSize * seqLen * featureSize)
        for b in 0..<batchSize {
            for t in 0..<seqLen {
                for f in 0..<featureSize {
                    let idx = b * seqLen * featureSize + t * featureSize + f
                    input[idx] = Float(t * 10 + f)  // t=0: [0,1,2], t=1: [10,11,12], etc.
                }
            }
        }

        let reversed = try await engine.reverseSequence(
            input,
            batchSize: batchSize,
            seqLen: seqLen,
            featureSize: featureSize
        )

        // Verify reversal: t=0 should now have t=3's values, etc.
        for b in 0..<batchSize {
            for t in 0..<seqLen {
                let originalT = seqLen - 1 - t
                for f in 0..<featureSize {
                    let idx = b * seqLen * featureSize + t * featureSize + f
                    let expectedVal = Float(originalT * 10 + f)
                    XCTAssertEqual(
                        reversed[idx],
                        expectedVal,
                        accuracy: tolerance,
                        "Reversal mismatch at b=\(b), t=\(t), f=\(f)"
                    )
                }
            }
        }
    }

    // MARK: - Concatenation Tests

    func testConcatenateBidirectional() async throws {
        guard LSTMEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try LSTMEngine()
        let batchSize = 2
        let seqLen = 3
        let hiddenSize = 4

        // Forward: fill with 1s
        let forward = [Float](repeating: 1.0, count: batchSize * seqLen * hiddenSize)
        // Backward: fill with 2s
        let backward = [Float](repeating: 2.0, count: batchSize * seqLen * hiddenSize)

        let concatenated = try await engine.concatenateBidirectional(
            forward: forward,
            backward: backward,
            batchSize: batchSize,
            seqLen: seqLen,
            hiddenSize: hiddenSize
        )

        XCTAssertEqual(concatenated.count, batchSize * seqLen * hiddenSize * 2)

        // Verify concatenation: first half should be 1s, second half should be 2s
        for b in 0..<batchSize {
            for t in 0..<seqLen {
                for h in 0..<(hiddenSize * 2) {
                    let idx = b * seqLen * hiddenSize * 2 + t * hiddenSize * 2 + h
                    let expected: Float = h < hiddenSize ? 1.0 : 2.0
                    XCTAssertEqual(
                        concatenated[idx],
                        expected,
                        accuracy: tolerance,
                        "Concat mismatch at b=\(b), t=\(t), h=\(h)"
                    )
                }
            }
        }
    }
}
