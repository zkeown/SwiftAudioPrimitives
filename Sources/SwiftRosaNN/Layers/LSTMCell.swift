//
//  LSTMCell.swift
//  SwiftRosaNN
//
//  Single-timestep LSTM cell with Metal acceleration.
//

import Foundation
import MetalPerformanceShaders

/// Single-timestep LSTM cell with Metal acceleration.
///
/// Processes one timestep of input and produces new hidden and cell states.
/// Uses MPS for matrix multiplications and custom Metal kernels for state updates.
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let cell = try await LSTMCell(weights: layerWeights, config: config, engine: engine)
///
/// var state = LSTMLayerState.zeros(batchSize: 1, hiddenSize: 256)
/// for t in 0..<seqLen {
///     let x_t = inputSlice(at: t)
///     state = try await cell.forward(input: x_t, state: state)
/// }
/// ```
public actor LSTMCell {
    private let engine: LSTMEngine
    private let weights: LSTMLayerWeights
    private let config: LSTMConfig

    // Cached MPS weight matrices
    private let weightIHMatrix: MPSMatrix
    private let weightHHMatrix: MPSMatrix

    /// Create an LSTM cell with pre-loaded weights.
    ///
    /// - Parameters:
    ///   - weights: Layer weights for this cell.
    ///   - config: LSTM configuration.
    ///   - engine: LSTMEngine for GPU operations.
    public init(weights: LSTMLayerWeights, config: LSTMConfig, engine: LSTMEngine) async throws {
        self.weights = weights
        self.config = config
        self.engine = engine

        // Create MPS weight matrices (stored as [4*H, inputSize] and [4*H, hiddenSize])
        self.weightIHMatrix = try await engine.createWeightMatrix(
            weights.weightIH,
            rows: 4 * config.hiddenSize,
            cols: weights.inputSize
        )
        self.weightHHMatrix = try await engine.createWeightMatrix(
            weights.weightHH,
            rows: 4 * config.hiddenSize,
            cols: config.hiddenSize
        )
    }

    /// Process a single timestep.
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - state: Previous hidden and cell state.
    /// - Returns: New state after processing this timestep.
    public func forward(
        input: [Float],
        state: LSTMLayerState
    ) async throws -> LSTMLayerState {
        let (hNew, cNew) = try await engine.lstmCellForward(
            input: input,
            hPrev: state.hidden,
            cPrev: state.cell,
            weightIH: weightIHMatrix,
            weightHH: weightHHMatrix,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH,
            batchSize: state.batchSize,
            inputSize: weights.inputSize,
            hiddenSize: config.hiddenSize
        )

        return LSTMLayerState(
            hidden: hNew,
            cell: cNew,
            batchSize: state.batchSize,
            hiddenSize: config.hiddenSize
        )
    }

    /// Process a single timestep with explicit hidden and cell states.
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - hPrev: Previous hidden state [batchSize, hiddenSize].
    ///   - cPrev: Previous cell state [batchSize, hiddenSize].
    ///   - batchSize: Batch dimension.
    /// - Returns: Tuple of (new hidden state, new cell state).
    public func forward(
        input: [Float],
        hPrev: [Float],
        cPrev: [Float],
        batchSize: Int
    ) async throws -> (h: [Float], c: [Float]) {
        return try await engine.lstmCellForward(
            input: input,
            hPrev: hPrev,
            cPrev: cPrev,
            weightIH: weightIHMatrix,
            weightHH: weightHHMatrix,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH,
            batchSize: batchSize,
            inputSize: weights.inputSize,
            hiddenSize: config.hiddenSize
        )
    }

    /// Input size for this cell.
    public var inputSize: Int {
        weights.inputSize
    }

    /// Hidden size for this cell.
    public var hiddenSize: Int {
        config.hiddenSize
    }
}
