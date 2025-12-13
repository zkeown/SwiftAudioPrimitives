//
//  GRUCell.swift
//  SwiftRosaNN
//
//  Single-timestep GRU cell with Metal acceleration.
//

import Foundation
import MetalPerformanceShaders

/// Single-timestep GRU cell with Metal acceleration.
///
/// Processes one timestep of input and produces new hidden state.
/// GRU differs from LSTM in having 3 gates (reset, update, new) and no cell state.
/// Uses MPS for matrix multiplications and custom Metal kernels for state updates.
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let cell = try await GRUCell(weights: layerWeights, config: config, engine: engine)
///
/// var state = GRULayerState.zeros(batchSize: 1, hiddenSize: 256)
/// for t in 0..<seqLen {
///     let x_t = inputSlice(at: t)
///     state = try await cell.forward(input: x_t, state: state)
/// }
/// ```
public actor GRUCell {
    private let engine: LSTMEngine
    private let weights: GRULayerWeights
    private let config: GRUConfig

    // Cached MPS weight matrices
    private let weightIHMatrix: MPSMatrix
    private let weightHHMatrix: MPSMatrix

    /// Create a GRU cell with pre-loaded weights.
    ///
    /// - Parameters:
    ///   - weights: Layer weights for this cell.
    ///   - config: GRU configuration.
    ///   - engine: LSTMEngine for GPU operations.
    public init(weights: GRULayerWeights, config: GRUConfig, engine: LSTMEngine) async throws {
        self.weights = weights
        self.config = config
        self.engine = engine

        // Create MPS weight matrices (stored as [3*H, inputSize] and [3*H, hiddenSize])
        self.weightIHMatrix = try await engine.createWeightMatrix(
            weights.weightIH,
            rows: 3 * config.hiddenSize,
            cols: weights.inputSize
        )
        self.weightHHMatrix = try await engine.createWeightMatrix(
            weights.weightHH,
            rows: 3 * config.hiddenSize,
            cols: config.hiddenSize
        )
    }

    /// Process a single timestep.
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - state: Previous hidden state.
    /// - Returns: New state after processing this timestep.
    public func forward(
        input: [Float],
        state: GRULayerState
    ) async throws -> GRULayerState {
        let hNew = try await engine.gruCellForward(
            input: input,
            hPrev: state.hidden,
            weightIH: weightIHMatrix,
            weightHH: weightHHMatrix,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH,
            batchSize: state.batchSize,
            inputSize: weights.inputSize,
            hiddenSize: config.hiddenSize
        )

        return GRULayerState(
            hidden: hNew,
            batchSize: state.batchSize,
            hiddenSize: config.hiddenSize
        )
    }

    /// Process a single timestep with explicit hidden state.
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - hPrev: Previous hidden state [batchSize, hiddenSize].
    ///   - batchSize: Batch dimension.
    /// - Returns: New hidden state.
    public func forward(
        input: [Float],
        hPrev: [Float],
        batchSize: Int
    ) async throws -> [Float] {
        return try await engine.gruCellForward(
            input: input,
            hPrev: hPrev,
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
