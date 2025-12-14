//
//  GRULayer.swift
//  SwiftRosaNN
//
//  Unidirectional GRU layer for processing full sequences.
//

import Foundation
import MetalPerformanceShaders

/// Unidirectional GRU layer for processing full sequences.
///
/// Processes a sequence of inputs through a GRU, producing outputs at each timestep.
/// Can process forward or backward (for use in BiGRU).
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let layer = try await GRULayer(weights: layerWeights, config: config, engine: engine)
///
/// let (output, finalState) = try await layer.forward(
///     input,
///     batchSize: 1,
///     seqLen: 100
/// )
/// // output: [batchSize, seqLen, hiddenSize]
/// ```
public actor GRULayer {
    private let engine: LSTMEngine
    private let weights: GRULayerWeights
    private let config: GRUConfig
    private let reverse: Bool

    // Cached MPS weight matrices
    private let weightIHMatrix: MPSMatrix
    private let weightHHMatrix: MPSMatrix

    /// Create a GRU layer.
    ///
    /// - Parameters:
    ///   - weights: Layer weights.
    ///   - config: GRU configuration.
    ///   - engine: LSTMEngine for GPU operations.
    ///   - reverse: If true, process sequence in reverse order (for backward pass in BiGRU).
    public init(
        weights: GRULayerWeights,
        config: GRUConfig,
        engine: LSTMEngine,
        reverse: Bool = false
    ) async throws {
        self.weights = weights
        self.config = config
        self.engine = engine
        self.reverse = reverse

        // Create MPS weight matrices
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

    /// Process entire sequence.
    ///
    /// - Parameters:
    ///   - input: Input sequence [batchSize, seqLen, inputSize] (flattened row-major).
    ///   - batchSize: Batch dimension.
    ///   - seqLen: Sequence length.
    ///   - initialState: Optional initial state. If nil, zeros are used.
    /// - Returns: Tuple of (output sequence, final state).
    ///   Output shape: [batchSize, seqLen, hiddenSize] (flattened row-major).
    public func forward(
        _ input: [Float],
        batchSize: Int,
        seqLen: Int,
        initialState: GRULayerState? = nil
    ) async throws -> (output: [Float], finalState: GRULayerState) {
        let inputSize = weights.inputSize
        let hiddenSize = config.hiddenSize

        // Use optimized batched sequence API that groups multiple timesteps per command buffer
        // This reduces GPU sync points from seqLen to seqLen/16 (default batch size)
        let (output, finalHidden) = try await engine.gruSequenceForwardBatched(
            input: input,
            initialState: initialState?.hidden,
            weightIH: weightIHMatrix,
            weightHH: weightHHMatrix,
            biasIH: weights.biasIH,
            biasHH: weights.biasHH,
            batchSize: batchSize,
            seqLen: seqLen,
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            batchFirst: config.batchFirst,
            reverse: reverse,
            timestepsPerBatch: 16
        )

        let finalState = GRULayerState(
            hidden: finalHidden,
            batchSize: batchSize,
            hiddenSize: hiddenSize
        )

        return (output, finalState)
    }

    /// Input size for this layer.
    public var inputSize: Int {
        weights.inputSize
    }

    /// Hidden size for this layer.
    public var hiddenSize: Int {
        config.hiddenSize
    }

    /// Output size for this layer (same as hiddenSize for unidirectional).
    public var outputSize: Int {
        config.hiddenSize
    }
}
