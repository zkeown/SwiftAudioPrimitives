//
//  BiLSTMLayer.swift
//  SwiftRosaNN
//
//  Bidirectional LSTM layer.
//

import Foundation

/// Bidirectional LSTM layer.
///
/// Processes sequence in both directions and concatenates outputs.
/// Output size is 2 * hiddenSize.
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let bilstm = try await BiLSTMLayer(weights: weights, engine: engine)
///
/// let (output, states) = try await bilstm.forward(
///     input,
///     batchSize: 1,
///     seqLen: 100
/// )
/// // output: [batchSize, seqLen, 2*hiddenSize]
/// ```
public actor BiLSTMLayer {
    private let forwardLayer: LSTMLayer
    private let backwardLayer: LSTMLayer
    private let engine: LSTMEngine
    private let config: LSTMConfig

    /// Create a bidirectional LSTM layer.
    ///
    /// - Parameters:
    ///   - weights: Complete LSTM weights (must include backward weights).
    ///   - engine: LSTMEngine for GPU operations.
    /// - Throws: If weights are not bidirectional or initialization fails.
    public init(weights: LSTMWeights, engine: LSTMEngine) async throws {
        precondition(weights.config.bidirectional, "Weights must be for bidirectional LSTM")
        precondition(weights.config.numLayers == 1, "BiLSTMLayer supports single layer only")

        self.config = weights.config
        self.engine = engine

        self.forwardLayer = try await LSTMLayer(
            weights: weights.forward[0],
            config: weights.config,
            engine: engine,
            reverse: false
        )

        self.backwardLayer = try await LSTMLayer(
            weights: weights.backward![0],
            config: weights.config,
            engine: engine,
            reverse: true
        )
    }

    /// Create a bidirectional LSTM layer from separate forward/backward weights.
    ///
    /// - Parameters:
    ///   - forwardWeights: Weights for forward direction.
    ///   - backwardWeights: Weights for backward direction.
    ///   - config: LSTM configuration (must have bidirectional=true).
    ///   - engine: LSTMEngine for GPU operations.
    public init(
        forwardWeights: LSTMLayerWeights,
        backwardWeights: LSTMLayerWeights,
        config: LSTMConfig,
        engine: LSTMEngine
    ) async throws {
        precondition(config.bidirectional, "Config must specify bidirectional")

        self.config = config
        self.engine = engine

        self.forwardLayer = try await LSTMLayer(
            weights: forwardWeights,
            config: config,
            engine: engine,
            reverse: false
        )

        self.backwardLayer = try await LSTMLayer(
            weights: backwardWeights,
            config: config,
            engine: engine,
            reverse: true
        )
    }

    /// Process sequence bidirectionally.
    ///
    /// - Parameters:
    ///   - input: Input sequence [batchSize, seqLen, inputSize] (flattened row-major).
    ///   - batchSize: Batch dimension.
    ///   - seqLen: Sequence length.
    ///   - initialState: Optional initial states for both directions.
    /// - Returns: Tuple of (output, final states).
    ///   Output shape: [batchSize, seqLen, 2*hiddenSize] (flattened row-major).
    public func forward(
        _ input: [Float],
        batchSize: Int,
        seqLen: Int,
        initialState: (forward: LSTMLayerState?, backward: LSTMLayerState?)? = nil
    ) async throws -> (
        output: [Float],
        finalState: (forward: LSTMLayerState, backward: LSTMLayerState)
    ) {
        // Run forward and backward in parallel
        async let forwardResult = forwardLayer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen,
            initialState: initialState?.forward
        )

        async let backwardResult = backwardLayer.forward(
            input,
            batchSize: batchSize,
            seqLen: seqLen,
            initialState: initialState?.backward
        )

        let (fwdOutput, fwdState) = try await forwardResult
        let (bwdOutput, bwdState) = try await backwardResult

        // Concatenate outputs along feature dimension using Metal
        let concatenated = try await engine.concatenateBidirectional(
            forward: fwdOutput,
            backward: bwdOutput,
            batchSize: batchSize,
            seqLen: seqLen,
            hiddenSize: config.hiddenSize
        )

        return (concatenated, (fwdState, bwdState))
    }

    /// Input size for this layer.
    public var inputSize: Int {
        config.inputSize
    }

    /// Hidden size per direction.
    public var hiddenSize: Int {
        config.hiddenSize
    }

    /// Output size (2 * hiddenSize for bidirectional).
    public var outputSize: Int {
        config.hiddenSize * 2
    }
}
