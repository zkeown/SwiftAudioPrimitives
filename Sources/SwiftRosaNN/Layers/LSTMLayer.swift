//
//  LSTMLayer.swift
//  SwiftRosaNN
//
//  Unidirectional LSTM layer for processing full sequences.
//

import Foundation
import MetalPerformanceShaders

/// Unidirectional LSTM layer for processing full sequences.
///
/// Processes a sequence of inputs through an LSTM, producing outputs at each timestep.
/// Can process forward or backward (for use in BiLSTM).
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let layer = try await LSTMLayer(weights: layerWeights, config: config, engine: engine)
///
/// let (output, finalState) = try await layer.forward(
///     input,
///     batchSize: 1,
///     seqLen: 100
/// )
/// // output: [batchSize, seqLen, hiddenSize]
/// ```
public actor LSTMLayer {
    private let engine: LSTMEngine
    private let weights: LSTMLayerWeights
    private let config: LSTMConfig
    private let reverse: Bool

    // Cached MPS weight matrices
    private let weightIHMatrix: MPSMatrix
    private let weightHHMatrix: MPSMatrix

    /// Create an LSTM layer.
    ///
    /// - Parameters:
    ///   - weights: Layer weights.
    ///   - config: LSTM configuration.
    ///   - engine: LSTMEngine for GPU operations.
    ///   - reverse: If true, process sequence in reverse order (for backward pass in BiLSTM).
    public init(
        weights: LSTMLayerWeights,
        config: LSTMConfig,
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
            rows: 4 * config.hiddenSize,
            cols: weights.inputSize
        )
        self.weightHHMatrix = try await engine.createWeightMatrix(
            weights.weightHH,
            rows: 4 * config.hiddenSize,
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
        initialState: LSTMLayerState? = nil
    ) async throws -> (output: [Float], finalState: LSTMLayerState) {
        let inputSize = weights.inputSize
        let hiddenSize = config.hiddenSize

        // Initialize output buffer
        let outputSize = batchSize * seqLen * hiddenSize
        var output = [Float](repeating: 0, count: outputSize)

        // Initialize state
        var h = initialState?.hidden ?? [Float](repeating: 0, count: batchSize * hiddenSize)
        var c = initialState?.cell ?? [Float](repeating: 0, count: batchSize * hiddenSize)

        // Determine iteration order
        let timeIndices: [Int]
        if reverse {
            timeIndices = (0..<seqLen).reversed().map { $0 }
        } else {
            timeIndices = Array(0..<seqLen)
        }

        // Process each timestep
        for (outputT, inputT) in timeIndices.enumerated() {
            // Extract input slice for this timestep
            // Input layout: [batch, seq, features]
            let inputStart = inputT * batchSize * inputSize
            let inputSlice: [Float]

            // For efficiency, we need to handle the memory layout properly
            // If batchFirst: input is [batch, seq, input] so slice is straightforward
            if config.batchFirst {
                // For batch-first: data at timestep t for all batches is at
                // [b * seqLen * inputSize + t * inputSize] for each batch b
                // This is non-contiguous, so we need to gather
                var slice = [Float](repeating: 0, count: batchSize * inputSize)
                for b in 0..<batchSize {
                    let srcOffset = b * seqLen * inputSize + inputT * inputSize
                    let dstOffset = b * inputSize
                    for i in 0..<inputSize {
                        slice[dstOffset + i] = input[srcOffset + i]
                    }
                }
                inputSlice = slice
            } else {
                // Sequence-first: data is contiguous per timestep
                inputSlice = Array(input[inputStart..<(inputStart + batchSize * inputSize)])
            }

            // Process through cell
            let (hNew, cNew) = try await engine.lstmCellForward(
                input: inputSlice,
                hPrev: h,
                cPrev: c,
                weightIH: weightIHMatrix,
                weightHH: weightHHMatrix,
                biasIH: weights.biasIH,
                biasHH: weights.biasHH,
                batchSize: batchSize,
                inputSize: inputSize,
                hiddenSize: hiddenSize
            )

            h = hNew
            c = cNew

            // Store output for this timestep
            // For reverse processing, we still store in forward order in output
            // (backward LSTM output gets reversed later or stored in reverse position)
            let outT = reverse ? (seqLen - 1 - outputT) : outputT

            if config.batchFirst {
                // Output layout: [batch, seq, hidden]
                for b in 0..<batchSize {
                    let dstOffset = b * seqLen * hiddenSize + outT * hiddenSize
                    let srcOffset = b * hiddenSize
                    for i in 0..<hiddenSize {
                        output[dstOffset + i] = h[srcOffset + i]
                    }
                }
            } else {
                // Sequence-first: store contiguously
                let outStart = outT * batchSize * hiddenSize
                for i in 0..<(batchSize * hiddenSize) {
                    output[outStart + i] = h[i]
                }
            }
        }

        let finalState = LSTMLayerState(
            hidden: h,
            cell: c,
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
