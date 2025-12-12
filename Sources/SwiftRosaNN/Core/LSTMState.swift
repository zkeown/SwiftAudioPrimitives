//
//  LSTMState.swift
//  SwiftRosaNN
//
//  Hidden and cell state management for LSTM layers.
//

import Foundation

/// Hidden and cell state for LSTM.
///
/// Stores the hidden state (h) and cell state (c) for LSTM processing.
/// For multi-layer and/or bidirectional LSTMs, states are organized as:
/// [numLayers * numDirections, batch, hiddenSize]
public struct LSTMState: Sendable {
    /// Hidden state values.
    /// Shape: [numLayers * numDirections, batchSize, hiddenSize] flattened.
    public let hidden: [Float]

    /// Cell state values.
    /// Shape: [numLayers * numDirections, batchSize, hiddenSize] flattened.
    public let cell: [Float]

    /// Number of LSTM layers.
    public let numLayers: Int

    /// Number of directions (1 = unidirectional, 2 = bidirectional).
    public let numDirections: Int

    /// Batch size.
    public let batchSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create LSTM state from existing values.
    ///
    /// - Parameters:
    ///   - hidden: Hidden state values.
    ///   - cell: Cell state values.
    ///   - numLayers: Number of LSTM layers.
    ///   - numDirections: Number of directions (1 or 2).
    ///   - batchSize: Batch size.
    ///   - hiddenSize: Hidden state dimension.
    public init(
        hidden: [Float],
        cell: [Float],
        numLayers: Int,
        numDirections: Int,
        batchSize: Int,
        hiddenSize: Int
    ) {
        let expectedSize = numLayers * numDirections * batchSize * hiddenSize
        precondition(
            hidden.count == expectedSize,
            "Hidden state size mismatch: expected \(expectedSize), got \(hidden.count)"
        )
        precondition(
            cell.count == expectedSize,
            "Cell state size mismatch: expected \(expectedSize), got \(cell.count)"
        )

        self.hidden = hidden
        self.cell = cell
        self.numLayers = numLayers
        self.numDirections = numDirections
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
    }

    /// Total number of elements in hidden (or cell) state.
    public var totalSize: Int {
        numLayers * numDirections * batchSize * hiddenSize
    }

    /// Create zero-initialized state.
    ///
    /// - Parameters:
    ///   - numLayers: Number of LSTM layers.
    ///   - numDirections: Number of directions (1 or 2).
    ///   - batchSize: Batch size.
    ///   - hiddenSize: Hidden state dimension.
    /// - Returns: Zero-initialized LSTM state.
    public static func zeros(
        numLayers: Int,
        numDirections: Int,
        batchSize: Int,
        hiddenSize: Int
    ) -> LSTMState {
        let size = numLayers * numDirections * batchSize * hiddenSize
        return LSTMState(
            hidden: [Float](repeating: 0, count: size),
            cell: [Float](repeating: 0, count: size),
            numLayers: numLayers,
            numDirections: numDirections,
            batchSize: batchSize,
            hiddenSize: hiddenSize
        )
    }

    /// Create zero-initialized state from config.
    ///
    /// - Parameters:
    ///   - config: LSTM configuration.
    ///   - batchSize: Batch size.
    /// - Returns: Zero-initialized LSTM state.
    public static func zeros(config: LSTMConfig, batchSize: Int) -> LSTMState {
        zeros(
            numLayers: config.numLayers,
            numDirections: config.numDirections,
            batchSize: batchSize,
            hiddenSize: config.hiddenSize
        )
    }

    /// Extract hidden state for a specific layer and direction.
    ///
    /// - Parameters:
    ///   - layer: Layer index (0-based).
    ///   - direction: Direction (0 = forward, 1 = backward).
    /// - Returns: Hidden state for that layer/direction [batchSize, hiddenSize].
    public func hiddenForLayer(_ layer: Int, direction: Int) -> [Float] {
        let offset = (layer * numDirections + direction) * batchSize * hiddenSize
        let end = offset + batchSize * hiddenSize
        return Array(hidden[offset..<end])
    }

    /// Extract cell state for a specific layer and direction.
    ///
    /// - Parameters:
    ///   - layer: Layer index (0-based).
    ///   - direction: Direction (0 = forward, 1 = backward).
    /// - Returns: Cell state for that layer/direction [batchSize, hiddenSize].
    public func cellForLayer(_ layer: Int, direction: Int) -> [Float] {
        let offset = (layer * numDirections + direction) * batchSize * hiddenSize
        let end = offset + batchSize * hiddenSize
        return Array(cell[offset..<end])
    }
}

/// Simple state container for a single LSTM layer (one direction).
///
/// Used internally during sequence processing.
public struct LSTMLayerState: Sendable {
    /// Hidden state [batchSize, hiddenSize].
    public var hidden: [Float]

    /// Cell state [batchSize, hiddenSize].
    public var cell: [Float]

    /// Batch size.
    public let batchSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create layer state.
    public init(hidden: [Float], cell: [Float], batchSize: Int, hiddenSize: Int) {
        precondition(hidden.count == batchSize * hiddenSize, "Hidden size mismatch")
        precondition(cell.count == batchSize * hiddenSize, "Cell size mismatch")

        self.hidden = hidden
        self.cell = cell
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
    }

    /// Create zero-initialized layer state.
    public static func zeros(batchSize: Int, hiddenSize: Int) -> LSTMLayerState {
        let size = batchSize * hiddenSize
        return LSTMLayerState(
            hidden: [Float](repeating: 0, count: size),
            cell: [Float](repeating: 0, count: size),
            batchSize: batchSize,
            hiddenSize: hiddenSize
        )
    }
}
