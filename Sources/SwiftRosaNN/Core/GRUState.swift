//
//  GRUState.swift
//  SwiftRosaNN
//
//  Hidden state management for GRU layers.
//

import Foundation

/// Hidden state for GRU.
///
/// GRU differs from LSTM in that it only has hidden state (no cell state).
/// For multi-layer and/or bidirectional GRUs, states are organized as:
/// [numLayers * numDirections, batch, hiddenSize]
public struct GRUState: Sendable {
    /// Hidden state values.
    /// Shape: [numLayers * numDirections, batchSize, hiddenSize] flattened.
    public let hidden: [Float]

    /// Number of GRU layers.
    public let numLayers: Int

    /// Number of directions (1 = unidirectional, 2 = bidirectional).
    public let numDirections: Int

    /// Batch size.
    public let batchSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create GRU state from existing values.
    ///
    /// - Parameters:
    ///   - hidden: Hidden state values.
    ///   - numLayers: Number of GRU layers.
    ///   - numDirections: Number of directions (1 or 2).
    ///   - batchSize: Batch size.
    ///   - hiddenSize: Hidden state dimension.
    public init(
        hidden: [Float],
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

        self.hidden = hidden
        self.numLayers = numLayers
        self.numDirections = numDirections
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
    }

    /// Total number of elements in hidden state.
    public var totalSize: Int {
        numLayers * numDirections * batchSize * hiddenSize
    }

    /// Create zero-initialized state.
    ///
    /// - Parameters:
    ///   - numLayers: Number of GRU layers.
    ///   - numDirections: Number of directions (1 or 2).
    ///   - batchSize: Batch size.
    ///   - hiddenSize: Hidden state dimension.
    /// - Returns: Zero-initialized GRU state.
    public static func zeros(
        numLayers: Int,
        numDirections: Int,
        batchSize: Int,
        hiddenSize: Int
    ) -> GRUState {
        let size = numLayers * numDirections * batchSize * hiddenSize
        return GRUState(
            hidden: [Float](repeating: 0, count: size),
            numLayers: numLayers,
            numDirections: numDirections,
            batchSize: batchSize,
            hiddenSize: hiddenSize
        )
    }

    /// Create zero-initialized state from config.
    ///
    /// - Parameters:
    ///   - config: GRU configuration.
    ///   - batchSize: Batch size.
    /// - Returns: Zero-initialized GRU state.
    public static func zeros(config: GRUConfig, batchSize: Int) -> GRUState {
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
}

/// Simple state container for a single GRU layer (one direction).
///
/// Used internally during sequence processing.
/// GRU only needs hidden state, unlike LSTM which also has cell state.
public struct GRULayerState: Sendable {
    /// Hidden state [batchSize, hiddenSize].
    public var hidden: [Float]

    /// Batch size.
    public let batchSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create layer state.
    public init(hidden: [Float], batchSize: Int, hiddenSize: Int) {
        precondition(hidden.count == batchSize * hiddenSize, "Hidden size mismatch")

        self.hidden = hidden
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
    }

    /// Create zero-initialized layer state.
    public static func zeros(batchSize: Int, hiddenSize: Int) -> GRULayerState {
        let size = batchSize * hiddenSize
        return GRULayerState(
            hidden: [Float](repeating: 0, count: size),
            batchSize: batchSize,
            hiddenSize: hiddenSize
        )
    }
}
