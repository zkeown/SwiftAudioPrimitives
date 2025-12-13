//
//  GRUConfig.swift
//  SwiftRosaNN
//
//  Configuration for GRU layers.
//

import Foundation

/// Configuration for GRU layers.
///
/// Defines the architecture parameters for GRU, including input/hidden sizes,
/// number of layers, and whether to use bidirectional processing.
///
/// GRU differs from LSTM in having 3 gates (reset, update, new) instead of 4,
/// and uses only hidden state without a separate cell state.
public struct GRUConfig: Sendable, Equatable {
    /// Input feature dimension.
    public let inputSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Number of stacked GRU layers (default: 1).
    public let numLayers: Int

    /// Whether to use bidirectional processing.
    public let bidirectional: Bool

    /// Whether to include bias terms (default: true).
    public let bias: Bool

    /// Whether to use batch_first ordering (default: true).
    /// If true: input shape is [batch, seq, features]
    /// If false: input shape is [seq, batch, features]
    public let batchFirst: Bool

    /// Create a GRU configuration.
    ///
    /// - Parameters:
    ///   - inputSize: Input feature dimension.
    ///   - hiddenSize: Hidden state dimension.
    ///   - numLayers: Number of stacked GRU layers (default: 1).
    ///   - bidirectional: Whether to use bidirectional processing (default: false).
    ///   - bias: Whether to include bias terms (default: true).
    ///   - batchFirst: Whether input uses batch_first ordering (default: true).
    public init(
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bidirectional: Bool = false,
        bias: Bool = true,
        batchFirst: Bool = true
    ) {
        precondition(inputSize > 0, "inputSize must be positive")
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(numLayers > 0, "numLayers must be positive")

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.bias = bias
        self.batchFirst = batchFirst
    }

    /// Output size per timestep.
    /// For bidirectional, this is 2 * hiddenSize (concatenated forward and backward).
    public var outputSize: Int {
        bidirectional ? hiddenSize * 2 : hiddenSize
    }

    /// Number of directions (1 for unidirectional, 2 for bidirectional).
    public var numDirections: Int {
        bidirectional ? 2 : 1
    }

    /// Gate size (3 * hiddenSize for GRU: reset, update, new gates).
    public var gateSize: Int {
        3 * hiddenSize
    }

    /// Configuration for Banquet BiGRU layers.
    ///
    /// Banquet uses BiGRU layers with hidden size 256.
    ///
    /// - Parameter inputSize: Input feature dimension for this layer.
    /// - Returns: Configuration suitable for Banquet BiGRU.
    public static func banquet(inputSize: Int) -> GRUConfig {
        GRUConfig(
            inputSize: inputSize,
            hiddenSize: 256,
            numLayers: 1,
            bidirectional: true,
            bias: true,
            batchFirst: true
        )
    }
}
