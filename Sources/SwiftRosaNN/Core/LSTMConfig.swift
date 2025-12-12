//
//  LSTMConfig.swift
//  SwiftRosaNN
//
//  Configuration for LSTM layers.
//

import Foundation

/// Configuration for LSTM layers.
///
/// Defines the architecture parameters for LSTM, including input/hidden sizes,
/// number of layers, and whether to use bidirectional processing.
public struct LSTMConfig: Sendable, Equatable {
    /// Input feature dimension.
    public let inputSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Number of stacked LSTM layers (default: 1).
    public let numLayers: Int

    /// Whether to use bidirectional processing.
    public let bidirectional: Bool

    /// Whether to include bias terms (default: true).
    public let bias: Bool

    /// Whether to use batch_first ordering (default: true).
    /// If true: input shape is [batch, seq, features]
    /// If false: input shape is [seq, batch, features]
    public let batchFirst: Bool

    /// Create an LSTM configuration.
    ///
    /// - Parameters:
    ///   - inputSize: Input feature dimension.
    ///   - hiddenSize: Hidden state dimension.
    ///   - numLayers: Number of stacked LSTM layers (default: 1).
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

    /// Configuration for Banquet BiLSTM layers.
    ///
    /// Banquet uses BiLSTM layers with hidden size 256 (2N where N=128).
    ///
    /// - Parameter inputSize: Input feature dimension for this layer.
    /// - Returns: Configuration suitable for Banquet BiLSTM.
    public static func banquet(inputSize: Int) -> LSTMConfig {
        LSTMConfig(
            inputSize: inputSize,
            hiddenSize: 256,
            numLayers: 1,
            bidirectional: true,
            bias: true,
            batchFirst: true
        )
    }
}
