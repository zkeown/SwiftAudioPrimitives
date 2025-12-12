//
//  LSTMWeights.swift
//  SwiftRosaNN
//
//  Weight containers for LSTM layers matching PyTorch layout.
//

import Foundation

/// Weights for a single LSTM layer (one direction).
///
/// PyTorch LSTM stores weights in a specific interleaved order:
/// - weight_ih: [4*hidden_size, input_size] - input weights for all 4 gates
/// - weight_hh: [4*hidden_size, hidden_size] - hidden weights for all 4 gates
/// - bias_ih: [4*hidden_size]
/// - bias_hh: [4*hidden_size]
///
/// The "4" represents the gates, stored in order: input, forget, cell, output (i, f, g, o).
public struct LSTMLayerWeights: Sendable {
    /// Input-to-hidden weights: [4*hiddenSize, inputSize]
    /// Gates order: input, forget, cell, output
    public let weightIH: [Float]

    /// Hidden-to-hidden weights: [4*hiddenSize, hiddenSize]
    public let weightHH: [Float]

    /// Input-to-hidden bias: [4*hiddenSize]
    public let biasIH: [Float]?

    /// Hidden-to-hidden bias: [4*hiddenSize]
    public let biasHH: [Float]?

    /// Input feature dimension.
    public let inputSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create LSTM layer weights.
    ///
    /// - Parameters:
    ///   - weightIH: Input-to-hidden weights [4*hiddenSize, inputSize].
    ///   - weightHH: Hidden-to-hidden weights [4*hiddenSize, hiddenSize].
    ///   - biasIH: Input-to-hidden bias [4*hiddenSize], or nil if no bias.
    ///   - biasHH: Hidden-to-hidden bias [4*hiddenSize], or nil if no bias.
    ///   - inputSize: Input feature dimension.
    ///   - hiddenSize: Hidden state dimension.
    public init(
        weightIH: [Float],
        weightHH: [Float],
        biasIH: [Float]?,
        biasHH: [Float]?,
        inputSize: Int,
        hiddenSize: Int
    ) {
        let expectedIH = 4 * hiddenSize * inputSize
        let expectedHH = 4 * hiddenSize * hiddenSize
        let expectedBias = 4 * hiddenSize

        precondition(
            weightIH.count == expectedIH,
            "weightIH shape mismatch: expected \(expectedIH), got \(weightIH.count)"
        )
        precondition(
            weightHH.count == expectedHH,
            "weightHH shape mismatch: expected \(expectedHH), got \(weightHH.count)"
        )
        if let b = biasIH {
            precondition(
                b.count == expectedBias,
                "biasIH shape mismatch: expected \(expectedBias), got \(b.count)"
            )
        }
        if let b = biasHH {
            precondition(
                b.count == expectedBias,
                "biasHH shape mismatch: expected \(expectedBias), got \(b.count)"
            )
        }

        self.weightIH = weightIH
        self.weightHH = weightHH
        self.biasIH = biasIH
        self.biasHH = biasHH
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
    }

    /// Gate size (4 * hiddenSize).
    public var gateSize: Int {
        4 * hiddenSize
    }
}

/// Weights for a complete LSTM (potentially multi-layer, bidirectional).
public struct LSTMWeights: Sendable {
    /// Forward direction weights for each layer.
    public let forward: [LSTMLayerWeights]

    /// Backward direction weights for each layer (nil if unidirectional).
    public let backward: [LSTMLayerWeights]?

    /// Configuration this weight set was created for.
    public let config: LSTMConfig

    /// Create complete LSTM weights.
    ///
    /// - Parameters:
    ///   - forward: Forward direction weights for each layer.
    ///   - backward: Backward direction weights for each layer (nil if unidirectional).
    ///   - config: LSTM configuration.
    public init(forward: [LSTMLayerWeights], backward: [LSTMLayerWeights]?, config: LSTMConfig) {
        precondition(
            forward.count == config.numLayers,
            "Forward weights count must match numLayers: expected \(config.numLayers), got \(forward.count)"
        )
        if config.bidirectional {
            precondition(
                backward?.count == config.numLayers,
                "Backward weights count must match numLayers for bidirectional"
            )
        } else {
            precondition(
                backward == nil,
                "Backward weights should be nil for unidirectional LSTM"
            )
        }

        self.forward = forward
        self.backward = backward
        self.config = config
    }

    /// Get weights for a specific layer and direction.
    ///
    /// - Parameters:
    ///   - layer: Layer index (0-based).
    ///   - direction: Direction (0 = forward, 1 = backward).
    /// - Returns: Layer weights for the specified layer and direction.
    public func weights(layer: Int, direction: Int) -> LSTMLayerWeights {
        precondition(layer >= 0 && layer < config.numLayers, "Layer index out of bounds")
        precondition(direction >= 0 && direction < config.numDirections, "Direction out of bounds")

        if direction == 0 {
            return forward[layer]
        } else {
            return backward![layer]
        }
    }
}
