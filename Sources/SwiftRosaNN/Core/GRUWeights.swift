//
//  GRUWeights.swift
//  SwiftRosaNN
//
//  Weight containers for GRU layers matching PyTorch layout.
//

import Foundation

/// Weights for a single GRU layer (one direction).
///
/// PyTorch GRU stores weights in a specific interleaved order:
/// - weight_ih: [3*hidden_size, input_size] - input weights for all 3 gates
/// - weight_hh: [3*hidden_size, hidden_size] - hidden weights for all 3 gates
/// - bias_ih: [3*hidden_size]
/// - bias_hh: [3*hidden_size]
///
/// The "3" represents the gates, stored in order: reset, update, new (r, z, n).
public struct GRULayerWeights: Sendable {
    /// Input-to-hidden weights: [3*hiddenSize, inputSize]
    /// Gates order: reset, update, new
    public let weightIH: [Float]

    /// Hidden-to-hidden weights: [3*hiddenSize, hiddenSize]
    public let weightHH: [Float]

    /// Input-to-hidden bias: [3*hiddenSize]
    public let biasIH: [Float]?

    /// Hidden-to-hidden bias: [3*hiddenSize]
    public let biasHH: [Float]?

    /// Input feature dimension.
    public let inputSize: Int

    /// Hidden state dimension.
    public let hiddenSize: Int

    /// Create GRU layer weights.
    ///
    /// - Parameters:
    ///   - weightIH: Input-to-hidden weights [3*hiddenSize, inputSize].
    ///   - weightHH: Hidden-to-hidden weights [3*hiddenSize, hiddenSize].
    ///   - biasIH: Input-to-hidden bias [3*hiddenSize], or nil if no bias.
    ///   - biasHH: Hidden-to-hidden bias [3*hiddenSize], or nil if no bias.
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
        let expectedIH = 3 * hiddenSize * inputSize
        let expectedHH = 3 * hiddenSize * hiddenSize
        let expectedBias = 3 * hiddenSize

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

    /// Gate size (3 * hiddenSize).
    public var gateSize: Int {
        3 * hiddenSize
    }
}

/// Weights for a complete GRU (potentially multi-layer, bidirectional).
public struct GRUWeights: Sendable {
    /// Forward direction weights for each layer.
    public let forward: [GRULayerWeights]

    /// Backward direction weights for each layer (nil if unidirectional).
    public let backward: [GRULayerWeights]?

    /// Configuration this weight set was created for.
    public let config: GRUConfig

    /// Create complete GRU weights.
    ///
    /// - Parameters:
    ///   - forward: Forward direction weights for each layer.
    ///   - backward: Backward direction weights for each layer (nil if unidirectional).
    ///   - config: GRU configuration.
    public init(forward: [GRULayerWeights], backward: [GRULayerWeights]?, config: GRUConfig) {
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
                "Backward weights should be nil for unidirectional GRU"
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
    public func weights(layer: Int, direction: Int) -> GRULayerWeights {
        precondition(layer >= 0 && layer < config.numLayers, "Layer index out of bounds")
        precondition(direction >= 0 && direction < config.numDirections, "Direction out of bounds")

        if direction == 0 {
            return forward[layer]
        } else {
            guard let backward = backward else {
                preconditionFailure("Backward weights not available for unidirectional GRU")
            }
            return backward[layer]
        }
    }
}
