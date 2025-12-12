//
//  WeightLoader.swift
//  SwiftRosaNN
//
//  Utilities for loading pre-converted weight files.
//

import Foundation

/// Errors that can occur during weight loading.
public enum WeightLoadError: Error, Sendable {
    /// Weight file not found at the specified path.
    case fileNotFound(String)

    /// Invalid weight file format.
    case invalidFormat(String)

    /// Weight size doesn't match expected dimensions.
    case sizeMismatch(expected: Int, got: Int)

    /// Invalid or incompatible configuration.
    case invalidConfig(String)
}

/// Loads pre-converted weight files for LSTM layers.
///
/// Expected binary format: contiguous float32 values, little-endian.
/// This matches the output of `numpy.ndarray.tofile()` with dtype=float32.
///
/// ## Python Export Script
///
/// ```python
/// import torch
/// import numpy as np
///
/// ckpt = torch.load("model.ckpt", map_location="cpu")
/// for name, tensor in ckpt['state_dict'].items():
///     arr = tensor.numpy().astype(np.float32)
///     arr.tofile(f"weights/{name.replace('.', '_')}.bin")
/// ```
///
/// ## File Naming Convention
///
/// For LSTM layers, use PyTorch naming:
/// - `weight_ih_l{layer}.bin` - Input-to-hidden weights
/// - `weight_hh_l{layer}.bin` - Hidden-to-hidden weights
/// - `bias_ih_l{layer}.bin` - Input-to-hidden bias
/// - `bias_hh_l{layer}.bin` - Hidden-to-hidden bias
/// - `weight_ih_l{layer}_reverse.bin` - Backward direction (bidirectional)
/// - etc.
public struct WeightLoader {
    /// Load raw float32 weights from a binary file.
    ///
    /// - Parameters:
    ///   - url: URL to the .bin weight file.
    ///   - expectedCount: Expected number of float values.
    /// - Returns: Array of weight values.
    /// - Throws: `WeightLoadError` if file doesn't exist or size mismatches.
    public static func loadRawFloats(from url: URL, expectedCount: Int) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WeightLoadError.fileNotFound(url.path)
        }

        let data = try Data(contentsOf: url)
        let floatCount = data.count / MemoryLayout<Float>.size

        guard floatCount == expectedCount else {
            throw WeightLoadError.sizeMismatch(expected: expectedCount, got: floatCount)
        }

        var result = [Float](repeating: 0, count: floatCount)
        _ = result.withUnsafeMutableBytes { destPtr in
            data.copyBytes(to: destPtr)
        }

        return result
    }

    /// Load raw float32 weights from a binary file without size validation.
    ///
    /// - Parameter url: URL to the .bin weight file.
    /// - Returns: Array of weight values.
    /// - Throws: `WeightLoadError` if file doesn't exist.
    public static func loadRawFloats(from url: URL) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WeightLoadError.fileNotFound(url.path)
        }

        let data = try Data(contentsOf: url)
        let floatCount = data.count / MemoryLayout<Float>.size

        var result = [Float](repeating: 0, count: floatCount)
        _ = result.withUnsafeMutableBytes { destPtr in
            data.copyBytes(to: destPtr)
        }

        return result
    }

    /// Load complete LSTM weights from a directory.
    ///
    /// Expects files named according to PyTorch convention:
    /// - `weight_ih_l{layer}.bin`
    /// - `weight_hh_l{layer}.bin`
    /// - `bias_ih_l{layer}.bin` (if config.bias is true)
    /// - `bias_hh_l{layer}.bin` (if config.bias is true)
    /// - `weight_ih_l{layer}_reverse.bin` (if bidirectional)
    /// - etc.
    ///
    /// - Parameters:
    ///   - directory: Directory containing weight files.
    ///   - config: LSTM configuration.
    /// - Returns: Complete LSTMWeights.
    /// - Throws: `WeightLoadError` if files are missing or have wrong sizes.
    public static func loadLSTMWeights(
        from directory: URL,
        config: LSTMConfig
    ) throws -> LSTMWeights {
        var forwardWeights: [LSTMLayerWeights] = []
        var backwardWeights: [LSTMLayerWeights]? = config.bidirectional ? [] : nil

        for layer in 0..<config.numLayers {
            // Determine input size for this layer
            // Layer 0 uses config.inputSize
            // Subsequent layers use previous layer's output size
            let layerInputSize: Int
            if layer == 0 {
                layerInputSize = config.inputSize
            } else {
                // Previous layer output: hiddenSize * numDirections
                layerInputSize = config.bidirectional ? config.hiddenSize * 2 : config.hiddenSize
            }

            // Load forward direction
            let fwdWeights = try loadLayerWeights(
                from: directory,
                layer: layer,
                inputSize: layerInputSize,
                hiddenSize: config.hiddenSize,
                hasBias: config.bias,
                reverse: false
            )
            forwardWeights.append(fwdWeights)

            // Load backward direction if bidirectional
            if config.bidirectional {
                let bwdWeights = try loadLayerWeights(
                    from: directory,
                    layer: layer,
                    inputSize: layerInputSize,
                    hiddenSize: config.hiddenSize,
                    hasBias: config.bias,
                    reverse: true
                )
                backwardWeights?.append(bwdWeights)
            }
        }

        return LSTMWeights(forward: forwardWeights, backward: backwardWeights, config: config)
    }

    /// Load weights for a single LSTM layer direction.
    private static func loadLayerWeights(
        from directory: URL,
        layer: Int,
        inputSize: Int,
        hiddenSize: Int,
        hasBias: Bool,
        reverse: Bool
    ) throws -> LSTMLayerWeights {
        let suffix = reverse ? "_reverse" : ""

        let weightIHURL = directory.appendingPathComponent("weight_ih_l\(layer)\(suffix).bin")
        let weightHHURL = directory.appendingPathComponent("weight_hh_l\(layer)\(suffix).bin")

        let weightIH = try loadRawFloats(
            from: weightIHURL,
            expectedCount: 4 * hiddenSize * inputSize
        )

        let weightHH = try loadRawFloats(
            from: weightHHURL,
            expectedCount: 4 * hiddenSize * hiddenSize
        )

        var biasIH: [Float]? = nil
        var biasHH: [Float]? = nil

        if hasBias {
            let biasIHURL = directory.appendingPathComponent("bias_ih_l\(layer)\(suffix).bin")
            let biasHHURL = directory.appendingPathComponent("bias_hh_l\(layer)\(suffix).bin")

            biasIH = try loadRawFloats(
                from: biasIHURL,
                expectedCount: 4 * hiddenSize
            )
            biasHH = try loadRawFloats(
                from: biasHHURL,
                expectedCount: 4 * hiddenSize
            )
        }

        return LSTMLayerWeights(
            weightIH: weightIH,
            weightHH: weightHH,
            biasIH: biasIH,
            biasHH: biasHH,
            inputSize: inputSize,
            hiddenSize: hiddenSize
        )
    }

    /// Save weights to a binary file (for testing/debugging).
    ///
    /// - Parameters:
    ///   - weights: Array of float values.
    ///   - url: Destination URL.
    public static func saveRawFloats(_ weights: [Float], to url: URL) throws {
        let data = weights.withUnsafeBytes { Data($0) }
        try data.write(to: url)
    }
}
