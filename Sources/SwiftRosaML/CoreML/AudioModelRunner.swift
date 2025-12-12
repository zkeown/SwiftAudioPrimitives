import SwiftRosaCore
import Accelerate
import CoreML
import Foundation

/// Errors that can occur during model inference.
public enum AudioModelRunnerError: Error, Sendable {
    case modelLoadFailed(String)
    case predictionFailed(String)
    case invalidInputShape(expected: [Int], got: [Int])
    case missingInput(String)
    case missingOutput(String)
    case mlArrayCreationFailed
}

/// Thread-safe wrapper for Core ML model inference on audio data.
///
/// `AudioModelRunner` provides an actor-isolated interface to run Core ML models,
/// with utilities for converting between Swift arrays and `MLMultiArray`.
///
/// ## Example Usage
///
/// ```swift
/// let runner = try await AudioModelRunner(modelURL: modelURL)
///
/// // Create input tensor
/// let input = try AudioModelRunner.toMLMultiArray(spectrogramData, shape: [1, 2, 1025, 256])
///
/// // Run inference
/// let outputs = try await runner.predict(["mixture_stft": input])
///
/// // Extract output
/// if let mask = outputs["mask"] {
///     let maskData = AudioModelRunner.fromMLMultiArray(mask)
/// }
/// ```
public actor AudioModelRunner {
    /// The loaded Core ML model.
    private let model: MLModel

    /// Cached input feature descriptions.
    private let inputDescriptions: [String: MLFeatureDescription]

    /// Cached output feature descriptions.
    private let outputDescriptions: [String: MLFeatureDescription]

    /// Initialize with a compiled Core ML model URL.
    ///
    /// - Parameters:
    ///   - modelURL: URL to a `.mlmodelc` (compiled) or `.mlpackage` model.
    ///   - configuration: Model configuration for compute units, etc.
    /// - Throws: `AudioModelRunnerError.modelLoadFailed` if the model cannot be loaded.
    public init(
        modelURL: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) async throws {
        // Compile model if needed (mlpackage -> mlmodelc)
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            do {
                compiledURL = try await MLModel.compileModel(at: modelURL)
            } catch {
                throw AudioModelRunnerError.modelLoadFailed("Failed to compile model: \(error)")
            }
        }

        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw AudioModelRunnerError.modelLoadFailed("Failed to load model: \(error)")
        }

        // Cache feature descriptions
        var inputs: [String: MLFeatureDescription] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            inputs[name] = desc
        }
        self.inputDescriptions = inputs

        var outputs: [String: MLFeatureDescription] = [:]
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            outputs[name] = desc
        }
        self.outputDescriptions = outputs
    }

    // MARK: - Model Information

    /// Names of all model inputs.
    public var inputNames: [String] {
        Array(inputDescriptions.keys)
    }

    /// Names of all model outputs.
    public var outputNames: [String] {
        Array(outputDescriptions.keys)
    }

    /// Get the expected shape for an input tensor.
    ///
    /// - Parameter name: Input feature name.
    /// - Returns: Shape array, or nil if input not found or shape unavailable.
    public func inputShape(for name: String) -> [Int]? {
        guard let desc = inputDescriptions[name],
              let constraint = desc.multiArrayConstraint else {
            return nil
        }
        return constraint.shape.map { $0.intValue }
    }

    /// Get the expected shape for an output tensor.
    ///
    /// - Parameter name: Output feature name.
    /// - Returns: Shape array, or nil if output not found or shape unavailable.
    public func outputShape(for name: String) -> [Int]? {
        guard let desc = outputDescriptions[name],
              let constraint = desc.multiArrayConstraint else {
            return nil
        }
        return constraint.shape.map { $0.intValue }
    }

    // MARK: - Inference

    /// Run model inference with the provided inputs.
    ///
    /// - Parameter inputs: Dictionary mapping input names to MLMultiArray tensors.
    /// - Returns: Dictionary mapping output names to MLMultiArray results.
    /// - Throws: `AudioModelRunnerError` on inference failure.
    public func predict(_ inputs: [String: MLMultiArray]) async throws -> [String: MLMultiArray] {
        // Validate all required inputs are provided
        for inputName in inputDescriptions.keys {
            if inputs[inputName] == nil {
                throw AudioModelRunnerError.missingInput(inputName)
            }
        }

        // Create feature provider
        let featureProvider = try MLDictionaryFeatureProvider(dictionary: inputs)

        // Run prediction
        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: featureProvider)
        } catch {
            throw AudioModelRunnerError.predictionFailed("Prediction failed: \(error)")
        }

        // Extract outputs
        var outputs: [String: MLMultiArray] = [:]
        for outputName in outputDescriptions.keys {
            guard let feature = result.featureValue(for: outputName),
                  let multiArray = feature.multiArrayValue else {
                throw AudioModelRunnerError.missingOutput(outputName)
            }
            outputs[outputName] = multiArray
        }

        return outputs
    }

    /// Run model inference with named inputs for convenience.
    ///
    /// - Parameters:
    ///   - input: Primary input tensor.
    ///   - inputName: Name of the primary input (defaults to first input).
    /// - Returns: Primary output MLMultiArray.
    public func predict(
        _ input: MLMultiArray,
        inputName: String? = nil
    ) async throws -> MLMultiArray {
        let name = inputName ?? inputNames.first ?? "input"
        let outputs = try await predict([name: input])
        guard let output = outputs.values.first else {
            throw AudioModelRunnerError.missingOutput("No outputs returned")
        }
        return output
    }
}

// MARK: - MLMultiArray Conversion Helpers

extension AudioModelRunner {
    /// Create an MLMultiArray from a flat Float array.
    ///
    /// - Parameters:
    ///   - data: Flat array of Float values.
    ///   - shape: Shape of the tensor.
    /// - Returns: MLMultiArray with the specified shape.
    /// - Throws: `AudioModelRunnerError.mlArrayCreationFailed` on failure.
    public static func toMLMultiArray(_ data: [Float], shape: [Int]) throws -> MLMultiArray {
        let expectedCount = shape.reduce(1, *)
        guard data.count == expectedCount else {
            throw AudioModelRunnerError.invalidInputShape(
                expected: [expectedCount],
                got: [data.count]
            )
        }

        guard let array = try? MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32) else {
            throw AudioModelRunnerError.mlArrayCreationFailed
        }

        // Copy data efficiently
        let dataPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        data.withUnsafeBufferPointer { srcPtr in
            dataPtr.update(from: srcPtr.baseAddress!, count: data.count)
        }

        return array
    }

    /// Create an MLMultiArray from a 2D Float array.
    ///
    /// - Parameters:
    ///   - data: 2D array of shape (rows, cols).
    ///   - shape: Optional explicit shape. If nil, uses [rows, cols].
    /// - Returns: MLMultiArray with the specified shape.
    public static func toMLMultiArray(_ data: [[Float]], shape: [Int]? = nil) throws -> MLMultiArray {
        let rows = data.count
        let cols = data.first?.count ?? 0
        let flatData = data.flatMap { $0 }
        let actualShape = shape ?? [rows, cols]
        return try toMLMultiArray(flatData, shape: actualShape)
    }

    /// Create an MLMultiArray from a 3D Float array.
    ///
    /// - Parameters:
    ///   - data: 3D array of shape (d1, d2, d3).
    ///   - shape: Optional explicit shape.
    /// - Returns: MLMultiArray with the specified shape.
    public static func toMLMultiArray(_ data: [[[Float]]], shape: [Int]? = nil) throws -> MLMultiArray {
        let d1 = data.count
        let d2 = data.first?.count ?? 0
        let d3 = data.first?.first?.count ?? 0
        let flatData = data.flatMap { $0.flatMap { $0 } }
        let actualShape = shape ?? [d1, d2, d3]
        return try toMLMultiArray(flatData, shape: actualShape)
    }

    /// Create an MLMultiArray from a 4D Float array.
    ///
    /// - Parameters:
    ///   - data: 4D array of shape (d1, d2, d3, d4).
    ///   - shape: Optional explicit shape.
    /// - Returns: MLMultiArray with the specified shape.
    public static func toMLMultiArray(_ data: [[[[Float]]]], shape: [Int]? = nil) throws -> MLMultiArray {
        let d1 = data.count
        let d2 = data.first?.count ?? 0
        let d3 = data.first?.first?.count ?? 0
        let d4 = data.first?.first?.first?.count ?? 0
        let flatData = data.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        let actualShape = shape ?? [d1, d2, d3, d4]
        return try toMLMultiArray(flatData, shape: actualShape)
    }

    /// Convert MLMultiArray to a flat Float array.
    ///
    /// - Parameter array: Source MLMultiArray.
    /// - Returns: Flat array of Float values.
    public static func fromMLMultiArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)

        let dataPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        result.withUnsafeMutableBufferPointer { destPtr in
            destPtr.baseAddress!.update(from: dataPtr, count: count)
        }

        return result
    }

    /// Convert MLMultiArray to a 2D Float array.
    ///
    /// - Parameters:
    ///   - array: Source MLMultiArray.
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    /// - Returns: 2D array of shape (rows, cols).
    public static func fromMLMultiArray2D(_ array: MLMultiArray, rows: Int, cols: Int) -> [[Float]] {
        let flat = fromMLMultiArray(array)
        var result = [[Float]]()
        result.reserveCapacity(rows)

        for i in 0..<rows {
            let start = i * cols
            let end = start + cols
            result.append(Array(flat[start..<end]))
        }

        return result
    }

    /// Convert MLMultiArray to a 3D Float array.
    ///
    /// - Parameters:
    ///   - array: Source MLMultiArray.
    ///   - d1: First dimension size.
    ///   - d2: Second dimension size.
    ///   - d3: Third dimension size.
    /// - Returns: 3D array of shape (d1, d2, d3).
    public static func fromMLMultiArray3D(_ array: MLMultiArray, d1: Int, d2: Int, d3: Int) -> [[[Float]]] {
        let flat = fromMLMultiArray(array)
        var result = [[[Float]]]()
        result.reserveCapacity(d1)

        for i in 0..<d1 {
            var slice2D = [[Float]]()
            slice2D.reserveCapacity(d2)
            for j in 0..<d2 {
                let start = (i * d2 + j) * d3
                let end = start + d3
                slice2D.append(Array(flat[start..<end]))
            }
            result.append(slice2D)
        }

        return result
    }

    /// Convert MLMultiArray to a 4D Float array.
    ///
    /// - Parameters:
    ///   - array: Source MLMultiArray.
    ///   - d1: First dimension size.
    ///   - d2: Second dimension size.
    ///   - d3: Third dimension size.
    ///   - d4: Fourth dimension size.
    /// - Returns: 4D array of shape (d1, d2, d3, d4).
    public static func fromMLMultiArray4D(
        _ array: MLMultiArray,
        d1: Int,
        d2: Int,
        d3: Int,
        d4: Int
    ) -> [[[[Float]]]] {
        let flat = fromMLMultiArray(array)
        var result = [[[[Float]]]]()
        result.reserveCapacity(d1)

        for i in 0..<d1 {
            var slice3D = [[[Float]]]()
            slice3D.reserveCapacity(d2)
            for j in 0..<d2 {
                var slice2D = [[Float]]()
                slice2D.reserveCapacity(d3)
                for k in 0..<d3 {
                    let start = ((i * d2 + j) * d3 + k) * d4
                    let end = start + d4
                    slice2D.append(Array(flat[start..<end]))
                }
                slice3D.append(slice2D)
            }
            result.append(slice3D)
        }

        return result
    }
}

// MARK: - ComplexMatrix Conversion

extension AudioModelRunner {
    /// Convert ComplexMatrix to MLMultiArray with shape [batch, 2, freqs, frames].
    ///
    /// Channel 0 contains real values, channel 1 contains imaginary values.
    ///
    /// - Parameters:
    ///   - complex: Complex spectrogram matrix.
    ///   - batch: Batch dimension (default 1).
    /// - Returns: MLMultiArray suitable for model input.
    public static func complexToMLMultiArray(_ complex: ComplexMatrix, batch: Int = 1) throws -> MLMultiArray {
        let freqs = complex.rows
        let frames = complex.cols

        // Shape: [batch, 2, freqs, frames]
        let shape = [batch, 2, freqs, frames]
        let totalCount = shape.reduce(1, *)

        guard let array = try? MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: .float32
        ) else {
            throw AudioModelRunnerError.mlArrayCreationFailed
        }

        let dataPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = freqs * frames

        // Copy real channel (channel 0)
        for f in 0..<freqs {
            for t in 0..<frames {
                let idx = f * frames + t
                dataPtr[idx] = complex.real[f][t]
            }
        }

        // Copy imaginary channel (channel 1)
        for f in 0..<freqs {
            for t in 0..<frames {
                let idx = channelStride + f * frames + t
                dataPtr[idx] = complex.imag[f][t]
            }
        }

        return array
    }

    /// Convert MLMultiArray with shape [batch, 2, freqs, frames] to ComplexMatrix.
    ///
    /// - Parameters:
    ///   - array: Source MLMultiArray.
    ///   - freqs: Number of frequency bins.
    ///   - frames: Number of time frames.
    /// - Returns: ComplexMatrix extracted from the array.
    public static func mlMultiArrayToComplex(
        _ array: MLMultiArray,
        freqs: Int,
        frames: Int
    ) -> ComplexMatrix {
        let dataPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = freqs * frames

        var real = [[Float]]()
        var imag = [[Float]]()
        real.reserveCapacity(freqs)
        imag.reserveCapacity(freqs)

        for f in 0..<freqs {
            var realRow = [Float](repeating: 0, count: frames)
            var imagRow = [Float](repeating: 0, count: frames)

            for t in 0..<frames {
                let idx = f * frames + t
                realRow[t] = dataPtr[idx]
                imagRow[t] = dataPtr[channelStride + idx]
            }

            real.append(realRow)
            imag.append(imagRow)
        }

        return ComplexMatrix(real: real, imag: imag)
    }
}
