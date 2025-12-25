import SwiftRosaCore
import Accelerate
import CoreML
import Foundation

/// Configuration for source separation processing.
public struct SourceSeparationConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// FFT size.
    public let nFFT: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Window function type.
    public let windowType: WindowType

    /// Whether to center the signal by padding.
    public let center: Bool

    /// Duration of each processing chunk in seconds. Nil for no chunking.
    public let chunkDuration: Float?

    /// Overlap ratio between chunks (0.0 to 1.0).
    public let chunkOverlap: Float

    /// Create source separation configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 44100).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: 512).
    ///   - windowType: Window function (default: hann).
    ///   - center: Pad signal for centered frames (default: true).
    ///   - chunkDuration: Processing chunk duration in seconds (default: 10.0).
    ///   - chunkOverlap: Overlap ratio between chunks (default: 0.5).
    public init(
        sampleRate: Float = 44100,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        windowType: WindowType = .hann,
        center: Bool = true,
        chunkDuration: Float? = 10.0,
        chunkOverlap: Float = 0.5
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.windowType = windowType
        self.center = center
        self.chunkDuration = chunkDuration
        self.chunkOverlap = max(0, min(1, chunkOverlap))
    }

    /// Default configuration matching Banquet model parameters.
    public static var banquetDefault: SourceSeparationConfig {
        SourceSeparationConfig(
            sampleRate: 44100,
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true,
            chunkDuration: 10.0,
            chunkOverlap: 0.5
        )
    }

    /// Number of samples per chunk.
    public var chunkSamples: Int? {
        guard let duration = chunkDuration else { return nil }
        return Int(duration * sampleRate)
    }

    /// STFT configuration derived from this config.
    public var stftConfig: STFTConfig {
        STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            winLength: nFFT,
            windowType: windowType,
            center: center,
            padMode: .reflect
        )
    }
}

/// Errors that can occur during source separation.
public enum SourceSeparationError: Error, Sendable {
    case modelLoadFailed(String)
    case invalidQueryLength(expected: Int, got: Int)
    case inferenceError(String)
    case shapeMismatch(String)
}

/// End-to-end source separation pipeline using Core ML.
///
/// This actor orchestrates the full source separation workflow:
/// audio → STFT → model inference → complex mask application → ISTFT → separated audio
///
/// ## Example Usage
///
/// ```swift
/// let pipeline = try await SourceSeparationPipeline(modelURL: modelURL)
///
/// // Separate vocals from mixture using a vocal query
/// let separatedVocals = try await pipeline.separate(mixture: mixAudio, query: vocalQuery)
///
/// // With progress tracking for long audio
/// let separated = try await pipeline.separate(mixture: longAudio, query: query) { progress in
///     print("Progress: \(Int(progress * 100))%")
/// }
/// ```
public actor SourceSeparationPipeline {
    /// The Core ML model runner.
    private let modelRunner: AudioModelRunner

    /// STFT processor.
    private let stft: STFT

    /// ISTFT processor.
    private let istft: ISTFT

    /// Pipeline configuration.
    public let config: SourceSeparationConfig

    /// Initialize the source separation pipeline.
    ///
    /// - Parameters:
    ///   - modelURL: URL to the Core ML model (.mlmodelc or .mlpackage).
    ///   - config: Pipeline configuration.
    /// - Throws: `SourceSeparationError.modelLoadFailed` if model cannot be loaded.
    public init(
        modelURL: URL,
        config: SourceSeparationConfig = .banquetDefault
    ) async throws {
        // Configure Core ML for optimal performance
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all  // Use Neural Engine when available

        do {
            self.modelRunner = try await AudioModelRunner(modelURL: modelURL, configuration: mlConfig)
        } catch {
            throw SourceSeparationError.modelLoadFailed("Failed to load model: \(error)")
        }

        self.config = config
        self.stft = STFT(config: config.stftConfig)
        self.istft = ISTFT(config: config.stftConfig)
    }

    // MARK: - Separation Methods

    /// Separate a target source from the mixture audio.
    ///
    /// - Parameters:
    ///   - mixture: Input mixture audio signal.
    ///   - query: Query audio representing the target source type (typically 10 seconds).
    /// - Returns: Separated source audio signal.
    public func separate(mixture: [Float], query: [Float]) async throws -> [Float] {
        try await separate(mixture: mixture, query: query, progress: nil)
    }

    /// Separate a target source from the mixture audio with progress tracking.
    ///
    /// - Parameters:
    ///   - mixture: Input mixture audio signal.
    ///   - query: Query audio representing the target source type.
    ///   - progress: Optional callback reporting progress (0.0 to 1.0).
    /// - Returns: Separated source audio signal.
    public func separate(
        mixture: [Float],
        query: [Float],
        progress: (@Sendable (Float) -> Void)?
    ) async throws -> [Float] {
        // Process query to get embedding (or use raw query depending on model)
        let queryEmbedding = try await processQuery(query)

        // Check if we need chunked processing
        if let chunkSamples = config.chunkSamples, mixture.count > chunkSamples {
            return try await separateChunked(
                mixture: mixture,
                queryEmbedding: queryEmbedding,
                progress: progress
            )
        } else {
            progress?(0.0)
            let result = try await separateSingle(mixture: mixture, queryEmbedding: queryEmbedding)
            progress?(1.0)
            return result
        }
    }

    /// Separate multiple sources in parallel.
    ///
    /// - Parameters:
    ///   - mixture: Input mixture audio signal.
    ///   - queries: Dictionary mapping source names to query audio.
    /// - Returns: Dictionary mapping source names to separated audio.
    public func separateMultiple(
        mixture: [Float],
        queries: [String: [Float]]
    ) async throws -> [String: [Float]] {
        var results: [String: [Float]] = [:]

        for (name, query) in queries {
            results[name] = try await separate(mixture: mixture, query: query)
        }

        return results
    }

    // MARK: - Private Implementation

    /// Process query audio to get embedding (model-dependent).
    private func processQuery(_ query: [Float]) async throws -> MLMultiArray {
        // For Banquet, the query may go through a separate encoder or be processed directly
        // This implementation assumes the model takes raw query STFT or embedding

        // Option 1: If model expects query STFT
        let queryStft = await stft.transform(query)
        return try AudioModelRunner.complexToMLMultiArray(queryStft)

        // Option 2: If model expects pre-computed embedding, you would:
        // return try AudioModelRunner.toMLMultiArray(embeddingVector, shape: [1, embeddingDim])
    }

    /// Process a single chunk (or entire signal if not chunking).
    private func separateSingle(
        mixture: [Float],
        queryEmbedding: MLMultiArray
    ) async throws -> [Float] {
        // 1. Compute STFT of mixture
        let mixtureStft = await stft.transform(mixture)

        // 2. Prepare model input
        let mixtureInput = try AudioModelRunner.complexToMLMultiArray(mixtureStft)

        // 3. Run model inference
        let inputs = [
            "mixture_stft": mixtureInput,
            "query": queryEmbedding
        ]

        let outputs: [String: MLMultiArray]
        do {
            outputs = try await modelRunner.predict(inputs)
        } catch {
            throw SourceSeparationError.inferenceError("Model inference failed: \(error)")
        }

        // 4. Extract mask from output
        guard let maskArray = outputs["mask"] ?? outputs.values.first else {
            throw SourceSeparationError.inferenceError("No mask output from model")
        }

        let mask = AudioModelRunner.mlMultiArrayToComplex(
            maskArray,
            freqs: mixtureStft.rows,
            frames: mixtureStft.cols
        )

        // 5. Apply complex mask to mixture STFT
        let maskedStft = applyComplexMask(mask: mask, to: mixtureStft)

        // 6. Inverse STFT to get separated audio
        let separated = await istft.transform(maskedStft, length: mixture.count)

        return separated
    }

    /// Process long audio in overlapping chunks.
    private func separateChunked(
        mixture: [Float],
        queryEmbedding: MLMultiArray,
        progress: (@Sendable (Float) -> Void)?
    ) async throws -> [Float] {
        guard let chunkSamples = config.chunkSamples else {
            return try await separateSingle(mixture: mixture, queryEmbedding: queryEmbedding)
        }

        let overlapSamples = Int(Float(chunkSamples) * config.chunkOverlap)
        let hopSamples = chunkSamples - overlapSamples

        // Calculate number of chunks
        let numChunks = max(1, Int(ceil(Float(mixture.count - overlapSamples) / Float(hopSamples))))

        // Prepare output buffer with overlap-add
        var output = [Float](repeating: 0, count: mixture.count)
        var weightSum = [Float](repeating: 0, count: mixture.count)

        // Generate crossfade window for overlap regions
        let fadeWindow = generateCrossfadeWindow(length: overlapSamples)

        for chunkIdx in 0..<numChunks {
            let startSample = chunkIdx * hopSamples
            let endSample = min(startSample + chunkSamples, mixture.count)

            // Extract chunk
            let chunk = Array(mixture[startSample..<endSample])

            // Process chunk
            let separatedChunk = try await separateSingle(
                mixture: chunk,
                queryEmbedding: queryEmbedding
            )

            // Apply windowing and add to output
            for i in 0..<separatedChunk.count {
                let outputIdx = startSample + i
                guard outputIdx < mixture.count else { break }

                var weight: Float = 1.0

                // Apply fade-in at chunk start (except first chunk)
                if chunkIdx > 0 && i < overlapSamples {
                    weight = fadeWindow[i]
                }

                // Apply fade-out at chunk end (except last chunk)
                if chunkIdx < numChunks - 1 && i >= separatedChunk.count - overlapSamples {
                    let fadeIdx = i - (separatedChunk.count - overlapSamples)
                    weight = 1.0 - fadeWindow[fadeIdx]
                }

                output[outputIdx] += separatedChunk[i] * weight
                weightSum[outputIdx] += weight
            }

            // Report progress
            progress?(Float(chunkIdx + 1) / Float(numChunks))
        }

        // Normalize by weight sum
        for i in 0..<output.count {
            if weightSum[i] > 0 {
                output[i] /= weightSum[i]
            }
        }

        return output
    }

    /// Generate a smooth crossfade window (half of a Hann window).
    private func generateCrossfadeWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let scale = Float.pi / Float(length)

        for i in 0..<length {
            // Fade-in: sin^2 curve (0 to 1)
            let angle = Float(i) * scale
            window[i] = sin(angle) * sin(angle)
        }

        return window
    }

    /// Apply complex mask to spectrogram using element-wise complex multiplication.
    ///
    /// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    private func applyComplexMask(mask: ComplexMatrix, to input: ComplexMatrix) -> ComplexMatrix {
        let rows = input.rows
        let cols = input.cols

        var resultReal = [[Float]]()
        var resultImag = [[Float]]()
        resultReal.reserveCapacity(rows)
        resultImag.reserveCapacity(rows)

        for f in 0..<rows {
            var realRow = [Float](repeating: 0, count: cols)
            var imagRow = [Float](repeating: 0, count: cols)

            // Use Accelerate for vectorized complex multiplication
            let a = input.real[f]  // Real part of input
            let b = input.imag[f]  // Imag part of input
            let c = mask.real[f]   // Real part of mask
            let d = mask.imag[f]   // Imag part of mask

            // ac - bd
            var ac = [Float](repeating: 0, count: cols)
            var bd = [Float](repeating: 0, count: cols)
            vDSP_vmul(a, 1, c, 1, &ac, 1, vDSP_Length(cols))
            vDSP_vmul(b, 1, d, 1, &bd, 1, vDSP_Length(cols))
            vDSP_vsub(bd, 1, ac, 1, &realRow, 1, vDSP_Length(cols))

            // ad + bc
            var ad = [Float](repeating: 0, count: cols)
            var bc = [Float](repeating: 0, count: cols)
            vDSP_vmul(a, 1, d, 1, &ad, 1, vDSP_Length(cols))
            vDSP_vmul(b, 1, c, 1, &bc, 1, vDSP_Length(cols))
            vDSP_vadd(ad, 1, bc, 1, &imagRow, 1, vDSP_Length(cols))

            resultReal.append(realRow)
            resultImag.append(imagRow)
        }

        return ComplexMatrix(real: resultReal, imag: resultImag)
    }
}

// MARK: - ComplexMatrix Extensions

extension ComplexMatrix {
    /// Element-wise complex multiplication with another matrix.
    ///
    /// - Parameter other: Matrix to multiply with.
    /// - Returns: Result of element-wise complex multiplication.
    public func complexMultiply(with other: ComplexMatrix) -> ComplexMatrix {
        precondition(rows == other.rows && cols == other.cols, "Matrix dimensions must match")

        var resultReal = [[Float]]()
        var resultImag = [[Float]]()
        resultReal.reserveCapacity(rows)
        resultImag.reserveCapacity(rows)

        for f in 0..<rows {
            var realRow = [Float](repeating: 0, count: cols)
            var imagRow = [Float](repeating: 0, count: cols)

            for t in 0..<cols {
                let a = real[f][t]
                let b = imag[f][t]
                let c = other.real[f][t]
                let d = other.imag[f][t]

                realRow[t] = a * c - b * d
                imagRow[t] = a * d + b * c
            }

            resultReal.append(realRow)
            resultImag.append(imagRow)
        }

        return ComplexMatrix(real: resultReal, imag: resultImag)
    }
}
