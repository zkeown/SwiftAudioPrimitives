import Accelerate
import Foundation
import Metal

/// Errors that can occur during Metal operations.
public enum MetalEngineError: Error, Sendable {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case libraryLoadFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferCreationFailed
    case commandBufferFailed
    case encodingFailed
}

/// GPU-accelerated audio DSP operations using Metal compute shaders.
///
/// `MetalEngine` provides GPU-accelerated implementations of computationally
/// intensive operations like overlap-add, matrix multiplication for mel
/// filterbanks, and complex number operations.
///
/// ## When to Use Metal
///
/// Metal acceleration is most beneficial for:
/// - Long audio files (> 30 seconds)
/// - Batch processing of multiple files
/// - Real-time processing pipelines
///
/// For shorter audio, CPU-based Accelerate framework may be faster due to
/// GPU dispatch overhead.
///
/// ## Example Usage
///
/// ```swift
/// guard MetalEngine.isAvailable else {
///     // Fall back to CPU implementation
///     return
/// }
///
/// let engine = try MetalEngine()
///
/// // GPU-accelerated overlap-add
/// let output = try await engine.overlapAdd(
///     frames: stftFrames,
///     window: hannWindow,
///     hopLength: 512,
///     outputLength: signalLength
/// )
/// ```
public actor MetalEngine {
    /// The Metal device.
    private let device: MTLDevice

    /// Command queue for dispatching work.
    private let commandQueue: MTLCommandQueue

    /// Shader library.
    private let library: MTLLibrary

    /// Cached pipeline states.
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    // MARK: - Initialization

    /// Check if Metal is available on this device.
    public static var isAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }

    /// Create a new Metal engine.
    ///
    /// - Throws: `MetalEngineError` if Metal is not available or setup fails.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalEngineError.deviceNotAvailable
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalEngineError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue

        // Load shader library - try multiple approaches
        do {
            // 1. Try pre-compiled metallib (fastest)
            if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
                self.library = try device.makeLibrary(URL: libraryURL)
            }
            // 2. Try default library from bundle
            else if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
                self.library = lib
            }
            // 3. Compile from source at runtime
            else if let sourceURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal") {
                let source = try String(contentsOf: sourceURL, encoding: .utf8)
                self.library = try device.makeLibrary(source: source, options: nil)
            }
            else {
                throw MetalEngineError.libraryLoadFailed("No Metal shader source found in bundle")
            }
        } catch let error as MetalEngineError {
            throw error
        } catch {
            throw MetalEngineError.libraryLoadFailed("Failed to load Metal library: \(error)")
        }
    }

    // MARK: - Pipeline Management

    /// Get or create a compute pipeline for a kernel function.
    private func getPipeline(for functionName: String) throws -> MTLComputePipelineState {
        if let cached = pipelineCache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw MetalEngineError.functionNotFound(functionName)
        }

        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelineCache[functionName] = pipeline
            return pipeline
        } catch {
            throw MetalEngineError.pipelineCreationFailed("Failed to create pipeline for \(functionName): \(error)")
        }
    }

    // MARK: - Overlap-Add

    /// Perform overlap-add reconstruction using GPU.
    ///
    /// - Parameters:
    ///   - frames: Array of ISTFT output frames.
    ///   - window: Synthesis window.
    ///   - hopLength: Hop between frames.
    ///   - outputLength: Expected output signal length.
    /// - Returns: Reconstructed signal.
    public func overlapAdd(
        frames: [[Float]],
        window: [Float],
        hopLength: Int,
        outputLength: Int
    ) async throws -> [Float] {
        guard !frames.isEmpty else { return [] }

        let frameCount = frames.count
        let frameLength = frames[0].count

        // Flatten frames for GPU transfer
        let flatFrames = frames.flatMap { $0 }

        // Create buffers
        guard let framesBuffer = device.makeBuffer(
            bytes: flatFrames,
            length: flatFrames.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalEngineError.bufferCreationFailed
        }

        guard let windowBuffer = device.makeBuffer(
            bytes: window,
            length: window.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalEngineError.bufferCreationFailed
        }

        // Output buffer initialized to zero
        guard let outputBuffer = device.makeBuffer(
            length: outputLength * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalEngineError.bufferCreationFailed
        }
        memset(outputBuffer.contents(), 0, outputLength * MemoryLayout<Float>.size)

        // Constants
        var frameCountVar = UInt32(frameCount)
        var frameLengthVar = UInt32(frameLength)
        var hopLengthVar = UInt32(hopLength)

        // Get pipeline
        let pipeline = try getPipeline(for: "overlapAdd")

        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(framesBuffer, offset: 0, index: 0)
        encoder.setBuffer(windowBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&frameCountVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&frameLengthVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&hopLengthVar, length: MemoryLayout<UInt32>.size, index: 5)

        // Dispatch threads
        let threadsPerGroup = MTLSize(width: min(frameLength, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (frameLength + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: frameCount,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: outputLength))
    }

    // MARK: - Mel Filterbank

    /// Apply mel filterbank to spectrogram using GPU matrix multiply.
    ///
    /// - Parameters:
    ///   - spectrogram: Power spectrogram of shape (nFreqs, nFrames).
    ///   - filterbank: Mel filterbank matrix of shape (nMels, nFreqs).
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func applyMelFilterbank(
        spectrogram: [[Float]],
        filterbank: [[Float]]
    ) async throws -> [[Float]] {
        guard !spectrogram.isEmpty && !filterbank.isEmpty else { return [] }

        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count
        let nMels = filterbank.count

        // Flatten for GPU
        let flatSpec = spectrogram.flatMap { $0 }
        let flatFilter = filterbank.flatMap { $0 }

        // Create buffers
        guard let specBuffer = device.makeBuffer(
            bytes: flatSpec,
            length: flatSpec.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let filterBuffer = device.makeBuffer(
            bytes: flatFilter,
            length: flatFilter.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ),
        let outputBuffer = device.makeBuffer(
            length: nMels * nFrames * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalEngineError.bufferCreationFailed
        }

        // Constants
        var nFreqsVar = UInt32(nFreqs)
        var nMelsVar = UInt32(nMels)
        var nFramesVar = UInt32(nFrames)

        // Get pipeline
        let pipeline = try getPipeline(for: "applyMelFilterbank")

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(specBuffer, offset: 0, index: 0)
        encoder.setBuffer(filterBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nMelsVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 5)

        // Dispatch 2D grid
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (nFrames + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (nMels + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Reshape output
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let flat = Array(UnsafeBufferPointer(start: outputPtr, count: nMels * nFrames))

        var result = [[Float]]()
        result.reserveCapacity(nMels)
        for m in 0..<nMels {
            let start = m * nFrames
            result.append(Array(flat[start..<(start + nFrames)]))
        }

        return result
    }

    // MARK: - Complex Multiplication

    /// Element-wise complex multiplication using GPU.
    ///
    /// - Parameters:
    ///   - aReal: Real part of first operand.
    ///   - aImag: Imaginary part of first operand.
    ///   - bReal: Real part of second operand.
    ///   - bImag: Imaginary part of second operand.
    /// - Returns: Tuple of (real, imaginary) result matrices.
    public func complexMultiply(
        aReal: [[Float]],
        aImag: [[Float]],
        bReal: [[Float]],
        bImag: [[Float]]
    ) async throws -> (real: [[Float]], imag: [[Float]]) {
        guard !aReal.isEmpty && !aReal[0].isEmpty else {
            return ([], [])
        }

        let rows = aReal.count
        let cols = aReal[0].count
        let count = rows * cols

        // Flatten
        let flatAR = aReal.flatMap { $0 }
        let flatAI = aImag.flatMap { $0 }
        let flatBR = bReal.flatMap { $0 }
        let flatBI = bImag.flatMap { $0 }

        // Create buffers
        guard let arBuffer = device.makeBuffer(bytes: flatAR, length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let aiBuffer = device.makeBuffer(bytes: flatAI, length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let brBuffer = device.makeBuffer(bytes: flatBR, length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let biBuffer = device.makeBuffer(bytes: flatBI, length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let outRBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let outIBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw MetalEngineError.bufferCreationFailed
        }

        var countVar = UInt32(count)

        let pipeline = try getPipeline(for: "complexMultiply")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(arBuffer, offset: 0, index: 0)
        encoder.setBuffer(aiBuffer, offset: 0, index: 1)
        encoder.setBuffer(brBuffer, offset: 0, index: 2)
        encoder.setBuffer(biBuffer, offset: 0, index: 3)
        encoder.setBuffer(outRBuffer, offset: 0, index: 4)
        encoder.setBuffer(outIBuffer, offset: 0, index: 5)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 6)

        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Reshape output
        let outRPtr = outRBuffer.contents().assumingMemoryBound(to: Float.self)
        let outIPtr = outIBuffer.contents().assumingMemoryBound(to: Float.self)

        let flatOutR = Array(UnsafeBufferPointer(start: outRPtr, count: count))
        let flatOutI = Array(UnsafeBufferPointer(start: outIPtr, count: count))

        var resultReal = [[Float]]()
        var resultImag = [[Float]]()
        resultReal.reserveCapacity(rows)
        resultImag.reserveCapacity(rows)

        for r in 0..<rows {
            let start = r * cols
            resultReal.append(Array(flatOutR[start..<(start + cols)]))
            resultImag.append(Array(flatOutI[start..<(start + cols)]))
        }

        return (resultReal, resultImag)
    }
}

// MARK: - CPU Fallback

extension MetalEngine {
    /// Check if Metal acceleration would be beneficial for given data size.
    ///
    /// - Parameter elementCount: Number of elements to process.
    /// - Returns: True if Metal is likely to be faster than CPU.
    public static func shouldUseGPU(elementCount: Int) -> Bool {
        // Rough heuristic: GPU overhead is ~100μs, so need enough work to amortize
        // At ~10 GFLOPS throughput, need at least 1M elements for 100μs of compute
        return elementCount > 100_000
    }
}
