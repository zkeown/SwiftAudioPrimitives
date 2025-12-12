import Accelerate
import Foundation
import Metal

/// Pool of reusable MTLBuffers to reduce allocation overhead.
///
/// Buffers are organized by size class (powers of 2) for efficient reuse.
/// This significantly reduces allocation overhead for repeated Metal operations.
final class MetalBufferPool: @unchecked Sendable {
    private let device: MTLDevice
    private var pools: [Int: [MTLBuffer]] = [:]  // Size class -> available buffers
    private let maxBuffersPerClass = 4
    private let lock = NSLock()

    init(device: MTLDevice) {
        self.device = device
    }

    /// Get the size class for a given byte count (next power of 2, min 4KB).
    private func sizeClass(for bytes: Int) -> Int {
        let minSize = 4096  // 4KB minimum
        let size = max(bytes, minSize)
        // Round up to next power of 2
        var power = 1
        while power < size {
            power *= 2
        }
        return power
    }

    /// Acquire a buffer of at least the specified size.
    func acquire(minBytes: Int) -> MTLBuffer? {
        let sizeClass = sizeClass(for: minBytes)

        lock.lock()
        defer { lock.unlock() }

        // Check if we have a cached buffer
        if var available = pools[sizeClass], !available.isEmpty {
            let buffer = available.removeLast()
            pools[sizeClass] = available
            return buffer
        }

        // Create new buffer
        return device.makeBuffer(length: sizeClass, options: .storageModeShared)
    }

    /// Release a buffer back to the pool.
    func release(_ buffer: MTLBuffer) {
        let sizeClass = sizeClass(for: buffer.length)

        lock.lock()
        defer { lock.unlock() }

        var available = pools[sizeClass] ?? []
        if available.count < maxBuffersPerClass {
            available.append(buffer)
            pools[sizeClass] = available
        }
        // If pool is full, let buffer be deallocated
    }

    /// Clear all pooled buffers.
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        pools.removeAll()
    }
}

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

    /// Buffer pool for reusing MTLBuffers.
    private let bufferPool: MetalBufferPool

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
            // 1. Try pre-compiled metallib from build plugin output
            if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
                self.library = try device.makeLibrary(URL: libraryURL)
            }
            // 2. Try default library from bundle (Xcode projects)
            else if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
                self.library = lib
            }
            // 3. FAIL FAST - don't attempt runtime compilation (it hangs)
            else {
                throw MetalEngineError.libraryLoadFailed(
                    "No pre-compiled Metal library found. Ensure MetalCompilerPlugin ran during build."
                )
            }
        } catch let error as MetalEngineError {
            throw error
        } catch {
            throw MetalEngineError.libraryLoadFailed("Failed to load Metal library: \(error)")
        }

        // Initialize buffer pool
        self.bufferPool = MetalBufferPool(device: device)
    }

    /// Clear the buffer pool to release cached GPU memory.
    public func clearBufferPool() {
        bufferPool.clear()
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

        let framesBytes = flatFrames.count * MemoryLayout<Float>.size
        let windowBytes = window.count * MemoryLayout<Float>.size
        let outputBytes = outputLength * MemoryLayout<Float>.size

        // Acquire buffers from pool
        guard let framesBuffer = bufferPool.acquire(minBytes: framesBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(framesBuffer) }

        guard let windowBuffer = bufferPool.acquire(minBytes: windowBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(windowBuffer) }

        guard let outputBuffer = bufferPool.acquire(minBytes: outputBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outputBuffer) }

        // Copy data to buffers
        memcpy(framesBuffer.contents(), flatFrames, framesBytes)
        memcpy(windowBuffer.contents(), window, windowBytes)
        memset(outputBuffer.contents(), 0, outputBytes)

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

        let specBytes = flatSpec.count * MemoryLayout<Float>.size
        let filterBytes = flatFilter.count * MemoryLayout<Float>.size
        let outputBytes = nMels * nFrames * MemoryLayout<Float>.size

        // Acquire buffers from pool
        guard let specBuffer = bufferPool.acquire(minBytes: specBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(specBuffer) }

        guard let filterBuffer = bufferPool.acquire(minBytes: filterBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(filterBuffer) }

        guard let outputBuffer = bufferPool.acquire(minBytes: outputBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outputBuffer) }

        // Copy data to buffers
        memcpy(specBuffer.contents(), flatSpec, specBytes)
        memcpy(filterBuffer.contents(), flatFilter, filterBytes)

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
        let bufferBytes = count * MemoryLayout<Float>.size

        // Flatten
        let flatAR = aReal.flatMap { $0 }
        let flatAI = aImag.flatMap { $0 }
        let flatBR = bReal.flatMap { $0 }
        let flatBI = bImag.flatMap { $0 }

        // Acquire buffers from pool
        guard let arBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(arBuffer) }

        guard let aiBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(aiBuffer) }

        guard let brBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(brBuffer) }

        guard let biBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(biBuffer) }

        guard let outRBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outRBuffer) }

        guard let outIBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outIBuffer) }

        // Copy data to buffers
        memcpy(arBuffer.contents(), flatAR, bufferBytes)
        memcpy(aiBuffer.contents(), flatAI, bufferBytes)
        memcpy(brBuffer.contents(), flatBR, bufferBytes)
        memcpy(biBuffer.contents(), flatBI, bufferBytes)

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

    // MARK: - Spectral Features

    /// Compute spectral centroid using GPU.
    ///
    /// - Parameters:
    ///   - magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - frequencies: Frequency array of length nFreqs.
    /// - Returns: Spectral centroid per frame.
    public func spectralCentroid(
        magnitudeSpec: [[Float]],
        frequencies: [Float]
    ) async throws -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        // Flatten magnitude spectrogram (freq-major layout)
        let flatMag = magnitudeSpec.flatMap { $0 }

        let magBytes = flatMag.count * MemoryLayout<Float>.size
        let freqBytes = frequencies.count * MemoryLayout<Float>.size
        let outputBytes = nFrames * MemoryLayout<Float>.size

        // Acquire buffers from pool
        guard let magBuffer = bufferPool.acquire(minBytes: magBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(magBuffer) }

        guard let freqBuffer = bufferPool.acquire(minBytes: freqBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(freqBuffer) }

        guard let outputBuffer = bufferPool.acquire(minBytes: outputBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outputBuffer) }

        // Copy data to buffers
        memcpy(magBuffer.contents(), flatMag, magBytes)
        memcpy(freqBuffer.contents(), frequencies, freqBytes)

        // Constants
        var nFreqsVar = UInt32(nFreqs)
        var nFramesVar = UInt32(nFrames)

        // Get pipeline
        let pipeline = try getPipeline(for: "spectralCentroid")

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(magBuffer, offset: 0, index: 0)
        encoder.setBuffer(freqBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 4)

        // Dispatch 1D grid (one thread per frame)
        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (nFrames + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: nFrames))
    }

    /// Compute spectral flux using GPU.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Spectral flux per frame.
    public func spectralFlux(
        magnitudeSpec: [[Float]]
    ) async throws -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        // Flatten magnitude spectrogram (freq-major layout)
        let flatMag = magnitudeSpec.flatMap { $0 }

        let magBytes = flatMag.count * MemoryLayout<Float>.size
        let outputBytes = nFrames * MemoryLayout<Float>.size

        // Acquire buffers from pool
        guard let magBuffer = bufferPool.acquire(minBytes: magBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(magBuffer) }

        guard let outputBuffer = bufferPool.acquire(minBytes: outputBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer { bufferPool.release(outputBuffer) }

        // Copy data to buffer
        memcpy(magBuffer.contents(), flatMag, magBytes)

        // Constants
        var nFreqsVar = UInt32(nFreqs)
        var nFramesVar = UInt32(nFrames)

        // Get pipeline
        let pipeline = try getPipeline(for: "spectralFlux")

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(magBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 3)

        // Dispatch 1D grid (one thread per frame)
        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (nFrames + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: nFrames))
    }

    // MARK: - GPU FFT

    /// Compute batch FFT using GPU.
    ///
    /// Performs radix-2 DIT FFT on multiple signals in parallel.
    /// Input must have power-of-2 length.
    ///
    /// - Parameters:
    ///   - real: Real parts, flat array [batchSize * n].
    ///   - imag: Imaginary parts, flat array [batchSize * n].
    ///   - n: FFT size (must be power of 2).
    ///   - batchSize: Number of independent FFTs.
    /// - Returns: Tuple of (real, imag) result arrays.
    public func batchFFT(
        real: [Float],
        imag: [Float],
        n: Int,
        batchSize: Int
    ) async throws -> (real: [Float], imag: [Float]) {
        guard n > 0 && (n & (n - 1)) == 0 else {
            // n must be power of 2
            throw MetalEngineError.encodingFailed
        }

        let log2n = Int(log2(Double(n)))
        let totalElements = n * batchSize
        let bufferBytes = totalElements * MemoryLayout<Float>.size

        // Acquire buffers for input, working space, and output
        guard let inputRealBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let inputImagBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let workRealBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let workImagBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(inputRealBuffer)
            bufferPool.release(inputImagBuffer)
            bufferPool.release(workRealBuffer)
            bufferPool.release(workImagBuffer)
        }

        // Copy input data
        memcpy(inputRealBuffer.contents(), real, bufferBytes)
        memcpy(inputImagBuffer.contents(), imag, bufferBytes)

        // Get pipelines
        let bitReversePipeline = try getPipeline(for: "fftBatchBitReverse")
        let butterflyPipeline = try getPipeline(for: "fftBatchButterflyStage")

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalEngineError.commandBufferFailed
        }

        // Step 1: Bit-reversal permutation
        guard let bitReverseEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.encodingFailed
        }

        var nVar = UInt32(n)
        var log2nVar = UInt32(log2n)
        var batchSizeVar = UInt32(batchSize)

        bitReverseEncoder.setComputePipelineState(bitReversePipeline)
        bitReverseEncoder.setBuffer(inputRealBuffer, offset: 0, index: 0)
        bitReverseEncoder.setBuffer(inputImagBuffer, offset: 0, index: 1)
        bitReverseEncoder.setBuffer(workRealBuffer, offset: 0, index: 2)
        bitReverseEncoder.setBuffer(workImagBuffer, offset: 0, index: 3)
        bitReverseEncoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 4)
        bitReverseEncoder.setBytes(&log2nVar, length: MemoryLayout<UInt32>.size, index: 5)
        bitReverseEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 6)

        let threadsPerGroup2D = MTLSize(width: min(n, 256), height: 1, depth: 1)
        let threadGroups2D = MTLSize(
            width: (n + threadsPerGroup2D.width - 1) / threadsPerGroup2D.width,
            height: batchSize,
            depth: 1
        )
        bitReverseEncoder.dispatchThreadgroups(threadGroups2D, threadsPerThreadgroup: threadsPerGroup2D)
        bitReverseEncoder.endEncoding()

        // Step 2: Butterfly stages (log2n stages)
        for stage in 0..<log2n {
            guard let butterflyEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalEngineError.encodingFailed
            }

            var stageVar = UInt32(stage)

            butterflyEncoder.setComputePipelineState(butterflyPipeline)
            butterflyEncoder.setBuffer(workRealBuffer, offset: 0, index: 0)
            butterflyEncoder.setBuffer(workImagBuffer, offset: 0, index: 1)
            butterflyEncoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 2)
            butterflyEncoder.setBytes(&stageVar, length: MemoryLayout<UInt32>.size, index: 3)
            butterflyEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 4)

            // Each stage processes n/2 butterflies per batch
            let butterfliesPerBatch = n / 2
            let butterflyThreadsPerGroup = MTLSize(width: min(butterfliesPerBatch, 256), height: 1, depth: 1)
            let butterflyThreadGroups = MTLSize(
                width: (butterfliesPerBatch + butterflyThreadsPerGroup.width - 1) / butterflyThreadsPerGroup.width,
                height: batchSize,
                depth: 1
            )

            butterflyEncoder.dispatchThreadgroups(butterflyThreadGroups, threadsPerThreadgroup: butterflyThreadsPerGroup)
            butterflyEncoder.endEncoding()
        }

        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let realPtr = workRealBuffer.contents().assumingMemoryBound(to: Float.self)
        let imagPtr = workImagBuffer.contents().assumingMemoryBound(to: Float.self)

        return (
            Array(UnsafeBufferPointer(start: realPtr, count: totalElements)),
            Array(UnsafeBufferPointer(start: imagPtr, count: totalElements))
        )
    }

    /// Compute STFT magnitude using GPU FFT.
    ///
    /// This performs the entire STFT pipeline on GPU:
    /// 1. Frame extraction and windowing (CPU)
    /// 2. Batch FFT (GPU)
    /// 3. Magnitude computation (GPU)
    ///
    /// - Parameters:
    ///   - frames: Windowed frames, flat array [nFrames * nFFT].
    ///   - nFFT: FFT size (must be power of 2).
    ///   - nFrames: Number of frames.
    /// - Returns: Magnitude spectrogram as ContiguousMatrix (nFreqs × nFrames).
    public func stftMagnitude(
        frames: [Float],
        nFFT: Int,
        nFrames: Int
    ) async throws -> ContiguousMatrix {
        guard nFFT > 0 && (nFFT & (nFFT - 1)) == 0 else {
            throw MetalEngineError.encodingFailed
        }

        let nFreqs = nFFT / 2 + 1

        // Prepare imaginary part (zeros)
        let imag = [Float](repeating: 0, count: frames.count)

        // Compute batch FFT
        let (fftReal, fftImag) = try await batchFFT(
            real: frames,
            imag: imag,
            n: nFFT,
            batchSize: nFrames
        )

        // Compute magnitude on GPU
        let fftBytes = fftReal.count * MemoryLayout<Float>.size
        let magBytes = nFreqs * nFrames * MemoryLayout<Float>.size

        guard let fftRealBuffer = bufferPool.acquire(minBytes: fftBytes),
              let fftImagBuffer = bufferPool.acquire(minBytes: fftBytes),
              let magBuffer = bufferPool.acquire(minBytes: magBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(fftRealBuffer)
            bufferPool.release(fftImagBuffer)
            bufferPool.release(magBuffer)
        }

        memcpy(fftRealBuffer.contents(), fftReal, fftBytes)
        memcpy(fftImagBuffer.contents(), fftImag, fftBytes)

        let magPipeline = try getPipeline(for: "stftMagnitude")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        var nFFTVar = UInt32(nFFT)
        var nFramesVar = UInt32(nFrames)
        var nFreqsVar = UInt32(nFreqs)

        encoder.setComputePipelineState(magPipeline)
        encoder.setBuffer(fftRealBuffer, offset: 0, index: 0)
        encoder.setBuffer(fftImagBuffer, offset: 0, index: 1)
        encoder.setBuffer(magBuffer, offset: 0, index: 2)
        encoder.setBytes(&nFFTVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 5)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (nFrames + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (nFreqs + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let magPtr = magBuffer.contents().assumingMemoryBound(to: Float.self)
        let magData = Array(UnsafeBufferPointer(start: magPtr, count: nFreqs * nFrames))

        return ContiguousMatrix(data: magData, rows: nFreqs, cols: nFrames)
    }
}

// MARK: - GPU Threshold Configuration

/// Configuration for automatic GPU/CPU dispatch decisions.
///
/// Tune these thresholds based on your specific hardware and use case.
/// Lower thresholds favor GPU more aggressively, higher thresholds prefer CPU.
public struct GPUThresholdConfig: Sendable {
    /// Minimum elements for general GPU operations (default: 100,000).
    public var generalThreshold: Int

    /// Minimum elements for matrix multiply operations (default: 50,000).
    /// Matrix operations have higher arithmetic intensity, so GPU wins earlier.
    public var matrixMultiplyThreshold: Int

    /// Minimum elements for FFT-related operations (default: 32,768).
    /// FFT benefits significantly from GPU parallelism.
    public var fftThreshold: Int

    /// Minimum elements for simple element-wise operations (default: 200,000).
    /// Simple ops have low arithmetic intensity, need more data to offset overhead.
    public var elementWiseThreshold: Int

    /// Whether to disable GPU entirely and force CPU fallback.
    public var forceDisableGPU: Bool

    /// Whether to force GPU usage regardless of data size (for testing).
    public var forceEnableGPU: Bool

    /// Default configuration with balanced thresholds.
    public static let `default` = GPUThresholdConfig(
        generalThreshold: 100_000,
        matrixMultiplyThreshold: 50_000,
        fftThreshold: 32_768,
        elementWiseThreshold: 200_000,
        forceDisableGPU: false,
        forceEnableGPU: false
    )

    /// Aggressive GPU configuration - use GPU for smaller workloads.
    public static let aggressiveGPU = GPUThresholdConfig(
        generalThreshold: 10_000,
        matrixMultiplyThreshold: 5_000,
        fftThreshold: 4_096,
        elementWiseThreshold: 20_000,
        forceDisableGPU: false,
        forceEnableGPU: false
    )

    /// Conservative configuration - prefer CPU except for very large workloads.
    public static let preferCPU = GPUThresholdConfig(
        generalThreshold: 500_000,
        matrixMultiplyThreshold: 250_000,
        fftThreshold: 131_072,
        elementWiseThreshold: 1_000_000,
        forceDisableGPU: false,
        forceEnableGPU: false
    )

    /// Create custom threshold configuration.
    public init(
        generalThreshold: Int = 100_000,
        matrixMultiplyThreshold: Int = 50_000,
        fftThreshold: Int = 32_768,
        elementWiseThreshold: Int = 200_000,
        forceDisableGPU: Bool = false,
        forceEnableGPU: Bool = false
    ) {
        self.generalThreshold = generalThreshold
        self.matrixMultiplyThreshold = matrixMultiplyThreshold
        self.fftThreshold = fftThreshold
        self.elementWiseThreshold = elementWiseThreshold
        self.forceDisableGPU = forceDisableGPU
        self.forceEnableGPU = forceEnableGPU
    }
}

/// Type of operation for GPU threshold selection.
public enum GPUOperationType: Sendable {
    case general
    case matrixMultiply
    case fft
    case elementWise
}

// MARK: - CPU Fallback

extension MetalEngine {
    /// Global GPU threshold configuration.
    /// Modify this to tune GPU/CPU dispatch behavior across the library.
    public static var thresholdConfig: GPUThresholdConfig = .default

    /// Check if Metal acceleration would be beneficial for given data size.
    ///
    /// - Parameter elementCount: Number of elements to process.
    /// - Returns: True if Metal is likely to be faster than CPU.
    public static func shouldUseGPU(elementCount: Int) -> Bool {
        return shouldUseGPU(elementCount: elementCount, operationType: .general)
    }

    /// Check if Metal acceleration would be beneficial for given data size and operation type.
    ///
    /// - Parameters:
    ///   - elementCount: Number of elements to process.
    ///   - operationType: Type of operation to perform.
    /// - Returns: True if Metal is likely to be faster than CPU.
    public static func shouldUseGPU(elementCount: Int, operationType: GPUOperationType) -> Bool {
        let config = thresholdConfig

        // Honor force flags
        if config.forceDisableGPU { return false }
        if config.forceEnableGPU { return true }

        // Check availability
        guard isAvailable else { return false }

        // Select threshold based on operation type
        let threshold: Int
        switch operationType {
        case .general:
            threshold = config.generalThreshold
        case .matrixMultiply:
            threshold = config.matrixMultiplyThreshold
        case .fft:
            threshold = config.fftThreshold
        case .elementWise:
            threshold = config.elementWiseThreshold
        }

        return elementCount >= threshold
    }
}
