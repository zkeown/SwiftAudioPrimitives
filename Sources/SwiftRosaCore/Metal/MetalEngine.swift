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

    // MARK: - NMF Operations

    /// Perform NMF matrix multiplication: C = A @ B
    ///
    /// - Parameters:
    ///   - A: Matrix of shape (M, K), flattened row-major.
    ///   - B: Matrix of shape (K, N), flattened row-major.
    ///   - M: Number of rows in A.
    ///   - K: Inner dimension.
    ///   - N: Number of columns in B.
    /// - Returns: Result matrix C of shape (M, N), flattened row-major.
    public func nmfMatrixMultiply(
        A: [Float],
        B: [Float],
        M: Int,
        K: Int,
        N: Int
    ) async throws -> [Float] {
        let aBytes = A.count * MemoryLayout<Float>.size
        let bBytes = B.count * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let aBuffer = bufferPool.acquire(minBytes: aBytes),
              let bBuffer = bufferPool.acquire(minBytes: bBytes),
              let cBuffer = bufferPool.acquire(minBytes: cBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(aBuffer)
            bufferPool.release(bBuffer)
            bufferPool.release(cBuffer)
        }

        memcpy(aBuffer.contents(), A, aBytes)
        memcpy(bBuffer.contents(), B, bBytes)

        var mVar = UInt32(M)
        var kVar = UInt32(K)
        var nVar = UInt32(N)

        let pipeline = try getPipeline(for: "nmfMatrixMultiply")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (N + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (M + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let cPtr = cBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: cPtr, count: M * N))
    }

    /// Perform NMF multiplicative update for W matrix on GPU.
    ///
    /// W_new = W * (V @ H^T) / (W @ H @ H^T + eps)
    ///
    /// - Parameters:
    ///   - W: Current W matrix (M × K), flattened row-major.
    ///   - V: Input matrix (M × N), flattened row-major.
    ///   - H: Current H matrix (K × N), flattened row-major.
    ///   - WH: Pre-computed W @ H (M × N), flattened row-major.
    ///   - M: Number of rows in V/W.
    ///   - N: Number of columns in V/H.
    ///   - K: Number of components.
    ///   - epsilon: Small constant for numerical stability.
    /// - Returns: Updated W matrix.
    public func nmfUpdateW(
        W: [Float],
        V: [Float],
        H: [Float],
        WH: [Float],
        M: Int,
        N: Int,
        K: Int,
        epsilon: Float
    ) async throws -> [Float] {
        let wBytes = W.count * MemoryLayout<Float>.size
        let vBytes = V.count * MemoryLayout<Float>.size
        let hBytes = H.count * MemoryLayout<Float>.size
        let whBytes = WH.count * MemoryLayout<Float>.size

        guard let wBuffer = bufferPool.acquire(minBytes: wBytes),
              let vBuffer = bufferPool.acquire(minBytes: vBytes),
              let hBuffer = bufferPool.acquire(minBytes: hBytes),
              let whBuffer = bufferPool.acquire(minBytes: whBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(wBuffer)
            bufferPool.release(vBuffer)
            bufferPool.release(hBuffer)
            bufferPool.release(whBuffer)
        }

        memcpy(wBuffer.contents(), W, wBytes)
        memcpy(vBuffer.contents(), V, vBytes)
        memcpy(hBuffer.contents(), H, hBytes)
        memcpy(whBuffer.contents(), WH, whBytes)

        var mVar = UInt32(M)
        var nVar = UInt32(N)
        var kVar = UInt32(K)
        var epsVar = epsilon

        let pipeline = try getPipeline(for: "nmfUpdateW")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(wBuffer, offset: 0, index: 0)
        encoder.setBuffer(vBuffer, offset: 0, index: 1)
        encoder.setBuffer(hBuffer, offset: 0, index: 2)
        encoder.setBuffer(whBuffer, offset: 0, index: 3)
        encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 7)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (K + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (M + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: wPtr, count: M * K))
    }

    /// Perform NMF multiplicative update for H matrix on GPU.
    ///
    /// H_new = H * (W^T @ V) / (W^T @ W @ H + eps)
    ///
    /// - Parameters:
    ///   - W: Current W matrix (M × K), flattened row-major.
    ///   - V: Input matrix (M × N), flattened row-major.
    ///   - H: Current H matrix (K × N), flattened row-major.
    ///   - WH: Pre-computed W @ H (M × N), flattened row-major.
    ///   - M: Number of rows in V/W.
    ///   - N: Number of columns in V/H.
    ///   - K: Number of components.
    ///   - epsilon: Small constant for numerical stability.
    /// - Returns: Updated H matrix.
    public func nmfUpdateH(
        W: [Float],
        V: [Float],
        H: [Float],
        WH: [Float],
        M: Int,
        N: Int,
        K: Int,
        epsilon: Float
    ) async throws -> [Float] {
        let wBytes = W.count * MemoryLayout<Float>.size
        let vBytes = V.count * MemoryLayout<Float>.size
        let hBytes = H.count * MemoryLayout<Float>.size
        let whBytes = WH.count * MemoryLayout<Float>.size

        guard let wBuffer = bufferPool.acquire(minBytes: wBytes),
              let vBuffer = bufferPool.acquire(minBytes: vBytes),
              let hBuffer = bufferPool.acquire(minBytes: hBytes),
              let whBuffer = bufferPool.acquire(minBytes: whBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(wBuffer)
            bufferPool.release(vBuffer)
            bufferPool.release(hBuffer)
            bufferPool.release(whBuffer)
        }

        memcpy(wBuffer.contents(), W, wBytes)
        memcpy(vBuffer.contents(), V, vBytes)
        memcpy(hBuffer.contents(), H, hBytes)
        memcpy(whBuffer.contents(), WH, whBytes)

        var mVar = UInt32(M)
        var nVar = UInt32(N)
        var kVar = UInt32(K)
        var epsVar = epsilon

        let pipeline = try getPipeline(for: "nmfUpdateH")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(wBuffer, offset: 0, index: 0)
        encoder.setBuffer(vBuffer, offset: 0, index: 1)
        encoder.setBuffer(hBuffer, offset: 0, index: 2)
        encoder.setBuffer(whBuffer, offset: 0, index: 3)
        encoder.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 7)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (N + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (K + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let hPtr = hBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: hPtr, count: K * N))
    }

    /// Compute NMF reconstruction error (Frobenius norm) on GPU.
    ///
    /// - Parameters:
    ///   - V: Original matrix, flattened.
    ///   - WH: Reconstruction (W @ H), flattened.
    /// - Returns: Sum of squared differences.
    public func nmfReconstructionError(
        V: [Float],
        WH: [Float]
    ) async throws -> Float {
        let count = V.count
        let bufferBytes = count * MemoryLayout<Float>.size
        let errorBytes = MemoryLayout<Float>.size

        guard let vBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let whBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let errorBuffer = bufferPool.acquire(minBytes: errorBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(vBuffer)
            bufferPool.release(whBuffer)
            bufferPool.release(errorBuffer)
        }

        memcpy(vBuffer.contents(), V, bufferBytes)
        memcpy(whBuffer.contents(), WH, bufferBytes)
        memset(errorBuffer.contents(), 0, errorBytes)

        var countVar = UInt32(count)

        let pipeline = try getPipeline(for: "nmfReconstructionError")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vBuffer, offset: 0, index: 0)
        encoder.setBuffer(whBuffer, offset: 0, index: 1)
        encoder.setBuffer(errorBuffer, offset: 0, index: 2)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let errorPtr = errorBuffer.contents().assumingMemoryBound(to: Float.self)
        return errorPtr.pointee
    }

    // MARK: - Distance Matrix

    /// Compute Euclidean distance matrix between two feature sets on GPU.
    ///
    /// - Parameters:
    ///   - xFeatures: First feature matrix (nFeatures × nX), flattened feature-major.
    ///   - yFeatures: Second feature matrix (nFeatures × nY), flattened feature-major.
    ///   - nFeatures: Number of features.
    ///   - nX: Number of samples in X.
    ///   - nY: Number of samples in Y.
    /// - Returns: Distance matrix (nX × nY), flattened row-major.
    public func euclideanDistanceMatrix(
        xFeatures: [Float],
        yFeatures: [Float],
        nFeatures: Int,
        nX: Int,
        nY: Int
    ) async throws -> [Float] {
        let xBytes = xFeatures.count * MemoryLayout<Float>.size
        let yBytes = yFeatures.count * MemoryLayout<Float>.size
        let distBytes = nX * nY * MemoryLayout<Float>.size

        guard let xBuffer = bufferPool.acquire(minBytes: xBytes),
              let yBuffer = bufferPool.acquire(minBytes: yBytes),
              let distBuffer = bufferPool.acquire(minBytes: distBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(xBuffer)
            bufferPool.release(yBuffer)
            bufferPool.release(distBuffer)
        }

        memcpy(xBuffer.contents(), xFeatures, xBytes)
        memcpy(yBuffer.contents(), yFeatures, yBytes)

        var nFeaturesVar = UInt32(nFeatures)
        var nXVar = UInt32(nX)
        var nYVar = UInt32(nY)

        let pipeline = try getPipeline(for: "euclideanDistanceMatrix")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(yBuffer, offset: 0, index: 1)
        encoder.setBuffer(distBuffer, offset: 0, index: 2)
        encoder.setBytes(&nFeaturesVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nXVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nYVar, length: MemoryLayout<UInt32>.size, index: 5)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (nX + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (nY + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let distPtr = distBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: distPtr, count: nX * nY))
    }

    // MARK: - CQT Operations

    /// Compute CQT correlation for a single octave using GPU.
    ///
    /// - Parameters:
    ///   - signal: Padded signal.
    ///   - filterReal: All filter real parts concatenated.
    ///   - filterImag: All filter imaginary parts concatenated.
    ///   - filterOffsets: Start offset in filterReal/Imag for each bin.
    ///   - filterLengths: Length of each filter.
    ///   - filterNorms: L1 norm for each filter.
    ///   - normFactors: sqrt(filterLength) * sqrt(downsampleFactor) for each bin.
    ///   - nBins: Number of frequency bins.
    ///   - nFrames: Number of output frames.
    ///   - hopLength: Hop length between frames.
    ///   - padLength: Padding applied to signal.
    /// - Returns: Tuple of (real, imag) output arrays [nBins × nFrames].
    public func cqtCorrelation(
        signal: [Float],
        filterReal: [Float],
        filterImag: [Float],
        filterOffsets: [UInt32],
        filterLengths: [UInt32],
        filterNorms: [Float],
        normFactors: [Float],
        nBins: Int,
        nFrames: Int,
        hopLength: Int,
        padLength: Int
    ) async throws -> (real: [Float], imag: [Float]) {
        let signalBytes = signal.count * MemoryLayout<Float>.size
        let filterBytes = filterReal.count * MemoryLayout<Float>.size
        let offsetBytes = filterOffsets.count * MemoryLayout<UInt32>.size
        let lengthBytes = filterLengths.count * MemoryLayout<UInt32>.size
        let normBytes = filterNorms.count * MemoryLayout<Float>.size
        let outputBytes = nBins * nFrames * MemoryLayout<Float>.size

        guard let signalBuffer = bufferPool.acquire(minBytes: signalBytes),
              let filterRealBuffer = bufferPool.acquire(minBytes: filterBytes),
              let filterImagBuffer = bufferPool.acquire(minBytes: filterBytes),
              let offsetBuffer = bufferPool.acquire(minBytes: offsetBytes),
              let lengthBuffer = bufferPool.acquire(minBytes: lengthBytes),
              let normBuffer = bufferPool.acquire(minBytes: normBytes),
              let factorBuffer = bufferPool.acquire(minBytes: normBytes),
              let outRealBuffer = bufferPool.acquire(minBytes: outputBytes),
              let outImagBuffer = bufferPool.acquire(minBytes: outputBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(signalBuffer)
            bufferPool.release(filterRealBuffer)
            bufferPool.release(filterImagBuffer)
            bufferPool.release(offsetBuffer)
            bufferPool.release(lengthBuffer)
            bufferPool.release(normBuffer)
            bufferPool.release(factorBuffer)
            bufferPool.release(outRealBuffer)
            bufferPool.release(outImagBuffer)
        }

        memcpy(signalBuffer.contents(), signal, signalBytes)
        memcpy(filterRealBuffer.contents(), filterReal, filterBytes)
        memcpy(filterImagBuffer.contents(), filterImag, filterBytes)
        memcpy(offsetBuffer.contents(), filterOffsets, offsetBytes)
        memcpy(lengthBuffer.contents(), filterLengths, lengthBytes)
        memcpy(normBuffer.contents(), filterNorms, normBytes)
        memcpy(factorBuffer.contents(), normFactors, normBytes)

        var nBinsVar = UInt32(nBins)
        var nFramesVar = UInt32(nFrames)
        var hopLengthVar = UInt32(hopLength)
        var padLengthVar = UInt32(padLength)
        var signalLengthVar = UInt32(signal.count)

        let pipeline = try getPipeline(for: "cqtCorrelation")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(signalBuffer, offset: 0, index: 0)
        encoder.setBuffer(filterRealBuffer, offset: 0, index: 1)
        encoder.setBuffer(filterImagBuffer, offset: 0, index: 2)
        encoder.setBuffer(offsetBuffer, offset: 0, index: 3)
        encoder.setBuffer(lengthBuffer, offset: 0, index: 4)
        encoder.setBuffer(normBuffer, offset: 0, index: 5)
        encoder.setBuffer(factorBuffer, offset: 0, index: 6)
        encoder.setBuffer(outRealBuffer, offset: 0, index: 7)
        encoder.setBuffer(outImagBuffer, offset: 0, index: 8)
        encoder.setBytes(&nBinsVar, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.setBytes(&hopLengthVar, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&padLengthVar, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.setBytes(&signalLengthVar, length: MemoryLayout<UInt32>.size, index: 13)

        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (nFrames + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (nBins + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let realPtr = outRealBuffer.contents().assumingMemoryBound(to: Float.self)
        let imagPtr = outImagBuffer.contents().assumingMemoryBound(to: Float.self)

        return (
            Array(UnsafeBufferPointer(start: realPtr, count: nBins * nFrames)),
            Array(UnsafeBufferPointer(start: imagPtr, count: nBins * nFrames))
        )
    }

    // MARK: - HPSS Operations

    /// Apply horizontal median filter (along time axis) using GPU.
    ///
    /// - Parameters:
    ///   - input: Input matrix flattened row-major [nFreqs × nFrames].
    ///   - nFreqs: Number of frequency bins.
    ///   - nFrames: Number of time frames.
    ///   - kernelSize: Median filter kernel size.
    /// - Returns: Filtered output.
    public func hpssMedianFilterHorizontal(
        input: [Float],
        nFreqs: Int,
        nFrames: Int,
        kernelSize: Int
    ) async throws -> [Float] {
        let bufferBytes = input.count * MemoryLayout<Float>.size

        guard let inputBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let outputBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(inputBuffer.contents(), input, bufferBytes)

        var nFreqsVar = UInt32(nFreqs)
        var nFramesVar = UInt32(nFrames)
        var kernelVar = UInt32(kernelSize)

        let pipeline = try getPipeline(for: "hpssMedianFilterHorizontal")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&kernelVar, length: MemoryLayout<UInt32>.size, index: 4)

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

        let outPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outPtr, count: input.count))
    }

    /// Apply vertical median filter (along frequency axis) using GPU.
    public func hpssMedianFilterVertical(
        input: [Float],
        nFreqs: Int,
        nFrames: Int,
        kernelSize: Int
    ) async throws -> [Float] {
        let bufferBytes = input.count * MemoryLayout<Float>.size

        guard let inputBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let outputBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(inputBuffer.contents(), input, bufferBytes)

        var nFreqsVar = UInt32(nFreqs)
        var nFramesVar = UInt32(nFrames)
        var kernelVar = UInt32(kernelSize)

        let pipeline = try getPipeline(for: "hpssMedianFilterVertical")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&nFreqsVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&nFramesVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&kernelVar, length: MemoryLayout<UInt32>.size, index: 4)

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

        let outPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outPtr, count: input.count))
    }

    /// Compute HPSS soft masks using GPU.
    ///
    /// - Parameters:
    ///   - harmonicEnhanced: Harmonic-enhanced spectrogram, flattened.
    ///   - percussiveEnhanced: Percussive-enhanced spectrogram, flattened.
    ///   - power: Mask power (2.0 for Wiener filtering).
    ///   - margin: Separation margin.
    ///   - epsilon: Numerical stability constant.
    /// - Returns: Tuple of (harmonicMask, percussiveMask).
    public func hpssSoftMask(
        harmonicEnhanced: [Float],
        percussiveEnhanced: [Float],
        power: Float,
        margin: Float,
        epsilon: Float
    ) async throws -> (harmonic: [Float], percussive: [Float]) {
        let count = harmonicEnhanced.count
        let bufferBytes = count * MemoryLayout<Float>.size

        guard let hBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let pBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let maskHBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let maskPBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(hBuffer)
            bufferPool.release(pBuffer)
            bufferPool.release(maskHBuffer)
            bufferPool.release(maskPBuffer)
        }

        memcpy(hBuffer.contents(), harmonicEnhanced, bufferBytes)
        memcpy(pBuffer.contents(), percussiveEnhanced, bufferBytes)

        var countVar = UInt32(count)
        var powerVar = power
        var marginVar = margin
        var epsVar = epsilon

        let pipeline = try getPipeline(for: "hpssSoftMask")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(hBuffer, offset: 0, index: 0)
        encoder.setBuffer(pBuffer, offset: 0, index: 1)
        encoder.setBuffer(maskHBuffer, offset: 0, index: 2)
        encoder.setBuffer(maskPBuffer, offset: 0, index: 3)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&powerVar, length: MemoryLayout<Float>.size, index: 5)
        encoder.setBytes(&marginVar, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 7)

        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let maskHPtr = maskHBuffer.contents().assumingMemoryBound(to: Float.self)
        let maskPPtr = maskPBuffer.contents().assumingMemoryBound(to: Float.self)

        return (
            Array(UnsafeBufferPointer(start: maskHPtr, count: count)),
            Array(UnsafeBufferPointer(start: maskPPtr, count: count))
        )
    }

    // MARK: - Griffin-Lim Operations

    /// Perform Griffin-Lim magnitude replacement using GPU.
    ///
    /// - Parameters:
    ///   - currentReal: Current STFT real part, flattened.
    ///   - currentImag: Current STFT imaginary part, flattened.
    ///   - targetMagnitude: Target magnitude spectrogram, flattened.
    ///   - epsilon: Numerical stability constant.
    /// - Returns: Tuple of (newReal, newImag) with target magnitude and preserved phase.
    public func griffinLimMagnitudeReplace(
        currentReal: [Float],
        currentImag: [Float],
        targetMagnitude: [Float],
        epsilon: Float
    ) async throws -> (real: [Float], imag: [Float]) {
        let count = currentReal.count
        let bufferBytes = count * MemoryLayout<Float>.size

        guard let realBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let imagBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let magBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let outRealBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let outImagBuffer = bufferPool.acquire(minBytes: bufferBytes) else {
            throw MetalEngineError.bufferCreationFailed
        }
        defer {
            bufferPool.release(realBuffer)
            bufferPool.release(imagBuffer)
            bufferPool.release(magBuffer)
            bufferPool.release(outRealBuffer)
            bufferPool.release(outImagBuffer)
        }

        memcpy(realBuffer.contents(), currentReal, bufferBytes)
        memcpy(imagBuffer.contents(), currentImag, bufferBytes)
        memcpy(magBuffer.contents(), targetMagnitude, bufferBytes)

        var countVar = UInt32(count)
        var epsVar = epsilon

        let pipeline = try getPipeline(for: "griffinLimMagnitudeReplace")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.commandBufferFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(realBuffer, offset: 0, index: 0)
        encoder.setBuffer(imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(magBuffer, offset: 0, index: 2)
        encoder.setBuffer(outRealBuffer, offset: 0, index: 3)
        encoder.setBuffer(outImagBuffer, offset: 0, index: 4)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 6)

        let threadsPerGroup = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outRealPtr = outRealBuffer.contents().assumingMemoryBound(to: Float.self)
        let outImagPtr = outImagBuffer.contents().assumingMemoryBound(to: Float.self)

        return (
            Array(UnsafeBufferPointer(start: outRealPtr, count: count)),
            Array(UnsafeBufferPointer(start: outImagPtr, count: count))
        )
    }
}
