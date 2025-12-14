//
//  LSTMEngine.swift
//  SwiftRosaNN
//
//  Metal-accelerated LSTM operations using MPS and custom kernels.
//

import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

/// Errors that can occur during LSTM Metal operations.
public enum LSTMEngineError: Error, Sendable {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case libraryLoadFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferCreationFailed
    case commandBufferFailed
    case encodingFailed
    case mpsOperationFailed(String)
}

/// Pool of reusable MTLBuffers to reduce allocation overhead.
///
/// Buffers are organized by size class (powers of 2) for efficient reuse.
final class LSTMBufferPool: @unchecked Sendable {
    private let device: MTLDevice
    private var pools: [Int: [MTLBuffer]] = [:]
    private let maxBuffersPerClass = 4
    private let lock = NSLock()

    init(device: MTLDevice) {
        self.device = device
    }

    /// Get the size class for a given byte count (next power of 2, min 4KB).
    private func sizeClass(for bytes: Int) -> Int {
        let minSize = 4096
        let size = max(bytes, minSize)
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

        if var available = pools[sizeClass], !available.isEmpty {
            let buffer = available.removeLast()
            pools[sizeClass] = available
            return buffer
        }

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
    }

    /// Clear all pooled buffers.
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        pools.removeAll()
    }
}

/// Metal-accelerated LSTM operations.
///
/// Uses MPS for matrix multiplications and custom kernels for fused state updates.
/// Manages GPU buffers and command buffer batching for efficiency.
///
/// ## Example Usage
///
/// ```swift
/// let engine = try LSTMEngine()
/// let weightMatrix = try await engine.createWeightMatrix(weights, rows: 1024, cols: 256)
///
/// let (h, c) = try await engine.lstmCellForward(
///     input: inputData,
///     hPrev: hiddenState,
///     cPrev: cellState,
///     weightIH: weightIHMatrix,
///     weightHH: weightHHMatrix,
///     biasIH: biasIHData,
///     biasHH: biasHHData,
///     batchSize: 1,
///     inputSize: 256,
///     hiddenSize: 256
/// )
/// ```
public actor LSTMEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private let bufferPool: LSTMBufferPool

    // Cached pipeline states
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    /// Check if Metal is available on this device.
    public static var isAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }

    /// Create a new LSTM engine.
    ///
    /// - Throws: `LSTMEngineError` if Metal is not available or setup fails.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LSTMEngineError.deviceNotAvailable
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw LSTMEngineError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue

        // Load shader library
        do {
            if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
                self.library = try device.makeLibrary(URL: libraryURL)
            } else if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
                self.library = lib
            } else {
                throw LSTMEngineError.libraryLoadFailed(
                    "No pre-compiled Metal library found. Ensure Metal shaders are compiled."
                )
            }
        } catch let error as LSTMEngineError {
            throw error
        } catch {
            throw LSTMEngineError.libraryLoadFailed("Failed to load Metal library: \(error)")
        }

        self.bufferPool = LSTMBufferPool(device: device)
    }

    /// Clear the buffer pool to release cached GPU memory.
    public func clearBufferPool() {
        bufferPool.clear()
    }

    // MARK: - Pipeline Management

    private func getPipeline(for functionName: String) throws -> MTLComputePipelineState {
        if let cached = pipelineCache[functionName] {
            return cached
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw LSTMEngineError.functionNotFound(functionName)
        }

        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelineCache[functionName] = pipeline
            return pipeline
        } catch {
            throw LSTMEngineError.pipelineCreationFailed(
                "Failed to create pipeline for \(functionName): \(error)"
            )
        }
    }

    // MARK: - MPS Matrix Operations

    /// Create an MPS matrix from weight data.
    ///
    /// - Parameters:
    ///   - weights: Weight values (row-major).
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    /// - Returns: MPS matrix ready for matrix multiplication.
    public func createWeightMatrix(_ weights: [Float], rows: Int, cols: Int) throws -> MPSMatrix {
        let bufferSize = weights.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: weights,
            length: bufferSize,
            options: .storageModeShared
        ) else {
            throw LSTMEngineError.bufferCreationFailed
        }

        let descriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: cols,
            rowBytes: cols * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        return MPSMatrix(buffer: buffer, descriptor: descriptor)
    }

    // MARK: - LSTM Cell Forward

    /// Perform single LSTM cell forward pass.
    ///
    /// Computes one timestep of LSTM:
    /// 1. Gate activations: gates = W_ih @ x + W_hh @ h_prev + bias
    /// 2. State update: (h_new, c_new) = lstm_update(gates, c_prev)
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - hPrev: Previous hidden state [batchSize, hiddenSize].
    ///   - cPrev: Previous cell state [batchSize, hiddenSize].
    ///   - weightIH: Input-to-hidden weight matrix (MPS).
    ///   - weightHH: Hidden-to-hidden weight matrix (MPS).
    ///   - biasIH: Input-to-hidden bias [4*hiddenSize], or nil.
    ///   - biasHH: Hidden-to-hidden bias [4*hiddenSize], or nil.
    ///   - batchSize: Batch dimension.
    ///   - inputSize: Input feature dimension.
    ///   - hiddenSize: Hidden state dimension.
    /// - Returns: Tuple of (new hidden state, new cell state).
    public func lstmCellForward(
        input: [Float],
        hPrev: [Float],
        cPrev: [Float],
        weightIH: MPSMatrix,
        weightHH: MPSMatrix,
        biasIH: [Float]?,
        biasHH: [Float]?,
        batchSize: Int,
        inputSize: Int,
        hiddenSize: Int
    ) async throws -> (h: [Float], c: [Float]) {
        let gateSize = 4 * hiddenSize

        // Calculate buffer sizes
        let inputBytes = batchSize * inputSize * MemoryLayout<Float>.stride
        let hPrevBytes = batchSize * hiddenSize * MemoryLayout<Float>.stride
        let cPrevBytes = hPrevBytes
        let gatesBytes = batchSize * gateSize * MemoryLayout<Float>.stride

        // Acquire buffers
        guard let inputBuffer = bufferPool.acquire(minBytes: inputBytes),
              let hPrevBuffer = bufferPool.acquire(minBytes: hPrevBytes),
              let cPrevBuffer = bufferPool.acquire(minBytes: cPrevBytes),
              let gatesIHBuffer = bufferPool.acquire(minBytes: gatesBytes),
              let gatesHHBuffer = bufferPool.acquire(minBytes: gatesBytes),
              let gatesBuffer = bufferPool.acquire(minBytes: gatesBytes),
              let hOutBuffer = bufferPool.acquire(minBytes: hPrevBytes),
              let cOutBuffer = bufferPool.acquire(minBytes: cPrevBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(hPrevBuffer)
            bufferPool.release(cPrevBuffer)
            bufferPool.release(gatesIHBuffer)
            bufferPool.release(gatesHHBuffer)
            bufferPool.release(gatesBuffer)
            bufferPool.release(hOutBuffer)
            bufferPool.release(cOutBuffer)
        }

        // Copy input data
        memcpy(inputBuffer.contents(), input, input.count * MemoryLayout<Float>.stride)
        memcpy(hPrevBuffer.contents(), hPrev, hPrev.count * MemoryLayout<Float>.stride)
        memcpy(cPrevBuffer.contents(), cPrev, cPrev.count * MemoryLayout<Float>.stride)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw LSTMEngineError.commandBufferFailed
        }

        // Create MPS matrices for inputs
        let inputMatrix = MPSMatrix(
            buffer: inputBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: inputSize,
                rowBytes: inputSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let hPrevMatrix = MPSMatrix(
            buffer: hPrevBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: hiddenSize,
                rowBytes: hiddenSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesIHMatrix = MPSMatrix(
            buffer: gatesIHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesHHMatrix = MPSMatrix(
            buffer: gatesHHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        // MPS matmul: gatesIH = input @ W_ih^T
        // Note: weightIH is [4*H, inputSize], we need input @ W^T
        let matmulIH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: inputSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulIH.encode(
            commandBuffer: commandBuffer,
            leftMatrix: inputMatrix,
            rightMatrix: weightIH,
            resultMatrix: gatesIHMatrix
        )

        // MPS matmul: gatesHH = hPrev @ W_hh^T
        let matmulHH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: hiddenSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulHH.encode(
            commandBuffer: commandBuffer,
            leftMatrix: hPrevMatrix,
            rightMatrix: weightHH,
            resultMatrix: gatesHHMatrix
        )

        // Custom kernel: gates = gatesIH + gatesHH
        let addPipeline = try getPipeline(for: "lstm_add_gates")
        guard let addEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LSTMEngineError.encodingFailed
        }

        var totalGates = UInt32(batchSize * gateSize)
        addEncoder.setComputePipelineState(addPipeline)
        addEncoder.setBuffer(gatesIHBuffer, offset: 0, index: 0)
        addEncoder.setBuffer(gatesHHBuffer, offset: 0, index: 1)
        addEncoder.setBuffer(gatesBuffer, offset: 0, index: 2)
        addEncoder.setBytes(&totalGates, length: MemoryLayout<UInt32>.size, index: 3)

        let addThreads = MTLSize(width: Int(totalGates), height: 1, depth: 1)
        let addThreadsPerGroup = MTLSize(
            width: min(256, addPipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )
        let addGroups = MTLSize(
            width: (Int(totalGates) + addThreadsPerGroup.width - 1) / addThreadsPerGroup.width,
            height: 1,
            depth: 1
        )
        addEncoder.dispatchThreadgroups(addGroups, threadsPerThreadgroup: addThreadsPerGroup)
        addEncoder.endEncoding()

        // Add biases if present
        if let biasIH = biasIH, let biasHH = biasHH {
            guard let biasIHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride),
                  let biasHHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride)
            else {
                throw LSTMEngineError.bufferCreationFailed
            }
            defer {
                bufferPool.release(biasIHBuffer)
                bufferPool.release(biasHHBuffer)
            }

            memcpy(biasIHBuffer.contents(), biasIH, biasIH.count * MemoryLayout<Float>.stride)
            memcpy(biasHHBuffer.contents(), biasHH, biasHH.count * MemoryLayout<Float>.stride)

            let biasPipeline = try getPipeline(for: "lstm_add_biases")
            guard let biasEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }

            var gateSizeVar = UInt32(gateSize)
            var batchSizeVar = UInt32(batchSize)

            biasEncoder.setComputePipelineState(biasPipeline)
            biasEncoder.setBuffer(gatesBuffer, offset: 0, index: 0)
            biasEncoder.setBuffer(biasIHBuffer, offset: 0, index: 1)
            biasEncoder.setBuffer(biasHHBuffer, offset: 0, index: 2)
            biasEncoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
            biasEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 4)

            let biasThreadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
            let biasGroups = MTLSize(
                width: (gateSize + biasThreadsPerGroup.width - 1) / biasThreadsPerGroup.width,
                height: (batchSize + biasThreadsPerGroup.height - 1) / biasThreadsPerGroup.height,
                depth: 1
            )
            biasEncoder.dispatchThreadgroups(biasGroups, threadsPerThreadgroup: biasThreadsPerGroup)
            biasEncoder.endEncoding()
        }

        // Custom kernel: state update
        let updatePipeline = try getPipeline(for: "lstm_state_update")
        guard let updateEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LSTMEngineError.encodingFailed
        }

        var hiddenSizeVar = UInt32(hiddenSize)
        var batchSizeVar = UInt32(batchSize)

        updateEncoder.setComputePipelineState(updatePipeline)
        updateEncoder.setBuffer(gatesBuffer, offset: 0, index: 0)
        updateEncoder.setBuffer(cPrevBuffer, offset: 0, index: 1)
        updateEncoder.setBuffer(hOutBuffer, offset: 0, index: 2)
        updateEncoder.setBuffer(cOutBuffer, offset: 0, index: 3)
        updateEncoder.setBytes(&hiddenSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
        updateEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 5)

        let updateThreadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
        let updateGroups = MTLSize(
            width: (hiddenSize + updateThreadsPerGroup.width - 1) / updateThreadsPerGroup.width,
            height: (batchSize + updateThreadsPerGroup.height - 1) / updateThreadsPerGroup.height,
            depth: 1
        )
        updateEncoder.dispatchThreadgroups(updateGroups, threadsPerThreadgroup: updateThreadsPerGroup)
        updateEncoder.endEncoding()

        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let hOutPtr = hOutBuffer.contents().assumingMemoryBound(to: Float.self)
        let cOutPtr = cOutBuffer.contents().assumingMemoryBound(to: Float.self)

        let hOut = Array(UnsafeBufferPointer(start: hOutPtr, count: batchSize * hiddenSize))
        let cOut = Array(UnsafeBufferPointer(start: cOutPtr, count: batchSize * hiddenSize))

        return (hOut, cOut)
    }

    // MARK: - GRU Cell Forward

    /// Perform single GRU cell forward pass.
    ///
    /// Computes one timestep of GRU:
    /// 1. Gate activations: gates_ih = W_ih @ x + bias_ih, gates_hh = W_hh @ h_prev + bias_hh
    /// 2. State update: h_new = gru_update(gates_ih, gates_hh, h_prev)
    ///
    /// Note: GRU keeps gates_ih and gates_hh separate because the "new" gate computation
    /// requires the reset gate to modulate only the hidden-to-hidden contribution:
    ///   r = sigmoid(gates_ih[0:H] + gates_hh[0:H])
    ///   z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])
    ///   n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])
    ///   h_new = (1-z) * n + z * h_prev
    ///
    /// - Parameters:
    ///   - input: Input at this timestep [batchSize, inputSize].
    ///   - hPrev: Previous hidden state [batchSize, hiddenSize].
    ///   - weightIH: Input-to-hidden weight matrix (MPS).
    ///   - weightHH: Hidden-to-hidden weight matrix (MPS).
    ///   - biasIH: Input-to-hidden bias [3*hiddenSize], or nil.
    ///   - biasHH: Hidden-to-hidden bias [3*hiddenSize], or nil.
    ///   - batchSize: Batch dimension.
    ///   - inputSize: Input feature dimension.
    ///   - hiddenSize: Hidden state dimension.
    /// - Returns: New hidden state.
    public func gruCellForward(
        input: [Float],
        hPrev: [Float],
        weightIH: MPSMatrix,
        weightHH: MPSMatrix,
        biasIH: [Float]?,
        biasHH: [Float]?,
        batchSize: Int,
        inputSize: Int,
        hiddenSize: Int
    ) async throws -> [Float] {
        let gateSize = 3 * hiddenSize

        // Calculate buffer sizes
        let inputBytes = batchSize * inputSize * MemoryLayout<Float>.stride
        let hPrevBytes = batchSize * hiddenSize * MemoryLayout<Float>.stride
        let gatesBytes = batchSize * gateSize * MemoryLayout<Float>.stride

        // Acquire buffers
        guard let inputBuffer = bufferPool.acquire(minBytes: inputBytes),
              let hPrevBuffer = bufferPool.acquire(minBytes: hPrevBytes),
              let gatesIHBuffer = bufferPool.acquire(minBytes: gatesBytes),
              let gatesHHBuffer = bufferPool.acquire(minBytes: gatesBytes),
              let hOutBuffer = bufferPool.acquire(minBytes: hPrevBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(hPrevBuffer)
            bufferPool.release(gatesIHBuffer)
            bufferPool.release(gatesHHBuffer)
            bufferPool.release(hOutBuffer)
        }

        // Copy input data
        memcpy(inputBuffer.contents(), input, input.count * MemoryLayout<Float>.stride)
        memcpy(hPrevBuffer.contents(), hPrev, hPrev.count * MemoryLayout<Float>.stride)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw LSTMEngineError.commandBufferFailed
        }

        // Create MPS matrices for inputs
        let inputMatrix = MPSMatrix(
            buffer: inputBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: inputSize,
                rowBytes: inputSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let hPrevMatrix = MPSMatrix(
            buffer: hPrevBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: hiddenSize,
                rowBytes: hiddenSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesIHMatrix = MPSMatrix(
            buffer: gatesIHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesHHMatrix = MPSMatrix(
            buffer: gatesHHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        // MPS matmul: gatesIH = input @ W_ih^T
        let matmulIH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: inputSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulIH.encode(
            commandBuffer: commandBuffer,
            leftMatrix: inputMatrix,
            rightMatrix: weightIH,
            resultMatrix: gatesIHMatrix
        )

        // MPS matmul: gatesHH = hPrev @ W_hh^T
        let matmulHH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: hiddenSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulHH.encode(
            commandBuffer: commandBuffer,
            leftMatrix: hPrevMatrix,
            rightMatrix: weightHH,
            resultMatrix: gatesHHMatrix
        )

        // Add biases if present (to separate buffers)
        // Acquire bias buffers upfront and use defer for consistent cleanup pattern
        var biasIHBuffer: MTLBuffer?
        var biasHHBuffer: MTLBuffer?

        if biasIH != nil {
            biasIHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride)
            guard biasIHBuffer != nil else {
                throw LSTMEngineError.bufferCreationFailed
            }
        }
        if biasHH != nil {
            biasHHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride)
            guard biasHHBuffer != nil else {
                if let buffer = biasIHBuffer { bufferPool.release(buffer) }
                throw LSTMEngineError.bufferCreationFailed
            }
        }

        defer {
            if let buffer = biasIHBuffer { bufferPool.release(buffer) }
            if let buffer = biasHHBuffer { bufferPool.release(buffer) }
        }

        if let biasIH = biasIH, let buffer = biasIHBuffer {
            memcpy(buffer.contents(), biasIH, biasIH.count * MemoryLayout<Float>.stride)

            let biasPipeline = try getPipeline(for: "gru_add_bias_ih")
            guard let biasEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }

            var gateSizeVar = UInt32(gateSize)
            var batchSizeVar = UInt32(batchSize)

            biasEncoder.setComputePipelineState(biasPipeline)
            biasEncoder.setBuffer(gatesIHBuffer, offset: 0, index: 0)
            biasEncoder.setBuffer(buffer, offset: 0, index: 1)
            biasEncoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            biasEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 3)

            let biasThreadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
            let biasGroups = MTLSize(
                width: (gateSize + biasThreadsPerGroup.width - 1) / biasThreadsPerGroup.width,
                height: (batchSize + biasThreadsPerGroup.height - 1) / biasThreadsPerGroup.height,
                depth: 1
            )
            biasEncoder.dispatchThreadgroups(biasGroups, threadsPerThreadgroup: biasThreadsPerGroup)
            biasEncoder.endEncoding()
        }

        if let biasHH = biasHH, let buffer = biasHHBuffer {
            memcpy(buffer.contents(), biasHH, biasHH.count * MemoryLayout<Float>.stride)

            let biasPipeline = try getPipeline(for: "gru_add_bias_hh")
            guard let biasEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }

            var gateSizeVar = UInt32(gateSize)
            var batchSizeVar = UInt32(batchSize)

            biasEncoder.setComputePipelineState(biasPipeline)
            biasEncoder.setBuffer(gatesHHBuffer, offset: 0, index: 0)
            biasEncoder.setBuffer(buffer, offset: 0, index: 1)
            biasEncoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            biasEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 3)

            let biasThreadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
            let biasGroups = MTLSize(
                width: (gateSize + biasThreadsPerGroup.width - 1) / biasThreadsPerGroup.width,
                height: (batchSize + biasThreadsPerGroup.height - 1) / biasThreadsPerGroup.height,
                depth: 1
            )
            biasEncoder.dispatchThreadgroups(biasGroups, threadsPerThreadgroup: biasThreadsPerGroup)
            biasEncoder.endEncoding()
        }

        // GRU state update kernel
        let updatePipeline = try getPipeline(for: "gru_state_update")
        guard let updateEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LSTMEngineError.encodingFailed
        }

        var hiddenSizeVar = UInt32(hiddenSize)
        var batchSizeVar = UInt32(batchSize)

        updateEncoder.setComputePipelineState(updatePipeline)
        updateEncoder.setBuffer(gatesIHBuffer, offset: 0, index: 0)
        updateEncoder.setBuffer(gatesHHBuffer, offset: 0, index: 1)
        updateEncoder.setBuffer(hPrevBuffer, offset: 0, index: 2)
        updateEncoder.setBuffer(hOutBuffer, offset: 0, index: 3)
        updateEncoder.setBytes(&hiddenSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
        updateEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 5)

        let updateThreadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
        let updateGroups = MTLSize(
            width: (hiddenSize + updateThreadsPerGroup.width - 1) / updateThreadsPerGroup.width,
            height: (batchSize + updateThreadsPerGroup.height - 1) / updateThreadsPerGroup.height,
            depth: 1
        )
        updateEncoder.dispatchThreadgroups(updateGroups, threadsPerThreadgroup: updateThreadsPerGroup)
        updateEncoder.endEncoding()

        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read result
        let hOutPtr = hOutBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: hOutPtr, count: batchSize * hiddenSize))
    }

    // MARK: - Sequence Operations

    /// Reverse a sequence along the time dimension.
    ///
    /// - Parameters:
    ///   - input: Input sequence [batchSize, seqLen, featureSize].
    ///   - batchSize: Batch dimension.
    ///   - seqLen: Sequence length.
    ///   - featureSize: Feature dimension.
    /// - Returns: Reversed sequence.
    public func reverseSequence(
        _ input: [Float],
        batchSize: Int,
        seqLen: Int,
        featureSize: Int
    ) async throws -> [Float] {
        let totalCount = batchSize * seqLen * featureSize
        let bufferBytes = totalCount * MemoryLayout<Float>.stride

        guard let inputBuffer = bufferPool.acquire(minBytes: bufferBytes),
              let outputBuffer = bufferPool.acquire(minBytes: bufferBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(inputBuffer.contents(), input, bufferBytes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw LSTMEngineError.commandBufferFailed
        }

        let pipeline = try getPipeline(for: "reverse_sequence")

        var batchSizeVar = UInt32(batchSize)
        var seqLenVar = UInt32(seqLen)
        var featureSizeVar = UInt32(featureSize)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&featureSizeVar, length: MemoryLayout<UInt32>.size, index: 4)

        let threadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
        let threadGroups = MTLSize(
            width: (featureSize + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (seqLen + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: batchSize
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: totalCount))
    }

    /// Concatenate bidirectional LSTM outputs.
    ///
    /// - Parameters:
    ///   - forward: Forward output [batchSize, seqLen, hiddenSize].
    ///   - backward: Backward output [batchSize, seqLen, hiddenSize].
    ///   - batchSize: Batch dimension.
    ///   - seqLen: Sequence length.
    ///   - hiddenSize: Hidden dimension per direction.
    /// - Returns: Concatenated output [batchSize, seqLen, 2*hiddenSize].
    public func concatenateBidirectional(
        forward: [Float],
        backward: [Float],
        batchSize: Int,
        seqLen: Int,
        hiddenSize: Int
    ) async throws -> [Float] {
        let inputCount = batchSize * seqLen * hiddenSize
        let outputCount = batchSize * seqLen * hiddenSize * 2
        let inputBytes = inputCount * MemoryLayout<Float>.stride
        let outputBytes = outputCount * MemoryLayout<Float>.stride

        guard let forwardBuffer = bufferPool.acquire(minBytes: inputBytes),
              let backwardBuffer = bufferPool.acquire(minBytes: inputBytes),
              let outputBuffer = bufferPool.acquire(minBytes: outputBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(forwardBuffer)
            bufferPool.release(backwardBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(forwardBuffer.contents(), forward, inputBytes)
        memcpy(backwardBuffer.contents(), backward, inputBytes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw LSTMEngineError.commandBufferFailed
        }

        let pipeline = try getPipeline(for: "concat_bidirectional")

        var batchSizeVar = UInt32(batchSize)
        var seqLenVar = UInt32(seqLen)
        var hiddenSizeVar = UInt32(hiddenSize)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(forwardBuffer, offset: 0, index: 0)
        encoder.setBuffer(backwardBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&hiddenSizeVar, length: MemoryLayout<UInt32>.size, index: 5)

        let threadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
        let threadGroups = MTLSize(
            width: (hiddenSize * 2 + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (seqLen + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: batchSize
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: outputCount))
    }

    // MARK: - Optimized GRU Sequence Forward

    /// Process an entire GRU sequence with a single command buffer.
    ///
    /// This is dramatically faster than calling gruCellForward in a loop because:
    /// 1. Single command buffer for the entire sequence (vs one per timestep)
    /// 2. All input-to-hidden matmuls computed upfront in one batch
    /// 3. Only one waitUntilCompleted at the end (vs one per timestep)
    ///
    /// The recurrent computation (h_prev -> gates_hh) must still be sequential,
    /// but the GPU stays busy processing the state updates.
    ///
    /// - Parameters:
    ///   - input: Input sequence [batchSize, seqLen, inputSize] or [seqLen, batchSize, inputSize]
    ///   - initialState: Initial hidden state [batchSize, hiddenSize], or nil for zeros
    ///   - weightIH: Input-to-hidden weight matrix [3*hiddenSize, inputSize]
    ///   - weightHH: Hidden-to-hidden weight matrix [3*hiddenSize, hiddenSize]
    ///   - biasIH: Input-to-hidden bias [3*hiddenSize], or nil
    ///   - biasHH: Hidden-to-hidden bias [3*hiddenSize], or nil
    ///   - batchSize: Batch dimension
    ///   - seqLen: Sequence length
    ///   - inputSize: Input feature dimension
    ///   - hiddenSize: Hidden state dimension
    ///   - batchFirst: If true, input is [batch, seq, input]; if false, [seq, batch, input]
    ///   - reverse: If true, process sequence in reverse order
    /// - Returns: Tuple of (output sequence, final hidden state)
    public func gruSequenceForward(
        input: [Float],
        initialState: [Float]?,
        weightIH: MPSMatrix,
        weightHH: MPSMatrix,
        biasIH: [Float]?,
        biasHH: [Float]?,
        batchSize: Int,
        seqLen: Int,
        inputSize: Int,
        hiddenSize: Int,
        batchFirst: Bool = true,
        reverse: Bool = false
    ) async throws -> (output: [Float], finalState: [Float]) {
        let gateSize = 3 * hiddenSize

        // Calculate buffer sizes
        let inputBytes = batchSize * seqLen * inputSize * MemoryLayout<Float>.stride
        let hStateBytes = batchSize * hiddenSize * MemoryLayout<Float>.stride
        let gatesIHBytes = seqLen * batchSize * gateSize * MemoryLayout<Float>.stride
        let gatesHHBytes = batchSize * gateSize * MemoryLayout<Float>.stride
        let outputBytes = batchSize * seqLen * hiddenSize * MemoryLayout<Float>.stride

        // Acquire all buffers upfront
        guard let inputBuffer = bufferPool.acquire(minBytes: inputBytes),
              let hStateBuffer = bufferPool.acquire(minBytes: hStateBytes),
              let hTempBuffer = bufferPool.acquire(minBytes: hStateBytes),
              let gatesIHBuffer = bufferPool.acquire(minBytes: gatesIHBytes),
              let gatesHHBuffer = bufferPool.acquire(minBytes: gatesHHBytes),
              let outputBuffer = bufferPool.acquire(minBytes: outputBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(hStateBuffer)
            bufferPool.release(hTempBuffer)
            bufferPool.release(gatesIHBuffer)
            bufferPool.release(gatesHHBuffer)
            bufferPool.release(outputBuffer)
        }

        // Copy input and initialize state
        memcpy(inputBuffer.contents(), input, input.count * MemoryLayout<Float>.stride)
        if let state = initialState {
            memcpy(hStateBuffer.contents(), state, state.count * MemoryLayout<Float>.stride)
        } else {
            memset(hStateBuffer.contents(), 0, hStateBytes)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw LSTMEngineError.commandBufferFailed
        }

        // Step 1: Compute ALL input-to-hidden gates at once
        // Reshape input from [batch, seq, input] to [batch*seq, input] for batched matmul
        // Result: gatesIH [batch*seq, gateSize]
        let flatInputMatrix = MPSMatrix(
            buffer: inputBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize * seqLen,
                columns: inputSize,
                rowBytes: inputSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesIHMatrix = MPSMatrix(
            buffer: gatesIHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize * seqLen,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let matmulIH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize * seqLen,
            resultColumns: gateSize,
            interiorColumns: inputSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulIH.encode(commandBuffer: commandBuffer, leftMatrix: flatInputMatrix, rightMatrix: weightIH, resultMatrix: gatesIHMatrix)

        // Add biasIH to all gates at once
        if let biasIH = biasIH {
            guard let biasBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride) else {
                throw LSTMEngineError.bufferCreationFailed
            }
            defer { bufferPool.release(biasBuffer) }
            memcpy(biasBuffer.contents(), biasIH, biasIH.count * MemoryLayout<Float>.stride)

            let biasPipeline = try getPipeline(for: "gru_add_bias_ih")
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }
            var gateSizeVar = UInt32(gateSize)
            var totalBatchVar = UInt32(batchSize * seqLen)
            encoder.setComputePipelineState(biasPipeline)
            encoder.setBuffer(gatesIHBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasBuffer, offset: 0, index: 1)
            encoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&totalBatchVar, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
            let groups = MTLSize(
                width: (gateSize + 31) / 32,
                height: (batchSize * seqLen + 7) / 8,
                depth: 1
            )
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        // Commit the batched matmul before starting the sequential loop
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Step 2: Process timesteps sequentially (recurrence requires this)
        // But we batch all GPU work for each timestep into a single command buffer per timestep
        let timeIndices: [Int] = reverse ? (0..<seqLen).reversed().map { $0 } : Array(0..<seqLen)

        // Setup bias buffer for HH path
        var biasHHBuffer: MTLBuffer?
        if let biasHH = biasHH {
            biasHHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride)
            guard biasHHBuffer != nil else {
                throw LSTMEngineError.bufferCreationFailed
            }
            memcpy(biasHHBuffer!.contents(), biasHH, biasHH.count * MemoryLayout<Float>.stride)
        }
        defer {
            if let b = biasHHBuffer { bufferPool.release(b) }
        }

        // Create matrices for recurrent computation
        let hMatrix = MPSMatrix(
            buffer: hStateBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: hiddenSize,
                rowBytes: hiddenSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesHHMatrix = MPSMatrix(
            buffer: gatesHHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let matmulHH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: hiddenSize,
            alpha: 1.0,
            beta: 0.0
        )

        // Process each timestep
        for (outT, inputT) in timeIndices.enumerated() {
            guard let stepBuffer = commandQueue.makeCommandBuffer() else {
                throw LSTMEngineError.commandBufferFailed
            }

            // Compute gates_hh = h_state @ W_hh^T
            matmulHH.encode(commandBuffer: stepBuffer, leftMatrix: hMatrix, rightMatrix: weightHH, resultMatrix: gatesHHMatrix)

            // Add biasHH if present
            if let biasBuffer = biasHHBuffer {
                let biasPipeline = try getPipeline(for: "gru_add_bias_hh")
                guard let encoder = stepBuffer.makeComputeCommandEncoder() else {
                    throw LSTMEngineError.encodingFailed
                }
                var gateSizeVar = UInt32(gateSize)
                var batchSizeVar = UInt32(batchSize)
                encoder.setComputePipelineState(biasPipeline)
                encoder.setBuffer(gatesHHBuffer, offset: 0, index: 0)
                encoder.setBuffer(biasBuffer, offset: 0, index: 1)
                encoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 3)

                let groups = MTLSize(width: (gateSize + 31) / 32, height: (batchSize + 7) / 8, depth: 1)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
                encoder.endEncoding()
            }

            // GRU state update
            let updatePipeline = try getPipeline(for: "gru_state_update")
            guard let updateEncoder = stepBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }

            // Offset into pre-computed gatesIH for this timestep
            let gatesIHOffset = inputT * batchSize * gateSize * MemoryLayout<Float>.stride

            var hiddenSizeVar = UInt32(hiddenSize)
            var batchSizeVar = UInt32(batchSize)

            updateEncoder.setComputePipelineState(updatePipeline)
            updateEncoder.setBuffer(gatesIHBuffer, offset: gatesIHOffset, index: 0)
            updateEncoder.setBuffer(gatesHHBuffer, offset: 0, index: 1)
            updateEncoder.setBuffer(hStateBuffer, offset: 0, index: 2)
            updateEncoder.setBuffer(hTempBuffer, offset: 0, index: 3)
            updateEncoder.setBytes(&hiddenSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
            updateEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 5)

            let updateGroups = MTLSize(
                width: (hiddenSize + 31) / 32,
                height: (batchSize + 7) / 8,
                depth: 1
            )
            updateEncoder.dispatchThreadgroups(updateGroups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            updateEncoder.endEncoding()

            // Copy new state from temp to state buffer and to output
            let copyPipeline = try getPipeline(for: "buffer_copy")

            // Copy hTemp -> hState
            guard let copyEncoder1 = stepBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }
            var copyCount = UInt32(batchSize * hiddenSize)
            copyEncoder1.setComputePipelineState(copyPipeline)
            copyEncoder1.setBuffer(hTempBuffer, offset: 0, index: 0)
            copyEncoder1.setBuffer(hStateBuffer, offset: 0, index: 1)
            copyEncoder1.setBytes(&copyCount, length: MemoryLayout<UInt32>.size, index: 2)
            let copyGroups = MTLSize(width: (Int(copyCount) + 255) / 256, height: 1, depth: 1)
            copyEncoder1.dispatchThreadgroups(copyGroups, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            copyEncoder1.endEncoding()

            // Store to output at correct timestep position
            // For reverse, we store in forward order (outT), not reverse order
            let storePipeline = try getPipeline(for: "store_timestep")
            guard let storeEncoder = stepBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }

            let outputT = reverse ? (seqLen - 1 - outT) : outT
            var seqLenVar = UInt32(seqLen)
            var featureSizeVar = UInt32(hiddenSize)
            var timestepVar = UInt32(outputT)

            storeEncoder.setComputePipelineState(storePipeline)
            storeEncoder.setBuffer(hTempBuffer, offset: 0, index: 0)
            storeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            storeEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            storeEncoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.size, index: 3)
            storeEncoder.setBytes(&featureSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
            storeEncoder.setBytes(&timestepVar, length: MemoryLayout<UInt32>.size, index: 5)

            let storeGroups = MTLSize(width: (hiddenSize + 31) / 32, height: (batchSize + 7) / 8, depth: 1)
            storeEncoder.dispatchThreadgroups(storeGroups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            storeEncoder.endEncoding()

            stepBuffer.commit()
            stepBuffer.waitUntilCompleted()
        }

        // Read results
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: batchSize * seqLen * hiddenSize))

        let statePtr = hStateBuffer.contents().assumingMemoryBound(to: Float.self)
        let finalState = Array(UnsafeBufferPointer(start: statePtr, count: batchSize * hiddenSize))

        return (output, finalState)
    }

    // MARK: - Batched GRU Sequence Forward (Optimized)

    /// Process an entire GRU sequence with reduced command buffer synchronization.
    ///
    /// This is an optimized version that batches multiple timesteps into each command buffer,
    /// dramatically reducing GPU sync overhead. For a 256-step sequence with timestepsPerBatch=16,
    /// this reduces sync points from 256 to 16.
    ///
    /// - Parameters:
    ///   - input: Input sequence [batchSize, seqLen, inputSize] or [seqLen, batchSize, inputSize]
    ///   - initialState: Initial hidden state [batchSize, hiddenSize], or nil for zeros
    ///   - weightIH: Input-to-hidden weight matrix [3*hiddenSize, inputSize]
    ///   - weightHH: Hidden-to-hidden weight matrix [3*hiddenSize, hiddenSize]
    ///   - biasIH: Input-to-hidden bias [3*hiddenSize], or nil
    ///   - biasHH: Hidden-to-hidden bias [3*hiddenSize], or nil
    ///   - batchSize: Batch dimension
    ///   - seqLen: Sequence length
    ///   - inputSize: Input feature dimension
    ///   - hiddenSize: Hidden state dimension
    ///   - batchFirst: If true, input is [batch, seq, input]; if false, [seq, batch, input]
    ///   - reverse: If true, process sequence in reverse order
    ///   - timestepsPerBatch: Number of timesteps to batch into each command buffer (default: 16)
    /// - Returns: Tuple of (output sequence, final hidden state)
    public func gruSequenceForwardBatched(
        input: [Float],
        initialState: [Float]?,
        weightIH: MPSMatrix,
        weightHH: MPSMatrix,
        biasIH: [Float]?,
        biasHH: [Float]?,
        batchSize: Int,
        seqLen: Int,
        inputSize: Int,
        hiddenSize: Int,
        batchFirst: Bool = true,
        reverse: Bool = false,
        timestepsPerBatch: Int = 16
    ) async throws -> (output: [Float], finalState: [Float]) {
        let gateSize = 3 * hiddenSize

        // Calculate buffer sizes
        let inputBytes = batchSize * seqLen * inputSize * MemoryLayout<Float>.stride
        let hStateBytes = batchSize * hiddenSize * MemoryLayout<Float>.stride
        let gatesIHBytes = seqLen * batchSize * gateSize * MemoryLayout<Float>.stride
        let gatesHHBytes = batchSize * gateSize * MemoryLayout<Float>.stride
        let outputBytes = batchSize * seqLen * hiddenSize * MemoryLayout<Float>.stride

        // Acquire all buffers upfront
        guard let inputBuffer = bufferPool.acquire(minBytes: inputBytes),
              let hStateBuffer = bufferPool.acquire(minBytes: hStateBytes),
              let hTempBuffer = bufferPool.acquire(minBytes: hStateBytes),
              let gatesIHBuffer = bufferPool.acquire(minBytes: gatesIHBytes),
              let gatesHHBuffer = bufferPool.acquire(minBytes: gatesHHBytes),
              let outputBuffer = bufferPool.acquire(minBytes: outputBytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(hStateBuffer)
            bufferPool.release(hTempBuffer)
            bufferPool.release(gatesIHBuffer)
            bufferPool.release(gatesHHBuffer)
            bufferPool.release(outputBuffer)
        }

        // Copy input and initialize state
        memcpy(inputBuffer.contents(), input, input.count * MemoryLayout<Float>.stride)
        if let state = initialState {
            memcpy(hStateBuffer.contents(), state, state.count * MemoryLayout<Float>.stride)
        } else {
            memset(hStateBuffer.contents(), 0, hStateBytes)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw LSTMEngineError.commandBufferFailed
        }

        // Step 1: Compute ALL input-to-hidden gates at once (same as before)
        let flatInputMatrix = MPSMatrix(
            buffer: inputBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize * seqLen,
                columns: inputSize,
                rowBytes: inputSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesIHMatrix = MPSMatrix(
            buffer: gatesIHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize * seqLen,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let matmulIH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize * seqLen,
            resultColumns: gateSize,
            interiorColumns: inputSize,
            alpha: 1.0,
            beta: 0.0
        )
        matmulIH.encode(commandBuffer: commandBuffer, leftMatrix: flatInputMatrix, rightMatrix: weightIH, resultMatrix: gatesIHMatrix)

        // Add biasIH to all gates at once
        if let biasIH = biasIH {
            guard let biasBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride) else {
                throw LSTMEngineError.bufferCreationFailed
            }
            defer { bufferPool.release(biasBuffer) }
            memcpy(biasBuffer.contents(), biasIH, biasIH.count * MemoryLayout<Float>.stride)

            let biasPipeline = try getPipeline(for: "gru_add_bias_ih")
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LSTMEngineError.encodingFailed
            }
            var gateSizeVar = UInt32(gateSize)
            var totalBatchVar = UInt32(batchSize * seqLen)
            encoder.setComputePipelineState(biasPipeline)
            encoder.setBuffer(gatesIHBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasBuffer, offset: 0, index: 1)
            encoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&totalBatchVar, length: MemoryLayout<UInt32>.size, index: 3)

            let threadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
            let groups = MTLSize(
                width: (gateSize + 31) / 32,
                height: (batchSize * seqLen + 7) / 8,
                depth: 1
            )
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        // Commit the batched matmul before starting the sequential loop
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Step 2: Process timesteps in batches
        let timeIndices: [Int] = reverse ? (0..<seqLen).reversed().map { $0 } : Array(0..<seqLen)

        // Setup bias buffer for HH path
        var biasHHBuffer: MTLBuffer?
        if let biasHH = biasHH {
            biasHHBuffer = bufferPool.acquire(minBytes: gateSize * MemoryLayout<Float>.stride)
            guard biasHHBuffer != nil else {
                throw LSTMEngineError.bufferCreationFailed
            }
            memcpy(biasHHBuffer!.contents(), biasHH, biasHH.count * MemoryLayout<Float>.stride)
        }
        defer {
            if let b = biasHHBuffer { bufferPool.release(b) }
        }

        // Create matrices for recurrent computation
        let hMatrix = MPSMatrix(
            buffer: hStateBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: hiddenSize,
                rowBytes: hiddenSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let gatesHHMatrix = MPSMatrix(
            buffer: gatesHHBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: batchSize,
                columns: gateSize,
                rowBytes: gateSize * MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )

        let matmulHH = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batchSize,
            resultColumns: gateSize,
            interiorColumns: hiddenSize,
            alpha: 1.0,
            beta: 0.0
        )

        // Pre-fetch pipelines
        let biasPipeline = biasHHBuffer != nil ? try getPipeline(for: "gru_add_bias_hh") : nil
        let updatePipeline = try getPipeline(for: "gru_state_update")
        let copyPipeline = try getPipeline(for: "buffer_copy")
        let storePipeline = try getPipeline(for: "store_timestep")

        // Process timesteps in batches - key optimization
        var timestepIndex = 0
        while timestepIndex < seqLen {
            let batchEnd = min(timestepIndex + timestepsPerBatch, seqLen)

            guard let batchBuffer = commandQueue.makeCommandBuffer() else {
                throw LSTMEngineError.commandBufferFailed
            }

            // Encode multiple timesteps into this command buffer
            for t in timestepIndex..<batchEnd {
                let inputT = timeIndices[t]
                let outT = t

                // Compute gates_hh = h_state @ W_hh^T
                matmulHH.encode(commandBuffer: batchBuffer, leftMatrix: hMatrix, rightMatrix: weightHH, resultMatrix: gatesHHMatrix)

                // Add biasHH if present
                if let biasBuffer = biasHHBuffer, let pipeline = biasPipeline {
                    guard let encoder = batchBuffer.makeComputeCommandEncoder() else {
                        throw LSTMEngineError.encodingFailed
                    }
                    var gateSizeVar = UInt32(gateSize)
                    var batchSizeVar = UInt32(batchSize)
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(gatesHHBuffer, offset: 0, index: 0)
                    encoder.setBuffer(biasBuffer, offset: 0, index: 1)
                    encoder.setBytes(&gateSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                    encoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 3)

                    let groups = MTLSize(width: (gateSize + 31) / 32, height: (batchSize + 7) / 8, depth: 1)
                    encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
                    encoder.endEncoding()
                }

                // GRU state update
                guard let updateEncoder = batchBuffer.makeComputeCommandEncoder() else {
                    throw LSTMEngineError.encodingFailed
                }

                let gatesIHOffset = inputT * batchSize * gateSize * MemoryLayout<Float>.stride
                var hiddenSizeVar = UInt32(hiddenSize)
                var batchSizeVar = UInt32(batchSize)

                updateEncoder.setComputePipelineState(updatePipeline)
                updateEncoder.setBuffer(gatesIHBuffer, offset: gatesIHOffset, index: 0)
                updateEncoder.setBuffer(gatesHHBuffer, offset: 0, index: 1)
                updateEncoder.setBuffer(hStateBuffer, offset: 0, index: 2)
                updateEncoder.setBuffer(hTempBuffer, offset: 0, index: 3)
                updateEncoder.setBytes(&hiddenSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
                updateEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 5)

                let updateGroups = MTLSize(
                    width: (hiddenSize + 31) / 32,
                    height: (batchSize + 7) / 8,
                    depth: 1
                )
                updateEncoder.dispatchThreadgroups(updateGroups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
                updateEncoder.endEncoding()

                // Copy hTemp -> hState
                guard let copyEncoder1 = batchBuffer.makeComputeCommandEncoder() else {
                    throw LSTMEngineError.encodingFailed
                }
                var copyCount = UInt32(batchSize * hiddenSize)
                copyEncoder1.setComputePipelineState(copyPipeline)
                copyEncoder1.setBuffer(hTempBuffer, offset: 0, index: 0)
                copyEncoder1.setBuffer(hStateBuffer, offset: 0, index: 1)
                copyEncoder1.setBytes(&copyCount, length: MemoryLayout<UInt32>.size, index: 2)
                let copyGroups = MTLSize(width: (Int(copyCount) + 255) / 256, height: 1, depth: 1)
                copyEncoder1.dispatchThreadgroups(copyGroups, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                copyEncoder1.endEncoding()

                // Store to output at correct timestep position
                guard let storeEncoder = batchBuffer.makeComputeCommandEncoder() else {
                    throw LSTMEngineError.encodingFailed
                }

                let outputT = reverse ? (seqLen - 1 - outT) : outT
                var seqLenVar = UInt32(seqLen)
                var featureSizeVar = UInt32(hiddenSize)
                var timestepVar = UInt32(outputT)

                storeEncoder.setComputePipelineState(storePipeline)
                storeEncoder.setBuffer(hTempBuffer, offset: 0, index: 0)
                storeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
                storeEncoder.setBytes(&batchSizeVar, length: MemoryLayout<UInt32>.size, index: 2)
                storeEncoder.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.size, index: 3)
                storeEncoder.setBytes(&featureSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
                storeEncoder.setBytes(&timestepVar, length: MemoryLayout<UInt32>.size, index: 5)

                let storeGroups = MTLSize(width: (hiddenSize + 31) / 32, height: (batchSize + 7) / 8, depth: 1)
                storeEncoder.dispatchThreadgroups(storeGroups, threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
                storeEncoder.endEncoding()
            }

            // Only wait once per batch of timesteps
            batchBuffer.commit()
            batchBuffer.waitUntilCompleted()

            timestepIndex = batchEnd
        }

        // Read results
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: batchSize * seqLen * hiddenSize))

        let statePtr = hStateBuffer.contents().assumingMemoryBound(to: Float.self)
        let finalState = Array(UnsafeBufferPointer(start: statePtr, count: batchSize * hiddenSize))

        return (output, finalState)
    }

    // MARK: - 4D Tensor Transpose

    /// Transpose dimensions 1 and 2 of a 4D tensor on GPU.
    /// [batch, dim1, dim2, features] -> [batch, dim2, dim1, features]
    ///
    /// - Parameters:
    ///   - input: Input tensor flattened
    ///   - batch: Batch dimension
    ///   - dim1: First spatial dimension
    ///   - dim2: Second spatial dimension
    ///   - features: Feature dimension
    /// - Returns: Transposed tensor
    public func transpose4D(
        _ input: [Float],
        batch: Int,
        dim1: Int,
        dim2: Int,
        features: Int
    ) async throws -> [Float] {
        let count = batch * dim1 * dim2 * features
        let bytes = count * MemoryLayout<Float>.stride

        guard let inputBuffer = bufferPool.acquire(minBytes: bytes),
              let outputBuffer = bufferPool.acquire(minBytes: bytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(inputBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(inputBuffer.contents(), input, bytes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw LSTMEngineError.commandBufferFailed
        }

        let pipeline = try getPipeline(for: "transpose_4d")

        var batchVar = UInt32(batch)
        var dim1Var = UInt32(dim1)
        var dim2Var = UInt32(dim2)
        var featuresVar = UInt32(features)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&batchVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&dim1Var, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dim2Var, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&featuresVar, length: MemoryLayout<UInt32>.size, index: 5)

        // Thread layout: [features, dim2, batch*dim1]
        let threadsPerGroup = MTLSize(width: 32, height: 8, depth: 1)
        let threadGroups = MTLSize(
            width: (features + 31) / 32,
            height: (dim2 + 7) / 8,
            depth: batch * dim1
        )

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: count))
    }

    // MARK: - Complex Multiplication

    /// GPU-accelerated complex multiplication.
    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    ///
    /// - Parameters:
    ///   - a: First complex array (interleaved real/imag)
    ///   - b: Second complex array (interleaved real/imag)
    /// - Returns: Result complex array
    public func complexMultiply(_ a: [Float], _ b: [Float]) async throws -> [Float] {
        precondition(a.count == b.count, "Array sizes must match")
        precondition(a.count % 2 == 0, "Array must have even length for complex pairs")

        let bytes = a.count * MemoryLayout<Float>.stride
        let complexCount = a.count / 2

        guard let aBuffer = bufferPool.acquire(minBytes: bytes),
              let bBuffer = bufferPool.acquire(minBytes: bytes),
              let outputBuffer = bufferPool.acquire(minBytes: bytes)
        else {
            throw LSTMEngineError.bufferCreationFailed
        }

        defer {
            bufferPool.release(aBuffer)
            bufferPool.release(bBuffer)
            bufferPool.release(outputBuffer)
        }

        memcpy(aBuffer.contents(), a, bytes)
        memcpy(bBuffer.contents(), b, bytes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw LSTMEngineError.commandBufferFailed
        }

        let pipeline = try getPipeline(for: "complex_multiply")

        var countVar = UInt32(complexCount)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (complexCount + 255) / 256, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: a.count))
    }
}
