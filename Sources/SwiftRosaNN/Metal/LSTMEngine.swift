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
}
