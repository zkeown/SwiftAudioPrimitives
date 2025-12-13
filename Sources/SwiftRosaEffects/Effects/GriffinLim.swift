import SwiftRosaCore
import Accelerate
import Foundation

/// Reusable workspace for Griffin-Lim iterations to minimize allocations.
///
/// This class holds pre-allocated buffers that are reused across iterations,
/// significantly reducing memory pressure during phase reconstruction.
final class GriffinLimWorkspace: @unchecked Sendable {
    /// Flat buffers for current complex spectrogram (real/imag).
    var currentReal: [Float]
    var currentImag: [Float]

    /// Flat buffers for previous iteration (for momentum).
    var prevReal: [Float]
    var prevImag: [Float]

    /// Flat buffer for phase.
    var phase: [Float]

    /// Flat buffer for best phase seen.
    var bestPhase: [Float]

    /// Flat buffers for momentum computation.
    var anglesReal: [Float]
    var anglesImag: [Float]

    /// Dimensions.
    let rows: Int
    let cols: Int

    /// Total element count.
    var count: Int { rows * cols }

    /// Create workspace for given dimensions.
    init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        let count = rows * cols

        self.currentReal = [Float](repeating: 0, count: count)
        self.currentImag = [Float](repeating: 0, count: count)
        self.prevReal = [Float](repeating: 0, count: count)
        self.prevImag = [Float](repeating: 0, count: count)
        self.phase = [Float](repeating: 0, count: count)
        self.bestPhase = [Float](repeating: 0, count: count)
        self.anglesReal = [Float](repeating: 0, count: count)
        self.anglesImag = [Float](repeating: 0, count: count)
    }

    /// Initialize phase with random values in [-π, π].
    func initializeRandomPhase() {
        for i in 0..<count {
            phase[i] = Float.random(in: -Float.pi...Float.pi)
        }
    }

    /// Initialize phase with zeros.
    func initializeZeroPhase() {
        _ = phase.withUnsafeMutableBytes { memset($0.baseAddress!, 0, $0.count) }
    }

    /// Copy current phase to best phase.
    func saveBestPhase() {
        _ = bestPhase.withUnsafeMutableBufferPointer { destPtr in
            phase.withUnsafeBufferPointer { srcPtr in
                memcpy(destPtr.baseAddress!, srcPtr.baseAddress!, count * MemoryLayout<Float>.size)
            }
        }
    }

    /// Swap current and previous buffers (O(1) pointer swap).
    func swapCurrentAndPrev() {
        swap(&currentReal, &prevReal)
        swap(&currentImag, &prevImag)
    }
}

/// Phase reconstruction from magnitude spectrogram using Griffin-Lim algorithm.
///
/// The Griffin-Lim algorithm iteratively estimates the phase of a signal
/// given only its magnitude spectrogram. This is useful when:
/// - A model outputs magnitude-only spectrograms
/// - You want to modify magnitude and reconstruct audio
/// - Converting magnitude-based audio features back to waveform
///
/// This implementation uses the Fast Griffin-Lim algorithm with momentum
/// for faster convergence.
///
/// ## Example Usage
///
/// ```swift
/// let griffinLim = GriffinLim(config: stftConfig, nIter: 32)
///
/// // Reconstruct from magnitude spectrogram
/// let audio = await griffinLim.reconstruct(magnitude, length: originalLength)
///
/// // With convergence monitoring
/// let audio = await griffinLim.reconstruct(magnitude, length: nil) { iter, error in
///     print("Iteration \(iter): error = \(error)")
/// }
/// ```
///
/// ## References
///
/// - Griffin, D. and Lim, J. (1984). "Signal estimation from modified short-time Fourier transform"
/// - Perraudin, N. et al. (2013). "A fast Griffin-Lim algorithm"
public struct GriffinLim: Sendable {
    /// STFT configuration.
    public let config: STFTConfig

    /// Number of iterations.
    public let nIter: Int

    /// Momentum for accelerated convergence (0 to 1).
    public let momentum: Float

    /// Whether to initialize phase randomly (vs zeros).
    public let randomPhaseInit: Bool

    /// STFT processor.
    private let stft: STFT

    /// ISTFT processor.
    private let istft: ISTFT

    /// Create a Griffin-Lim phase reconstructor.
    ///
    /// - Parameters:
    ///   - config: STFT configuration.
    ///   - nIter: Number of iterations (default: 32).
    ///   - momentum: Acceleration parameter for Fast Griffin-Lim (default: 0.99).
    ///               Higher values speed up convergence. Set to 0 for classic Griffin-Lim.
    ///   - randomPhaseInit: Use random initial phase (default: true).
    public init(
        config: STFTConfig = STFTConfig(),
        nIter: Int = 32,
        momentum: Float = 0.99,
        randomPhaseInit: Bool = true
    ) {
        self.config = config
        self.nIter = max(1, nIter)
        self.momentum = max(0, min(1, momentum))
        self.randomPhaseInit = randomPhaseInit
        self.stft = STFT(config: config)
        self.istft = ISTFT(config: config)
    }

    // MARK: - Reconstruction

    /// Reconstruct audio from magnitude spectrogram.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length. If nil, estimated from spectrogram.
    /// - Returns: Reconstructed audio signal.
    public func reconstruct(_ magnitude: [[Float]], length: Int? = nil) async -> [Float] {
        await reconstruct(magnitude, length: length, onIteration: nil)
    }

    /// Reconstruct audio with iteration callback for monitoring convergence.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length.
    ///   - onIteration: Callback receiving (iteration, reconstructionError).
    /// - Returns: Reconstructed audio signal.
    public func reconstruct(
        _ magnitude: [[Float]],
        length: Int?,
        onIteration: (@Sendable (Int, Float) -> Void)?
    ) async -> [Float] {
        guard !magnitude.isEmpty && !magnitude[0].isEmpty else { return [] }

        let nFreqs = magnitude.count
        let nFrames = magnitude[0].count

        // Estimate output length if not provided
        let outputLength = length ?? estimateOutputLength(nFrames: nFrames)

        // Flatten magnitude for efficient vectorized operations
        let flatMagnitude = flattenMatrix(magnitude)

        // Create workspace with all preallocated buffers
        let workspace = GriffinLimWorkspace(rows: nFreqs, cols: nFrames)

        // Initialize phase
        if randomPhaseInit {
            workspace.initializeRandomPhase()
        } else {
            workspace.initializeZeroPhase()
        }

        // Pre-compute momentum scale factor (librosa formula: momentum / (1 + momentum))
        let momScale = momentum / (1.0 + momentum)

        // Track best error
        var bestError: Float = Float.infinity
        var hasPrevious = false

        // Main iteration loop
        // Following librosa's algorithm exactly:
        // angles[:] = rebuilt
        // if tprev is not None: angles -= momentum_scale * tprev
        // angles /= np.abs(angles) + eps
        // angles *= S
        // rebuilt, tprev = tprev, rebuilt  <-- KEY: tprev gets RAW rebuilt, not momentum-adjusted!
        for iter in 0..<nIter {
            // 1. Create complex from current magnitude + phase estimate
            fromPolarInPlace(
                magnitude: flatMagnitude,
                phase: workspace.phase,
                realOut: &workspace.currentReal,
                imagOut: &workspace.currentImag
            )

            // 2. Convert to ComplexMatrix for ISTFT
            let complex = ContiguousComplexMatrix(
                real: workspace.currentReal,
                imag: workspace.currentImag,
                rows: nFreqs,
                cols: nFrames
            ).toComplexMatrix()

            // 3. Inverse STFT to get audio estimate
            let audio = await istft.transform(complex, length: outputLength)

            // 4. Forward STFT to get rebuilt spectrogram
            let reconstructed = await stft.transform(audio)

            // Copy rebuilt to current buffers
            copyComplexMatrixToFlat(
                reconstructed,
                realOut: &workspace.currentReal,
                imagOut: &workspace.currentImag
            )

            // 5. Compute error BEFORE momentum is applied (from raw rebuilt)
            let currentError = computeReconstructionErrorFlat(
                targetMagnitude: flatMagnitude,
                estimatedReal: workspace.currentReal,
                estimatedImag: workspace.currentImag
            )

            // 6. Copy current (RAW rebuilt) to angles buffers for momentum
            // angles[:] = rebuilt
            workspace.anglesReal.withUnsafeMutableBufferPointer { destReal in
                workspace.currentReal.withUnsafeBufferPointer { srcReal in
                    memcpy(destReal.baseAddress!, srcReal.baseAddress!, nFreqs * nFrames * MemoryLayout<Float>.size)
                }
            }
            workspace.anglesImag.withUnsafeMutableBufferPointer { destImag in
                workspace.currentImag.withUnsafeBufferPointer { srcImag in
                    memcpy(destImag.baseAddress!, srcImag.baseAddress!, nFreqs * nFrames * MemoryLayout<Float>.size)
                }
            }

            // 7. Apply Fast Griffin-Lim momentum to angles
            // if tprev is not None: angles -= momentum_scale * tprev
            if momentum > 0 && hasPrevious {
                applyMomentumInPlace(
                    currentReal: &workspace.anglesReal,
                    currentImag: &workspace.anglesImag,
                    prevReal: workspace.prevReal,  // prevReal/prevImag contain RAW rebuilt from prev iter
                    prevImag: workspace.prevImag,
                    momScale: momScale
                )
            }

            // 8. Swap: rebuilt, tprev = tprev, rebuilt
            // Store RAW rebuilt (current) as prev for next iteration
            workspace.swapCurrentAndPrev()
            hasPrevious = true

            // 9. Extract phase from momentum-adjusted angles (not from prev!)
            // This is the normalization step: angles /= |angles| + eps; angles *= S
            // The phase is: atan2(angles.imag, angles.real)
            extractPhaseInPlace(
                real: workspace.anglesReal,
                imag: workspace.anglesImag,
                phaseOut: &workspace.phase
            )

            // Track best phase
            if currentError < bestError {
                bestError = currentError
                workspace.saveBestPhase()
            }

            // Report progress
            if let callback = onIteration {
                callback(iter, currentError)
            }
        }

        // Final reconstruction with best phase seen
        fromPolarInPlace(
            magnitude: flatMagnitude,
            phase: workspace.bestPhase,
            realOut: &workspace.currentReal,
            imagOut: &workspace.currentImag
        )

        let finalComplex = ContiguousComplexMatrix(
            real: workspace.currentReal,
            imag: workspace.currentImag,
            rows: nFreqs,
            cols: nFrames
        ).toComplexMatrix()

        return await istft.transform(finalComplex, length: outputLength)
    }

    // MARK: - Private Implementation (Optimized In-Place Operations)

    /// Flatten a 2D matrix to 1D array (row-major order).
    private func flattenMatrix(_ matrix: [[Float]]) -> [Float] {
        matrix.flatMap { $0 }
    }

    /// Convert polar (magnitude, phase) to rectangular (real, imag) in-place using vDSP.
    private func fromPolarInPlace(
        magnitude: [Float],
        phase: [Float],
        realOut: inout [Float],
        imagOut: inout [Float]
    ) {
        let count = magnitude.count
        precondition(phase.count == count)
        precondition(realOut.count == count)
        precondition(imagOut.count == count)

        // Compute cos(phase) and sin(phase) using vectorized operations
        var cosVals = [Float](repeating: 0, count: count)
        var sinVals = [Float](repeating: 0, count: count)
        var n = Int32(count)

        phase.withUnsafeBufferPointer { phasePtr in
            vvcosf(&cosVals, phasePtr.baseAddress!, &n)
            vvsinf(&sinVals, phasePtr.baseAddress!, &n)
        }

        // real = magnitude * cos(phase)
        // imag = magnitude * sin(phase)
        magnitude.withUnsafeBufferPointer { magPtr in
            vDSP_vmul(magPtr.baseAddress!, 1, cosVals, 1, &realOut, 1, vDSP_Length(count))
            vDSP_vmul(magPtr.baseAddress!, 1, sinVals, 1, &imagOut, 1, vDSP_Length(count))
        }
    }

    /// Extract phase from complex numbers in-place using vDSP.
    private func extractPhaseInPlace(
        real: [Float],
        imag: [Float],
        phaseOut: inout [Float]
    ) {
        let count = real.count
        precondition(imag.count == count)
        precondition(phaseOut.count == count)

        real.withUnsafeBufferPointer { realPtr in
            imag.withUnsafeBufferPointer { imagPtr in
                phaseOut.withUnsafeMutableBufferPointer { outPtr in
                    var n = Int32(count)
                    vvatan2f(outPtr.baseAddress!, imagPtr.baseAddress!, realPtr.baseAddress!, &n)
                }
            }
        }
    }

    /// Apply momentum in-place: current = current - momScale * prev
    private func applyMomentumInPlace(
        currentReal: inout [Float],
        currentImag: inout [Float],
        prevReal: [Float],
        prevImag: [Float],
        momScale: Float
    ) {
        let count = currentReal.count
        var negMomScale = -momScale

        // currentReal += (-momScale) * prevReal
        // currentImag += (-momScale) * prevImag
        prevReal.withUnsafeBufferPointer { prevRealPtr in
            currentReal.withUnsafeMutableBufferPointer { currRealPtr in
                vDSP_vsma(
                    prevRealPtr.baseAddress!, 1,
                    &negMomScale,
                    currRealPtr.baseAddress!, 1,
                    currRealPtr.baseAddress!, 1,
                    vDSP_Length(count)
                )
            }
        }

        prevImag.withUnsafeBufferPointer { prevImagPtr in
            currentImag.withUnsafeMutableBufferPointer { currImagPtr in
                vDSP_vsma(
                    prevImagPtr.baseAddress!, 1,
                    &negMomScale,
                    currImagPtr.baseAddress!, 1,
                    currImagPtr.baseAddress!, 1,
                    vDSP_Length(count)
                )
            }
        }
    }

    /// Copy ComplexMatrix data to flat arrays.
    /// Note: The workspace dimensions must match the matrix dimensions.
    private func copyComplexMatrixToFlat(
        _ matrix: ComplexMatrix,
        realOut: inout [Float],
        imagOut: inout [Float]
    ) {
        let outputCount = realOut.count
        var idx = 0
        for row in 0..<matrix.rows {
            for col in 0..<matrix.cols {
                if idx < outputCount {
                    realOut[idx] = matrix.real[row][col]
                    imagOut[idx] = matrix.imag[row][col]
                    idx += 1
                }
            }
        }
        // Zero any remaining elements if matrix is smaller
        while idx < outputCount {
            realOut[idx] = 0
            imagOut[idx] = 0
            idx += 1
        }
    }

    /// Compute reconstruction error using flat arrays (optimized with vDSP).
    private func computeReconstructionErrorFlat(
        targetMagnitude: [Float],
        estimatedReal: [Float],
        estimatedImag: [Float]
    ) -> Float {
        let count = targetMagnitude.count

        // Compute estimated magnitude using vDSP_vdist
        var estimatedMag = [Float](repeating: 0, count: count)
        estimatedReal.withUnsafeBufferPointer { realPtr in
            estimatedImag.withUnsafeBufferPointer { imagPtr in
                estimatedMag.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_vdist(
                        realPtr.baseAddress!, 1,
                        imagPtr.baseAddress!, 1,
                        outPtr.baseAddress!, 1,
                        vDSP_Length(count)
                    )
                }
            }
        }

        // Compute difference: diff = target - estimated
        var diff = [Float](repeating: 0, count: count)
        targetMagnitude.withUnsafeBufferPointer { targetPtr in
            estimatedMag.withUnsafeBufferPointer { estPtr in
                vDSP_vsub(estPtr.baseAddress!, 1, targetPtr.baseAddress!, 1, &diff, 1, vDSP_Length(count))
            }
        }

        // Compute sum of squared differences
        var sumSquaredError: Float = 0
        vDSP_svesq(diff, 1, &sumSquaredError, vDSP_Length(count))

        // Compute sum of squared target
        var sumTargetSquared: Float = 0
        vDSP_svesq(targetMagnitude, 1, &sumTargetSquared, vDSP_Length(count))

        // Return normalized error
        if sumTargetSquared > 0 {
            return sqrt(sumSquaredError / sumTargetSquared)
        }
        return 0
    }

    /// Estimate output signal length from number of frames.
    private func estimateOutputLength(nFrames: Int) -> Int {
        // With centering: length = (nFrames - 1) * hopLength
        // Without centering: length = nFFT + (nFrames - 1) * hopLength
        if config.center {
            return (nFrames - 1) * config.hopLength
        } else {
            return config.nFFT + (nFrames - 1) * config.hopLength
        }
    }
}

// MARK: - Convenience Extensions

extension GriffinLim {
    /// Reconstruct multiple magnitude spectrograms (e.g., for batch processing).
    ///
    /// - Parameters:
    ///   - magnitudes: Array of magnitude spectrograms.
    ///   - lengths: Corresponding output lengths.
    /// - Returns: Array of reconstructed audio signals.
    public func reconstructBatch(
        _ magnitudes: [[[Float]]],
        lengths: [Int?]
    ) async -> [[Float]] {
        var results = [[Float]]()
        results.reserveCapacity(magnitudes.count)

        for (mag, len) in zip(magnitudes, lengths) {
            let audio = await reconstruct(mag, length: len)
            results.append(audio)
        }

        return results
    }

    /// Create a default Griffin-Lim configured for common audio ML use cases.
    public static var defaultForML: GriffinLim {
        GriffinLim(
            config: STFTConfig(nFFT: 1024, hopLength: 256, windowType: .hann),
            nIter: 60,
            momentum: 0.99,
            randomPhaseInit: true
        )
    }

    /// Create a Griffin-Lim configured for Banquet model (if magnitude output).
    public static var forBanquet: GriffinLim {
        GriffinLim(
            config: STFTConfig(nFFT: 2048, hopLength: 512, windowType: .hann),
            nIter: 32,
            momentum: 0.99,
            randomPhaseInit: true
        )
    }
}

// MARK: - GPU Acceleration

extension GriffinLim {
    /// Reconstruct audio using GPU-accelerated magnitude replacement.
    ///
    /// Uses Metal GPU for the magnitude replacement step within each iteration.
    /// The STFT/ISTFT are still performed with the existing (already optimized)
    /// implementations, but the magnitude replacement is GPU-accelerated.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length. If nil, estimated from spectrogram.
    /// - Returns: Reconstructed audio signal.
    public func reconstructGPU(_ magnitude: [[Float]], length: Int? = nil) async -> [Float] {
        await reconstructGPU(magnitude, length: length, onIteration: nil)
    }

    /// Reconstruct audio with GPU acceleration and iteration callback.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - length: Expected output length.
    ///   - onIteration: Callback receiving (iteration, reconstructionError).
    /// - Returns: Reconstructed audio signal.
    public func reconstructGPU(
        _ magnitude: [[Float]],
        length: Int?,
        onIteration: (@Sendable (Int, Float) -> Void)?
    ) async -> [Float] {
        guard !magnitude.isEmpty && !magnitude[0].isEmpty else { return [] }

        let nFreqs = magnitude.count
        let nFrames = magnitude[0].count
        let totalElements = nFreqs * nFrames

        // Check if GPU should be used
        guard MetalEngine.isAvailable &&
              MetalEngine.shouldUseGPU(elementCount: totalElements * nIter, operationType: .general) else {
            return await reconstruct(magnitude, length: length, onIteration: onIteration)
        }

        // Initialize Metal engine
        guard let engine = try? MetalEngine() else {
            return await reconstruct(magnitude, length: length, onIteration: onIteration)
        }

        // Estimate output length if not provided
        let outputLength = length ?? estimateOutputLength(nFrames: nFrames)

        // Flatten magnitude for GPU
        var flatMagnitude = [Float](repeating: 0, count: totalElements)
        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                flatMagnitude[f * nFrames + t] = magnitude[f][t]
            }
        }

        // Initialize phase
        var phase = [Float](repeating: 0, count: totalElements)
        if randomPhaseInit {
            for i in 0..<totalElements {
                phase[i] = Float.random(in: -Float.pi...Float.pi)
            }
        }

        // Working buffers
        var currentReal = [Float](repeating: 0, count: totalElements)
        var currentImag = [Float](repeating: 0, count: totalElements)
        var prevReal = [Float](repeating: 0, count: totalElements)
        var prevImag = [Float](repeating: 0, count: totalElements)

        let momScale = momentum / (1.0 + momentum)
        var bestError: Float = Float.infinity
        var bestPhase = phase
        var hasPrevious = false

        for iter in 0..<nIter {
            // 1. Create complex from magnitude + phase
            for i in 0..<totalElements {
                currentReal[i] = flatMagnitude[i] * cos(phase[i])
                currentImag[i] = flatMagnitude[i] * sin(phase[i])
            }

            // 2. Convert to ComplexMatrix
            var realRows = [[Float]]()
            var imagRows = [[Float]]()
            for f in 0..<nFreqs {
                let start = f * nFrames
                realRows.append(Array(currentReal[start..<(start + nFrames)]))
                imagRows.append(Array(currentImag[start..<(start + nFrames)]))
            }
            let complex = ComplexMatrix(real: realRows, imag: imagRows)

            // 3. ISTFT
            let audio = await istft.transform(complex, length: outputLength)

            // 4. Forward STFT
            let reconstructed = await stft.transform(audio)

            // Copy to flat buffers
            for f in 0..<nFreqs {
                let start = f * nFrames
                for t in 0..<nFrames {
                    currentReal[start + t] = reconstructed.real[f][t]
                    currentImag[start + t] = reconstructed.imag[f][t]
                }
            }

            // 5. Compute error (normalized to match CPU path)
            var errorSum: Float = 0
            var targetSum: Float = 0
            for i in 0..<totalElements {
                let currentMag = sqrt(currentReal[i] * currentReal[i] + currentImag[i] * currentImag[i])
                let diff = flatMagnitude[i] - currentMag
                errorSum += diff * diff
                targetSum += flatMagnitude[i] * flatMagnitude[i]
            }
            let currentError = targetSum > 0 ? sqrt(errorSum / targetSum) : 0

            // 6. GPU magnitude replacement
            do {
                let (newReal, newImag) = try await engine.griffinLimMagnitudeReplace(
                    currentReal: currentReal,
                    currentImag: currentImag,
                    targetMagnitude: flatMagnitude,
                    epsilon: 1e-10
                )

                // Apply momentum if not first iteration
                if momentum > 0 && hasPrevious {
                    for i in 0..<totalElements {
                        currentReal[i] = newReal[i] - momScale * prevReal[i]
                        currentImag[i] = newImag[i] - momScale * prevImag[i]
                    }
                } else {
                    currentReal = newReal
                    currentImag = newImag
                }

                // Save for next iteration's momentum
                prevReal = newReal
                prevImag = newImag
                hasPrevious = true

                // Extract phase
                for i in 0..<totalElements {
                    phase[i] = atan2(currentImag[i], currentReal[i])
                }
            } catch {
                // GPU failed, fall back to CPU for this iteration
                for i in 0..<totalElements {
                    let mag = sqrt(currentReal[i] * currentReal[i] + currentImag[i] * currentImag[i])
                    if mag > 1e-10 {
                        let scale = flatMagnitude[i] / mag
                        currentReal[i] *= scale
                        currentImag[i] *= scale
                    } else {
                        currentReal[i] = flatMagnitude[i]
                        currentImag[i] = 0
                    }
                    phase[i] = atan2(currentImag[i], currentReal[i])
                }
            }

            // Track best
            if currentError < bestError {
                bestError = currentError
                bestPhase = phase
            }

            if let callback = onIteration {
                callback(iter, currentError)
            }
        }

        // Final reconstruction with best phase
        for i in 0..<totalElements {
            currentReal[i] = flatMagnitude[i] * cos(bestPhase[i])
            currentImag[i] = flatMagnitude[i] * sin(bestPhase[i])
        }

        var finalRealRows = [[Float]]()
        var finalImagRows = [[Float]]()
        for f in 0..<nFreqs {
            let start = f * nFrames
            finalRealRows.append(Array(currentReal[start..<(start + nFrames)]))
            finalImagRows.append(Array(currentImag[start..<(start + nFrames)]))
        }
        let finalComplex = ComplexMatrix(real: finalRealRows, imag: finalImagRows)

        return await istft.transform(finalComplex, length: outputLength)
    }
}
