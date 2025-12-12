import Accelerate
import Foundation

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

        // Initialize phase
        var phase = initializePhase(rows: nFreqs, cols: nFrames)

        // Track previous reconstructed spectrogram for Fast Griffin-Lim momentum
        var previousReconstructed: ComplexMatrix? = nil

        // Pre-compute momentum scale factor (librosa formula: momentum / (1 + momentum))
        let momScale = momentum / (1.0 + momentum)

        // Track best phase seen across all iterations
        // This ensures more iterations can never produce worse results
        var bestPhase = phase
        var bestError: Float = Float.infinity

        // Main iteration loop
        for iter in 0..<nIter {
            // 1. Construct complex spectrogram from magnitude and current phase
            let complex = ComplexMatrix.fromPolar(magnitude: magnitude, phase: phase)

            // 2. Inverse STFT to get audio estimate
            let audio = await istft.transform(complex, length: outputLength)

            // 3. Forward STFT of audio estimate (this is G(c_n) = "rebuilt")
            let reconstructed = await stft.transform(audio)

            // Compute error for best-phase tracking
            let currentError = computeReconstructionError(
                targetMagnitude: magnitude,
                estimatedComplex: reconstructed
            )

            // Track best input phase seen - ensures monotonic improvement across runs
            // We save the input phase that produced this error, not reconstructed.phase
            if currentError < bestError {
                bestError = currentError
                bestPhase = phase.map { $0 }
            }

            // 4. Apply Fast Griffin-Lim momentum (librosa formula)
            // angles = rebuilt - (momentum / (1 + momentum)) * tprev
            var angles: ComplexMatrix
            if momentum > 0 && previousReconstructed != nil {
                angles = applyLibrosaMomentum(
                    rebuilt: reconstructed,
                    tprev: previousReconstructed!,
                    momScale: momScale
                )
            } else {
                angles = reconstructed
            }

            // Store current as previous for next iteration
            previousReconstructed = reconstructed

            // 5. Extract phase (normalize to unit magnitude)
            phase = angles.phase

            // Report progress
            if let callback = onIteration {
                callback(iter, currentError)
            }
        }

        // Final reconstruction with best phase seen (not just final phase)
        // This ensures more iterations can never produce worse results
        let finalComplex = ComplexMatrix.fromPolar(magnitude: magnitude, phase: bestPhase)
        return await istft.transform(finalComplex, length: outputLength)
    }

    // MARK: - Private Implementation

    /// Initialize phase matrix.
    ///
    /// For deterministic initialization (randomPhaseInit=false), uses zero phase
    /// which provides a neutral starting point. Griffin-Lim will converge to a
    /// valid phase estimate from this initialization.
    private func initializePhase(rows: Int, cols: Int) -> [[Float]] {
        var phase = [[Float]]()
        phase.reserveCapacity(rows)

        if randomPhaseInit {
            // Random phase in [-pi, pi]
            for _ in 0..<rows {
                let row = (0..<cols).map { _ in Float.random(in: -Float.pi...Float.pi) }
                phase.append(row)
            }
        } else {
            // Zero phase initialization - simple and effective
            for _ in 0..<rows {
                let row = [Float](repeating: 0, count: cols)
                phase.append(row)
            }
        }

        return phase
    }

    /// Compute the shortest angular difference between two angles.
    ///
    /// This correctly handles wrap-around at ±π and always returns
    /// a value in [-π, π] representing the shortest path from b to a.
    @inline(__always)
    private func angularDifference(_ a: Float, _ b: Float) -> Float {
        let diff = a - b
        // atan2(sin, cos) correctly handles wrap-around
        return atan2(sin(diff), cos(diff))
    }

    /// Apply momentum update in complex domain (librosa's Fast Griffin-Lim formula).
    ///
    /// Librosa uses: angles = rebuilt - (momentum / (1 + momentum)) * tprev
    /// where rebuilt is G(c_n) and tprev is G(c_{n-1}).
    ///
    /// This is then normalized to unit magnitude to extract phase.
    private func applyLibrosaMomentum(
        rebuilt: ComplexMatrix,
        tprev: ComplexMatrix,
        momScale: Float
    ) -> ComplexMatrix {
        let rows = rebuilt.rows
        let cols = rebuilt.cols

        var resultReal = [[Float]]()
        var resultImag = [[Float]]()
        resultReal.reserveCapacity(rows)
        resultImag.reserveCapacity(rows)

        for i in 0..<rows {
            var rowReal = [Float](repeating: 0, count: cols)
            var rowImag = [Float](repeating: 0, count: cols)

            for j in 0..<cols {
                // angles = rebuilt - momScale * tprev
                rowReal[j] = rebuilt.real[i][j] - momScale * tprev.real[i][j]
                rowImag[j] = rebuilt.imag[i][j] - momScale * tprev.imag[i][j]
            }

            resultReal.append(rowReal)
            resultImag.append(rowImag)
        }

        return ComplexMatrix(real: resultReal, imag: resultImag)
    }

    /// Compute reconstruction error (for monitoring convergence).
    private func computeReconstructionError(
        targetMagnitude: [[Float]],
        estimatedComplex: ComplexMatrix
    ) -> Float {
        let estimatedMag = estimatedComplex.magnitude
        let rows = targetMagnitude.count
        let cols = targetMagnitude[0].count

        var sumSquaredError: Float = 0
        var sumTargetSquared: Float = 0

        for i in 0..<rows {
            for j in 0..<cols {
                let diff = targetMagnitude[i][j] - estimatedMag[i][j]
                sumSquaredError += diff * diff
                sumTargetSquared += targetMagnitude[i][j] * targetMagnitude[i][j]
            }
        }

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
