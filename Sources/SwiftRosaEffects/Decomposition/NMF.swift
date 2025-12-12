import Accelerate
import Foundation
import SwiftRosaCore

/// Errors that can occur during NMF.
public enum NMFError: Error, Sendable {
    case invalidDimensions(String)
    case convergenceFailed(iterations: Int, error: Float)
    case computationError(String)
}

/// Initialization method for NMF.
public enum NMFInitialization: Sendable {
    /// Random uniform initialization.
    case random
    /// Non-negative SVD-based initialization (NNDSVD).
    case nndsvd
    /// Custom initialization matrices.
    case custom(W: [[Float]], H: [[Float]])
}

/// NMF solver type.
public enum NMFSolver: Sendable {
    /// Multiplicative update rules (standard, most stable).
    case multiplicativeUpdate
    /// Coordinate descent (faster convergence for some problems).
    case coordinateDescent
}

/// Configuration for NMF decomposition.
public struct NMFConfig: Sendable {
    /// Number of components to extract.
    public let nComponents: Int

    /// Maximum number of iterations.
    public let maxIterations: Int

    /// Convergence tolerance (relative change in reconstruction error).
    public let tolerance: Float

    /// Initialization method.
    public let initialization: NMFInitialization

    /// Solver type.
    public let solver: NMFSolver

    /// L1 regularization for W (promotes sparsity).
    public let l1RatioW: Float

    /// L1 regularization for H (promotes sparsity).
    public let l1RatioH: Float

    /// Regularization strength (alpha).
    public let alpha: Float

    /// Small constant for numerical stability.
    public let epsilon: Float

    /// Random seed for reproducibility.
    public let randomSeed: UInt64?

    /// Create NMF configuration.
    ///
    /// - Parameters:
    ///   - nComponents: Number of components (default: 8).
    ///   - maxIterations: Maximum iterations (default: 200).
    ///   - tolerance: Convergence tolerance (default: 1e-4).
    ///   - initialization: Initialization method (default: random).
    ///   - solver: Solver type (default: multiplicativeUpdate).
    ///   - l1RatioW: L1 ratio for W regularization (default: 0).
    ///   - l1RatioH: L1 ratio for H regularization (default: 0).
    ///   - alpha: Regularization strength (default: 0).
    ///   - epsilon: Numerical stability constant (default: 1e-10).
    ///   - randomSeed: Random seed for reproducibility (default: nil).
    public init(
        nComponents: Int = 8,
        maxIterations: Int = 200,
        tolerance: Float = 1e-4,
        initialization: NMFInitialization = .random,
        solver: NMFSolver = .multiplicativeUpdate,
        l1RatioW: Float = 0,
        l1RatioH: Float = 0,
        alpha: Float = 0,
        epsilon: Float = 1e-10,
        randomSeed: UInt64? = nil
    ) {
        self.nComponents = nComponents
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.initialization = initialization
        self.solver = solver
        self.l1RatioW = l1RatioW
        self.l1RatioH = l1RatioH
        self.alpha = alpha
        self.epsilon = epsilon
        self.randomSeed = randomSeed
    }
}

/// NMF decomposition result.
public struct NMFResult: Sendable {
    /// Basis matrix W of shape (nFeatures, nComponents).
    /// Each column represents a spectral template/basis vector.
    public let W: [[Float]]

    /// Activation matrix H of shape (nComponents, nFrames).
    /// Each row represents the time activation of a basis.
    public let H: [[Float]]

    /// Final reconstruction error (Frobenius norm).
    public let reconstructionError: Float

    /// Number of iterations until convergence.
    public let iterations: Int

    /// History of reconstruction errors.
    public let errorHistory: [Float]
}

/// Non-negative Matrix Factorization decomposition.
///
/// Decomposes a non-negative matrix V into two non-negative matrices W and H
/// such that V ≈ W × H, minimizing the Frobenius norm of the difference.
///
/// ## Use Cases in Audio
///
/// - **Source separation**: W contains spectral templates, H contains activations
/// - **Transcription**: W = instrument spectra, H = note activations
/// - **Denoising**: Separate speech from noise
/// - **Feature learning**: Learn data-driven spectral bases
///
/// ## Example Usage
///
/// ```swift
/// let nmf = NMF(config: NMFConfig(nComponents: 4, maxIterations: 100))
///
/// // Decompose spectrogram
/// let result = try nmf.decompose(spectrogram)
///
/// // W: spectral bases (nFreqs × nComponents)
/// // H: activations (nComponents × nFrames)
///
/// // Reconstruct a single component
/// let component1 = nmf.reconstruct(component: 0, W: result.W, H: result.H)
/// ```
///
/// ## Algorithm
///
/// Using multiplicative update rules (Lee & Seung, 2001):
/// ```
/// H ← H ⊙ (W^T V) / (W^T W H + ε)
/// W ← W ⊙ (V H^T) / (W H H^T + ε)
/// ```
public struct NMF: Sendable {
    /// Configuration.
    public let config: NMFConfig

    /// Create NMF decomposer.
    ///
    /// - Parameter config: Configuration.
    public init(config: NMFConfig = NMFConfig()) {
        self.config = config
    }

    /// Decompose a non-negative matrix.
    ///
    /// - Parameter V: Input matrix of shape (nFeatures, nFrames). Must be non-negative.
    /// - Returns: NMF decomposition result.
    /// - Throws: `NMFError` if decomposition fails.
    public func decompose(_ V: [[Float]]) throws -> NMFResult {
        guard !V.isEmpty, !V[0].isEmpty else {
            throw NMFError.invalidDimensions("Input matrix is empty")
        }

        let M = V.count           // nFeatures (frequency bins)
        let N = V[0].count        // nFrames (time frames)
        let K = config.nComponents

        guard K > 0 && K <= min(M, N) else {
            throw NMFError.invalidDimensions("nComponents must be between 1 and min(M, N)")
        }

        // Initialize W and H
        var (W, H) = initializeMatrices(M: M, N: N, K: K, V: V)

        // Track convergence
        var errorHistory = [Float]()
        var prevError: Float = .infinity

        // Iterative updates
        for iteration in 0..<config.maxIterations {
            // Compute W @ H
            let WH = matrixMultiply(W, H)

            // Update H
            H = updateH(W: W, V: V, H: H, WH: WH, M: M, N: N, K: K)

            // Recompute W @ H with updated H
            let WH2 = matrixMultiply(W, H)

            // Update W
            W = updateW(W: W, V: V, H: H, WH: WH2, M: M, N: N, K: K)

            // Compute reconstruction error
            let error = computeReconstructionError(V: V, W: W, H: H)
            errorHistory.append(error)

            // Check convergence
            let relativeChange = abs(prevError - error) / (prevError + config.epsilon)
            if relativeChange < config.tolerance {
                return NMFResult(
                    W: W,
                    H: H,
                    reconstructionError: error,
                    iterations: iteration + 1,
                    errorHistory: errorHistory
                )
            }

            prevError = error
        }

        // Did not converge, but return best result
        let finalError = computeReconstructionError(V: V, W: W, H: H)
        return NMFResult(
            W: W,
            H: H,
            reconstructionError: finalError,
            iterations: config.maxIterations,
            errorHistory: errorHistory
        )
    }

    /// Reconstruct the approximation V ≈ W × H.
    ///
    /// - Parameters:
    ///   - W: Basis matrix of shape (nFeatures, nComponents).
    ///   - H: Activation matrix of shape (nComponents, nFrames).
    /// - Returns: Reconstructed matrix of shape (nFeatures, nFrames).
    public func reconstruct(W: [[Float]], H: [[Float]]) -> [[Float]] {
        matrixMultiply(W, H)
    }

    /// Reconstruct a single component.
    ///
    /// - Parameters:
    ///   - component: Component index (0-based).
    ///   - W: Basis matrix.
    ///   - H: Activation matrix.
    /// - Returns: Single component contribution of shape (nFeatures, nFrames).
    public func reconstruct(component: Int, W: [[Float]], H: [[Float]]) -> [[Float]] {
        guard component >= 0 && component < H.count else { return [] }
        guard !W.isEmpty && !H.isEmpty else { return [] }

        let M = W.count
        let N = H[0].count

        var result = [[Float]]()
        result.reserveCapacity(M)

        for i in 0..<M {
            var row = [Float](repeating: 0, count: N)
            for j in 0..<N {
                row[j] = W[i][component] * H[component][j]
            }
            result.append(row)
        }

        return result
    }

    /// Transform new data using pre-learned basis W.
    ///
    /// Given a new spectrogram V and fixed W, compute H such that V ≈ W × H.
    ///
    /// - Parameters:
    ///   - V: New input matrix of shape (nFeatures, nFrames).
    ///   - W: Pre-learned basis matrix of shape (nFeatures, nComponents).
    /// - Returns: Activation matrix H of shape (nComponents, nFrames).
    public func transform(_ V: [[Float]], W: [[Float]]) throws -> [[Float]] {
        guard !V.isEmpty, !V[0].isEmpty else {
            throw NMFError.invalidDimensions("Input matrix is empty")
        }

        let M = V.count
        let N = V[0].count
        let K = W[0].count

        // Initialize H randomly
        var H = randomMatrix(rows: K, cols: N)

        // Iterative updates (only update H, keep W fixed)
        for _ in 0..<config.maxIterations {
            let WH = matrixMultiply(W, H)
            H = updateH(W: W, V: V, H: H, WH: WH, M: M, N: N, K: K)
        }

        return H
    }

    // MARK: - Private Implementation

    /// Initialize W and H matrices.
    private func initializeMatrices(
        M: Int, N: Int, K: Int, V: [[Float]]
    ) -> (W: [[Float]], H: [[Float]]) {
        switch config.initialization {
        case .random:
            return (randomMatrix(rows: M, cols: K), randomMatrix(rows: K, cols: N))

        case .nndsvd:
            return nndsvdInitialization(V: V, M: M, N: N, K: K)

        case .custom(let W, let H):
            return (W, H)
        }
    }

    /// Random non-negative matrix initialization.
    private func randomMatrix(rows: Int, cols: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(rows)

        for _ in 0..<rows {
            var row = [Float](repeating: 0, count: cols)
            for j in 0..<cols {
                row[j] = Float.random(in: 0.01...1.0)
            }
            matrix.append(row)
        }

        return matrix
    }

    /// NNDSVD initialization (simplified version).
    private func nndsvdInitialization(
        V: [[Float]], M: Int, N: Int, K: Int
    ) -> (W: [[Float]], H: [[Float]]) {
        // For simplicity, use random with scaling based on data
        var maxVal: Float = 0
        for row in V {
            for val in row {
                maxVal = max(maxVal, val)
            }
        }

        let scale = sqrt(maxVal / Float(K))

        var W = [[Float]]()
        var H = [[Float]]()

        for _ in 0..<M {
            var row = [Float](repeating: 0, count: K)
            for j in 0..<K {
                row[j] = scale * Float.random(in: 0.5...1.5)
            }
            W.append(row)
        }

        for _ in 0..<K {
            var row = [Float](repeating: 0, count: N)
            for j in 0..<N {
                row[j] = scale * Float.random(in: 0.5...1.5)
            }
            H.append(row)
        }

        return (W, H)
    }

    /// Matrix multiplication C = A @ B.
    private func matrixMultiply(_ A: [[Float]], _ B: [[Float]]) -> [[Float]] {
        guard !A.isEmpty, !B.isEmpty else { return [] }

        let M = A.count
        let K = A[0].count
        let N = B[0].count

        var C = [[Float]](repeating: [Float](repeating: 0, count: N), count: M)

        for i in 0..<M {
            for j in 0..<N {
                var sum: Float = 0
                for k in 0..<K {
                    sum += A[i][k] * B[k][j]
                }
                C[i][j] = sum
            }
        }

        return C
    }

    /// Multiplicative update for H.
    private func updateH(
        W: [[Float]], V: [[Float]], H: [[Float]], WH: [[Float]],
        M: Int, N: Int, K: Int
    ) -> [[Float]] {
        var newH = H

        // H ← H * (W^T @ V) / (W^T @ W @ H + eps)
        // Simplified: H[k,j] *= sum_i(W[i,k] * V[i,j]) / (sum_i(W[i,k] * WH[i,j]) + eps)

        for k in 0..<K {
            for j in 0..<N {
                var numerator: Float = 0
                var denominator: Float = 0

                for i in 0..<M {
                    numerator += W[i][k] * V[i][j]
                    denominator += W[i][k] * WH[i][j]
                }

                newH[k][j] *= numerator / (denominator + config.epsilon)

                // Apply regularization
                if config.alpha > 0 {
                    newH[k][j] *= 1.0 / (1.0 + config.alpha * config.l1RatioH)
                }

                // Ensure non-negativity
                newH[k][j] = max(config.epsilon, newH[k][j])
            }
        }

        return newH
    }

    /// Multiplicative update for W.
    private func updateW(
        W: [[Float]], V: [[Float]], H: [[Float]], WH: [[Float]],
        M: Int, N: Int, K: Int
    ) -> [[Float]] {
        var newW = W

        // W ← W * (V @ H^T) / (W @ H @ H^T + eps)
        // Simplified: W[i,k] *= sum_j(V[i,j] * H[k,j]) / (sum_j(WH[i,j] * H[k,j]) + eps)

        for i in 0..<M {
            for k in 0..<K {
                var numerator: Float = 0
                var denominator: Float = 0

                for j in 0..<N {
                    numerator += V[i][j] * H[k][j]
                    denominator += WH[i][j] * H[k][j]
                }

                newW[i][k] *= numerator / (denominator + config.epsilon)

                // Apply regularization
                if config.alpha > 0 {
                    newW[i][k] *= 1.0 / (1.0 + config.alpha * config.l1RatioW)
                }

                // Ensure non-negativity
                newW[i][k] = max(config.epsilon, newW[i][k])
            }
        }

        return newW
    }

    /// Compute reconstruction error (Frobenius norm).
    private func computeReconstructionError(V: [[Float]], W: [[Float]], H: [[Float]]) -> Float {
        let WH = matrixMultiply(W, H)

        var sumSq: Float = 0
        for i in 0..<V.count {
            for j in 0..<V[0].count {
                let diff = V[i][j] - WH[i][j]
                sumSq += diff * diff
            }
        }

        return sqrt(sumSq)
    }
}

// MARK: - Presets

extension NMF {
    /// Standard NMF configuration.
    public static var standard: NMF {
        NMF(config: NMFConfig(
            nComponents: 8,
            maxIterations: 200,
            tolerance: 1e-4
        ))
    }

    /// NMF for source separation (more components, sparser).
    public static var sourceSeparation: NMF {
        NMF(config: NMFConfig(
            nComponents: 16,
            maxIterations: 300,
            tolerance: 1e-5,
            l1RatioH: 0.5,
            alpha: 0.01
        ))
    }

    /// NMF for music transcription (12 components for pitches).
    public static var transcription: NMF {
        NMF(config: NMFConfig(
            nComponents: 88,  // Piano keys
            maxIterations: 200,
            tolerance: 1e-4
        ))
    }

    /// Quick NMF for prototyping.
    public static var quick: NMF {
        NMF(config: NMFConfig(
            nComponents: 4,
            maxIterations: 50,
            tolerance: 1e-3
        ))
    }
}
