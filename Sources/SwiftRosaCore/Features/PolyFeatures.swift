import Accelerate
import Foundation

/// Configuration for polynomial feature extraction.
public struct PolyFeaturesConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// FFT size for frequency bin computation.
    public let nFFT: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Polynomial order (number of coefficients = order + 1).
    public let order: Int

    /// Frequency range to fit (optional, defaults to full range).
    public let freqRange: (min: Float, max: Float)?

    /// Create polynomial features configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop length (default: 512).
    ///   - order: Polynomial order (default: 1).
    ///   - freqRange: Optional frequency range to fit.
    public init(
        sampleRate: Float = 22050,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        order: Int = 1,
        freqRange: (min: Float, max: Float)? = nil
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.order = max(0, order)
        self.freqRange = freqRange
    }

    /// Number of frequency bins.
    public var nFreqs: Int { nFFT / 2 + 1 }

    /// Frequency array for spectral bins.
    public var frequencies: [Float] {
        let binWidth = sampleRate / Float(nFFT)
        return (0..<nFreqs).map { Float($0) * binWidth }
    }
}

/// Polynomial feature extraction from spectrograms.
///
/// Fits polynomials to each frame of a spectrogram and extracts the
/// coefficients as features. This captures the overall spectral shape
/// (tilt, curvature, etc.) in a compact representation.
///
/// ## Example Usage
///
/// ```swift
/// let poly = PolyFeatures(config: PolyFeaturesConfig(order: 2))
///
/// // Extract polynomial features from magnitude spectrogram
/// let coeffs = poly.transform(magnitudeSpec: spectrogram)
/// // Result shape: (order+1, nFrames)
/// // coeffs[0] = intercepts, coeffs[1] = linear, coeffs[2] = quadratic
/// ```
///
/// ## Formula
///
/// For each frame, fit: `S(f) = c0 + c1*f + c2*f² + ... + cn*f^n`
///
/// Using least squares: `c = (F^T F)^(-1) F^T S`
public struct PolyFeatures: Sendable {
    /// Configuration.
    public let config: PolyFeaturesConfig

    /// Pre-computed design matrix for least squares.
    private let designMatrix: [[Float]]

    /// Pre-computed (F^T F)^(-1) F^T for efficient least squares.
    private let leastSquaresMatrix: [[Float]]

    /// Indices of frequency bins to use (based on freqRange).
    private let freqIndices: [Int]

    /// Create polynomial feature extractor.
    ///
    /// - Parameter config: Configuration.
    public init(config: PolyFeaturesConfig = PolyFeaturesConfig()) {
        self.config = config

        let frequencies = config.frequencies

        // Determine which frequency bins to use
        var indices: [Int] = []
        if let range = config.freqRange {
            for (i, f) in frequencies.enumerated() {
                if f >= range.min && f <= range.max {
                    indices.append(i)
                }
            }
        } else {
            indices = Array(0..<frequencies.count)
        }
        self.freqIndices = indices

        // Normalize frequencies to [0, 1] for numerical stability
        let selectedFreqs: [Float]
        if indices.isEmpty {
            selectedFreqs = [0]
        } else {
            let minF = frequencies[indices.first!]
            let maxF = frequencies[indices.last!]
            let range = maxF - minF
            if range > 0 {
                selectedFreqs = indices.map { (frequencies[$0] - minF) / range }
            } else {
                selectedFreqs = indices.map { _ in Float(0.5) }
            }
        }

        // Build Vandermonde design matrix F: (nFreqs, order+1)
        let nCoeffs = config.order + 1
        var F = [[Float]]()
        F.reserveCapacity(nCoeffs)

        for p in 0...config.order {
            var row = [Float](repeating: 0, count: selectedFreqs.count)
            for (i, f) in selectedFreqs.enumerated() {
                row[i] = pow(f, Float(p))
            }
            F.append(row)
        }
        self.designMatrix = F

        // Compute (F^T F)^(-1) F^T for least squares
        // This is the pseudo-inverse when F has more rows than columns
        self.leastSquaresMatrix = PolyFeatures.computeLeastSquaresMatrix(F, nCoeffs: nCoeffs)
    }

    /// Extract polynomial features from magnitude spectrogram.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Polynomial coefficients of shape (order+1, nFrames).
    public func transform(magnitudeSpec: [[Float]]) -> [[Float]] {
        guard !magnitudeSpec.isEmpty, !magnitudeSpec[0].isEmpty else { return [] }
        guard !freqIndices.isEmpty else { return [] }

        let nFrames = magnitudeSpec[0].count
        let nCoeffs = config.order + 1

        var result = [[Float]]()
        result.reserveCapacity(nCoeffs)
        for _ in 0..<nCoeffs {
            result.append([Float](repeating: 0, count: nFrames))
        }

        // For each frame, extract spectrum values and compute coefficients
        for frameIdx in 0..<nFrames {
            // Extract spectrum values at selected frequencies
            var spectrum = [Float](repeating: 0, count: freqIndices.count)
            for (i, freqIdx) in freqIndices.enumerated() {
                if freqIdx < magnitudeSpec.count {
                    spectrum[i] = magnitudeSpec[freqIdx][frameIdx]
                }
            }

            // Compute coefficients: c = leastSquaresMatrix @ spectrum
            let coeffs = matrixVectorMultiply(leastSquaresMatrix, spectrum)

            // Store in result
            for (p, coeff) in coeffs.enumerated() {
                result[p][frameIdx] = coeff
            }
        }

        return result
    }

    /// Extract polynomial features from STFT (computes magnitude internally).
    ///
    /// - Parameter stft: STFT processor.
    /// - Parameter signal: Input audio signal.
    /// - Returns: Polynomial coefficients of shape (order+1, nFrames).
    public func transform(stft: STFT, signal: [Float]) async -> [[Float]] {
        let magnitude = await stft.magnitude(signal)
        return transform(magnitudeSpec: magnitude)
    }

    // MARK: - Private Implementation

    /// Compute least squares matrix (F^T F)^(-1) F^T.
    private static func computeLeastSquaresMatrix(
        _ F: [[Float]],
        nCoeffs: Int
    ) -> [[Float]] {
        guard !F.isEmpty, !F[0].isEmpty else { return [] }

        let nFreqs = F[0].count

        // For polynomial fitting, we need to solve F^T @ S = coefficients
        // where F is (nCoeffs, nFreqs) and S is (nFreqs,)
        // Result: c = (F @ F^T)^(-1) @ F @ S

        // Compute F @ F^T: (nCoeffs, nCoeffs)
        var FFT = [[Float]](repeating: [Float](repeating: 0, count: nCoeffs), count: nCoeffs)
        for i in 0..<nCoeffs {
            for j in 0..<nCoeffs {
                var sum: Float = 0
                vDSP_dotpr(F[i], 1, F[j], 1, &sum, vDSP_Length(nFreqs))
                FFT[i][j] = sum
            }
        }

        // Invert (F @ F^T) using simple Gaussian elimination for small matrices
        guard let FFTInv = invertMatrix(FFT) else {
            // Return identity-like matrix on failure
            var identity = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nCoeffs)
            for i in 0..<min(nCoeffs, nFreqs) {
                identity[i][i] = 1
            }
            return identity
        }

        // Compute (F @ F^T)^(-1) @ F: (nCoeffs, nFreqs)
        var result = [[Float]]()
        result.reserveCapacity(nCoeffs)

        for i in 0..<nCoeffs {
            var row = [Float](repeating: 0, count: nFreqs)
            for j in 0..<nFreqs {
                var sum: Float = 0
                for k in 0..<nCoeffs {
                    sum += FFTInv[i][k] * F[k][j]
                }
                row[j] = sum
            }
            result.append(row)
        }

        return result
    }

    /// Simple matrix inversion for small matrices (up to ~10x10).
    private static func invertMatrix(_ matrix: [[Float]]) -> [[Float]]? {
        let n = matrix.count
        guard n > 0 && matrix[0].count == n else { return nil }

        // Augment with identity
        var aug = [[Float]]()
        for i in 0..<n {
            var row = matrix[i]
            for j in 0..<n {
                row.append(i == j ? 1 : 0)
            }
            aug.append(row)
        }

        // Gaussian elimination with partial pivoting
        for col in 0..<n {
            // Find pivot
            var maxRow = col
            var maxVal = abs(aug[col][col])
            for row in (col + 1)..<n {
                if abs(aug[row][col]) > maxVal {
                    maxVal = abs(aug[row][col])
                    maxRow = row
                }
            }

            // Swap rows
            if maxRow != col {
                let temp = aug[col]
                aug[col] = aug[maxRow]
                aug[maxRow] = temp
            }

            // Check for singularity
            guard abs(aug[col][col]) > 1e-10 else { return nil }

            // Scale pivot row
            let scale = aug[col][col]
            for j in 0..<(2 * n) {
                aug[col][j] /= scale
            }

            // Eliminate column
            for row in 0..<n {
                if row != col {
                    let factor = aug[row][col]
                    for j in 0..<(2 * n) {
                        aug[row][j] -= factor * aug[col][j]
                    }
                }
            }
        }

        // Extract inverse
        var inverse = [[Float]]()
        for i in 0..<n {
            inverse.append(Array(aug[i][n..<(2 * n)]))
        }

        return inverse
    }

    /// Matrix-vector multiplication.
    private func matrixVectorMultiply(_ matrix: [[Float]], _ vector: [Float]) -> [Float] {
        guard !matrix.isEmpty else { return [] }

        let rows = matrix.count
        var result = [Float](repeating: 0, count: rows)

        for i in 0..<rows {
            if matrix[i].count == vector.count {
                vDSP_dotpr(matrix[i], 1, vector, 1, &result[i], vDSP_Length(vector.count))
            }
        }

        return result
    }
}

// MARK: - Presets

extension PolyFeatures {
    /// Linear fit (spectral tilt).
    public static var linear: PolyFeatures {
        PolyFeatures(config: PolyFeaturesConfig(order: 1))
    }

    /// Quadratic fit (spectral tilt + curvature).
    public static var quadratic: PolyFeatures {
        PolyFeatures(config: PolyFeaturesConfig(order: 2))
    }
}
