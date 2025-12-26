import Accelerate
import Foundation
import SwiftRosaCore

/// Configuration for Tonnetz feature extraction.
public struct TonnetzConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Number of chroma bins (default: 12).
    public let nChroma: Int

    /// Create Tonnetz configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - nChroma: Number of chroma bins (default: 12).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        nChroma: Int = 12
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.nChroma = nChroma
    }
}

/// Tonal centroid features (Tonnetz).
///
/// Projects chroma features onto a 6-dimensional tonal space that
/// represents harmonic relationships:
/// - Fifths (perfect fifth intervals)
/// - Minor thirds
/// - Major thirds
///
/// Each relationship is represented by sin/cos components.
///
/// ## Example Usage
///
/// ```swift
/// let tonnetz = Tonnetz(config: TonnetzConfig(sampleRate: 22050))
///
/// // From raw audio
/// let features = await tonnetz.transform(audioSignal)
///
/// // From pre-computed chroma
/// let features = tonnetz.transform(chroma: chromaFeatures)
/// ```
///
/// ## Output
///
/// 6 features per frame:
/// 1. Fifth circle - sin component
/// 2. Fifth circle - cos component
/// 3. Minor third circle - sin component
/// 4. Minor third circle - cos component
/// 5. Major third circle - sin component
/// 6. Major third circle - cos component
public struct Tonnetz: Sendable {
    /// Configuration.
    public let config: TonnetzConfig

    /// Pre-computed transformation matrix (6 x nChroma).
    private let transformMatrix: [[Float]]

    /// Chromagram extractor.
    private let chromagram: Chromagram

    /// Create Tonnetz feature extractor.
    ///
    /// - Parameter config: Configuration.
    public init(config: TonnetzConfig = TonnetzConfig()) {
        self.config = config

        // Create chromagram with matching configuration
        self.chromagram = Chromagram(config: ChromagramConfig(
            sampleRate: config.sampleRate,
            hopLength: config.hopLength,
            nChroma: config.nChroma
        ))

        // Pre-compute transformation matrix
        self.transformMatrix = Tonnetz.computeTransformMatrix(nChroma: config.nChroma)
    }

    /// Compute Tonnetz features from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tonnetz features of shape (6, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        let chroma = await chromagram.transform(signal)
        return transform(chroma: chroma)
    }

    /// Compute Tonnetz features from pre-computed chroma.
    ///
    /// - Parameter chroma: Chromagram of shape (nChroma, nFrames).
    /// - Returns: Tonnetz features of shape (6, nFrames).
    public func transform(chroma: [[Float]]) -> [[Float]] {
        guard !chroma.isEmpty, !chroma[0].isEmpty else { return [] }
        guard chroma.count == config.nChroma else { return [] }

        let nFrames = chroma[0].count

        // Normalize chroma to sum to 1 per frame
        let normalizedChroma = normalizeChroma(chroma)

        // Apply transformation: tonnetz = transformMatrix @ chroma
        var result = [[Float]]()
        result.reserveCapacity(6)

        for d in 0..<6 {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                var sum: Float = 0
                for c in 0..<config.nChroma {
                    sum += transformMatrix[d][c] * normalizedChroma[c][frameIdx]
                }
                row[frameIdx] = sum
            }

            result.append(row)
        }

        return result
    }

    // MARK: - Private Implementation

    /// Compute the 6x12 transformation matrix.
    ///
    /// Matches librosa.feature.tonnetz exactly:
    /// ```python
    /// dim_map = np.linspace(0, 12, num=n_chroma, endpoint=False)
    /// scale = [7/6, 7/6, 3/2, 3/2, 2/3, 2/3]
    /// V = outer(scale, dim_map)
    /// V[::2] -= 0.5  # subtract 0.5 from even rows
    /// R = [1, 1, 1, 1, 0.5, 0.5]
    /// phi = R * cos(π * V)
    /// ```
    private static func computeTransformMatrix(nChroma: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(6)

        // Scale factors for each dimension
        let scale: [Float] = [7.0 / 6.0, 7.0 / 6.0, 3.0 / 2.0, 3.0 / 2.0, 2.0 / 3.0, 2.0 / 3.0]

        // Amplitude scaling (major third uses 0.5)
        let R: [Float] = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]

        for row in 0..<6 {
            var rowValues = [Float](repeating: 0, count: nChroma)

            for n in 0..<nChroma {
                let nf = Float(n)

                // V = scale[row] * n
                var v = scale[row] * nf

                // Even rows (0, 2, 4) get -0.5 offset
                if row % 2 == 0 {
                    v -= 0.5
                }

                // phi = R * cos(π * V)
                rowValues[n] = R[row] * cos(Float.pi * v)
            }

            matrix.append(rowValues)
        }

        return matrix
    }

    /// Normalize chroma to sum to 1 per frame.
    private func normalizeChroma(_ chroma: [[Float]]) -> [[Float]] {
        let nChroma = chroma.count
        let nFrames = chroma[0].count

        var normalized = chroma

        for frameIdx in 0..<nFrames {
            var sum: Float = 0
            for c in 0..<nChroma {
                sum += chroma[c][frameIdx]
            }

            // Avoid division by zero
            let norm = max(sum, 1e-10)

            for c in 0..<nChroma {
                normalized[c][frameIdx] /= norm
            }
        }

        return normalized
    }
}

// MARK: - Presets

extension Tonnetz {
    /// Standard Tonnetz configuration.
    public static var standard: Tonnetz {
        Tonnetz(config: TonnetzConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12
        ))
    }

    /// Tonnetz for music analysis with larger hop.
    public static var music: Tonnetz {
        Tonnetz(config: TonnetzConfig(
            sampleRate: 22050,
            hopLength: 1024,
            nChroma: 12
        ))
    }
}
