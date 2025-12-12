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
    /// phi[0,n] = sin(pi * n * 7 / 6)   # Fifth, sin
    /// phi[1,n] = cos(pi * n * 7 / 6)   # Fifth, cos
    /// phi[2,n] = sin(pi * n * 3 / 2)   # Minor third, sin (3 semitones)
    /// phi[3,n] = cos(pi * n * 3 / 2)   # Minor third, cos
    /// phi[4,n] = sin(pi * n * 2 / 3)   # Major third, sin (4 semitones)
    /// phi[5,n] = cos(pi * n * 2 / 3)   # Major third, cos
    private static func computeTransformMatrix(nChroma: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(6)

        // Fifth circle (7 semitones = perfect fifth)
        var fifthSin = [Float](repeating: 0, count: nChroma)
        var fifthCos = [Float](repeating: 0, count: nChroma)

        // Minor third circle (3 semitones)
        var minorSin = [Float](repeating: 0, count: nChroma)
        var minorCos = [Float](repeating: 0, count: nChroma)

        // Major third circle (4 semitones)
        var majorSin = [Float](repeating: 0, count: nChroma)
        var majorCos = [Float](repeating: 0, count: nChroma)

        for n in 0..<nChroma {
            let nf = Float(n)

            // Fifth: 7 semitones per step around circle of fifths
            let fifthAngle = Float.pi * nf * 7.0 / 6.0
            fifthSin[n] = sin(fifthAngle)
            fifthCos[n] = cos(fifthAngle)

            // Minor third: 3 semitones (4 minor thirds = octave)
            let minorAngle = Float.pi * nf * 3.0 / 2.0
            minorSin[n] = sin(minorAngle)
            minorCos[n] = cos(minorAngle)

            // Major third: 4 semitones (3 major thirds = octave)
            let majorAngle = Float.pi * nf * 2.0 / 3.0
            majorSin[n] = sin(majorAngle)
            majorCos[n] = cos(majorAngle)
        }

        matrix.append(fifthSin)
        matrix.append(fifthCos)
        matrix.append(minorSin)
        matrix.append(minorCos)
        matrix.append(majorSin)
        matrix.append(majorCos)

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
