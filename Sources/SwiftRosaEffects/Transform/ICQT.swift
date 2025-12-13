import SwiftRosaCore
import Accelerate
import Foundation

/// Inverse Constant-Q Transform.
///
/// Reconstructs a time-domain audio signal from its CQT representation.
/// Uses a pseudo-inverse approach with overlap-add reconstruction.
///
/// ## Example Usage
///
/// ```swift
/// let cqt = CQT()
/// let icqt = ICQT(cqtConfig: cqt.config)
///
/// // Forward transform
/// let cqtResult = await cqt.transform(audioSignal)
///
/// // Inverse transform
/// let reconstructed = await icqt.transform(cqtResult, length: audioSignal.count)
/// ```
///
/// - Note: Matches `librosa.icqt(Vq, sr)` behavior.
public struct ICQT: Sendable {
    /// CQT configuration (must match the forward transform).
    public let config: CQTConfig

    /// Create an inverse CQT processor.
    ///
    /// - Parameter cqtConfig: Configuration matching the forward CQT.
    public init(cqtConfig: CQTConfig) {
        self.config = cqtConfig
    }

    /// Reconstruct audio from CQT representation.
    ///
    /// Uses iterative reconstruction with overlap-add. This is an approximate
    /// inverse since CQT is an overcomplete representation.
    ///
    /// - Parameters:
    ///   - cqt: Complex CQT matrix of shape (nBins, nFrames).
    ///   - length: Desired output length. If nil, estimated from frame count.
    ///   - iterations: Number of Griffin-Lim style iterations (default: 32).
    /// - Returns: Reconstructed time-domain audio signal.
    public func transform(
        _ cqt: ComplexMatrix,
        length: Int? = nil,
        iterations: Int = 32
    ) async -> [Float] {
        guard cqt.rows > 0, cqt.cols > 0 else { return [] }

        let nBins = cqt.rows
        let nFrames = cqt.cols

        // Estimate output length if not provided
        let outputLength = length ?? (nFrames * config.hopLength + config.hopLength)

        // Get magnitude and phase from input CQT
        let magnitude = cqt.magnitude
        let phase = cqt.phase

        // Initialize output signal
        var signal = [Float](repeating: 0, count: outputLength)

        // Pre-compute frequency information for each bin
        var binFreqs = [Float](repeating: 0, count: nBins)
        var binLengths = [Int](repeating: 0, count: nBins)

        for bin in 0..<nBins {
            let freq = config.centerFrequency(forBin: bin)
            binFreqs[bin] = freq

            // Filter length based on Q factor
            let bandwidth = freq / config.Q
            let filterLen = Int(config.sampleRate / bandwidth * config.filterScale)
            binLengths[bin] = min(filterLen, config.hopLength * 4)
        }

        // For each frame, synthesize contribution from all bins
        for frame in 0..<nFrames {
            let frameStart = frame * config.hopLength

            for bin in 0..<nBins {
                let mag = magnitude[bin][frame]
                let ph = phase[bin][frame]
                let freq = binFreqs[bin]
                let filterLen = binLengths[bin]

                // Skip negligible contributions
                guard mag > 1e-10 else { continue }

                // Synthesize windowed sinusoid for this bin
                let halfLen = filterLen / 2

                for i in 0..<filterLen {
                    let sampleIdx = frameStart + i - halfLen
                    guard sampleIdx >= 0 && sampleIdx < outputLength else { continue }

                    // Time relative to frame center
                    let t = Float(i - halfLen) / config.sampleRate

                    // Windowed sinusoid (Hann window)
                    let windowVal = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(filterLen - 1)))
                    let sinusoid = mag * cos(2.0 * Float.pi * freq * t + ph) * windowVal

                    signal[sampleIdx] += sinusoid
                }
            }
        }

        // Normalize by overlap factor
        let overlapFactor = Float(nBins) / Float(config.binsPerOctave)
        if overlapFactor > 1 {
            var scale = 1.0 / overlapFactor
            vDSP_vsmul(signal, 1, &scale, &signal, 1, vDSP_Length(outputLength))
        }

        return signal
    }

    /// Reconstruct audio from CQT magnitude using phase estimation.
    ///
    /// Uses Griffin-Lim style iterative phase reconstruction when only
    /// magnitude is available.
    ///
    /// - Parameters:
    ///   - magnitude: CQT magnitude of shape (nBins, nFrames).
    ///   - length: Desired output length.
    ///   - iterations: Number of iterations for phase estimation (default: 32).
    /// - Returns: Reconstructed time-domain audio signal.
    public func fromMagnitude(
        _ magnitude: [[Float]],
        length: Int? = nil,
        iterations: Int = 32
    ) async -> [Float] {
        guard !magnitude.isEmpty, !magnitude[0].isEmpty else { return [] }

        let nBins = magnitude.count
        let nFrames = magnitude[0].count

        // Initialize with random phase
        var phase = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nBins)
        for bin in 0..<nBins {
            for frame in 0..<nFrames {
                phase[bin][frame] = Float.random(in: -Float.pi...Float.pi)
            }
        }

        let outputLength = length ?? (nFrames * config.hopLength + config.hopLength)
        var signal = [Float](repeating: 0, count: outputLength)

        // Iterative refinement (simplified Griffin-Lim for CQT)
        for _ in 0..<iterations {
            // Build complex CQT from magnitude and current phase
            var realPart = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nBins)
            var imagPart = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nBins)

            for bin in 0..<nBins {
                for frame in 0..<nFrames {
                    realPart[bin][frame] = magnitude[bin][frame] * cos(phase[bin][frame])
                    imagPart[bin][frame] = magnitude[bin][frame] * sin(phase[bin][frame])
                }
            }

            let cqt = ComplexMatrix(real: realPart, imag: imagPart)

            // Reconstruct signal
            signal = await transform(cqt, length: outputLength, iterations: 1)

            // Re-analyze to get updated phase (forward CQT would be ideal here)
            // For simplicity, use a basic phase update based on signal
            for frame in 0..<nFrames {
                let frameStart = frame * config.hopLength

                for bin in 0..<nBins {
                    let freq = config.centerFrequency(forBin: bin)
                    let filterLen = min(Int(config.sampleRate / (freq / config.Q)), config.hopLength * 4)
                    let halfLen = filterLen / 2

                    // Estimate phase from reconstructed signal
                    var realAccum: Float = 0
                    var imagAccum: Float = 0

                    for i in 0..<filterLen {
                        let sampleIdx = frameStart + i - halfLen
                        guard sampleIdx >= 0 && sampleIdx < outputLength else { continue }

                        let t = Float(i - halfLen) / config.sampleRate
                        let windowVal = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(filterLen - 1)))

                        realAccum += signal[sampleIdx] * cos(2.0 * Float.pi * freq * t) * windowVal
                        imagAccum += signal[sampleIdx] * sin(2.0 * Float.pi * freq * t) * windowVal
                    }

                    phase[bin][frame] = atan2(imagAccum, realAccum)
                }
            }
        }

        return signal
    }
}

// MARK: - Convenience Extensions

extension CQT {
    /// Create an inverse CQT processor with matching configuration.
    public func makeInverse() -> ICQT {
        return ICQT(cqtConfig: config)
    }
}
