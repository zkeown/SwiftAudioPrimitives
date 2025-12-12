import Accelerate
import Foundation

/// A-weighting frequency response calculator.
///
/// Computes the A-weighting curve per IEC 61672-1, which approximates
/// the frequency response of human hearing at moderate sound levels.
///
/// ## Example Usage
///
/// ```swift
/// // Get A-weighting curve for specific frequencies
/// let freqs: [Float] = [20, 100, 1000, 4000, 10000]
/// let weights = AWeighting.weights(frequencies: freqs)
///
/// // Get A-weighting for STFT frequency bins
/// let weights = AWeighting.weights(nFFT: 2048, sampleRate: 22050)
///
/// // Apply A-weighting to a power spectrogram
/// let weightedDB = AWeighting.apply(to: powerSpec, nFFT: 2048, sampleRate: 22050)
/// ```
///
/// ## Formula (IEC 61672-1)
///
/// ```
/// R_A(f) = 12194² × f⁴ / ((f² + 20.6²) × √((f² + 107.7²)(f² + 737.9²)) × (f² + 12194²))
/// A(f) = 20 × log10(R_A(f)) + 2.00  // Normalized to 0 dB at 1 kHz
/// ```
public struct AWeighting {
    private init() {}

    // IEC 61672-1 reference frequencies (Hz)
    private static let f1: Float = 20.598997
    private static let f2: Float = 107.65265
    private static let f3: Float = 737.86223
    private static let f4: Float = 12194.217

    // Normalization constant (makes A(1000) = 0 dB)
    private static let aWeightingRef: Float = 1.9997

    /// Compute A-weighting values for given frequencies.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: A-weighting values in dB.
    public static func weights(frequencies: [Float]) -> [Float] {
        guard !frequencies.isEmpty else { return [] }

        var weights = [Float](repeating: 0, count: frequencies.count)

        for (i, f) in frequencies.enumerated() {
            weights[i] = aWeight(f)
        }

        return weights
    }

    /// Compute A-weighting values for STFT frequency bins.
    ///
    /// - Parameters:
    ///   - nFFT: FFT size.
    ///   - sampleRate: Sample rate in Hz.
    /// - Returns: A-weighting values in dB for each frequency bin.
    public static func weights(nFFT: Int, sampleRate: Float) -> [Float] {
        let nFreqs = nFFT / 2 + 1
        let binWidth = sampleRate / Float(nFFT)

        var frequencies = [Float](repeating: 0, count: nFreqs)
        for i in 0..<nFreqs {
            frequencies[i] = Float(i) * binWidth
        }

        return weights(frequencies: frequencies)
    }

    /// Apply A-weighting to a power spectrogram.
    ///
    /// Converts power spectrogram to A-weighted decibels.
    ///
    /// - Parameters:
    ///   - powerSpec: Power spectrogram of shape (nFreqs, nFrames).
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - sampleRate: Sample rate in Hz.
    ///   - refPower: Reference power for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: A-weighted spectrogram in dB.
    public static func apply(
        to powerSpec: [[Float]],
        nFFT: Int,
        sampleRate: Float,
        refPower: Float = 1.0,
        minDB: Float = -80.0
    ) -> [[Float]] {
        guard !powerSpec.isEmpty, !powerSpec[0].isEmpty else { return [] }

        let nFreqs = powerSpec.count
        let nFrames = powerSpec[0].count
        let aWeights = weights(nFFT: nFFT, sampleRate: sampleRate)

        guard aWeights.count == nFreqs else { return powerSpec }

        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let epsilon: Float = 1e-10

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                // Convert power to dB
                let power = max(powerSpec[freqIdx][frameIdx], epsilon)
                var db = 10.0 * log10(power / refPower)

                // Apply A-weighting
                db += aWeights[freqIdx]

                // Clip to minimum
                row[frameIdx] = max(db, minDB)
            }

            result.append(row)
        }

        return result
    }

    /// Apply A-weighting to a magnitude spectrogram.
    ///
    /// - Parameters:
    ///   - magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - sampleRate: Sample rate in Hz.
    ///   - refAmplitude: Reference amplitude for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: A-weighted spectrogram in dB.
    public static func applyToMagnitude(
        _ magnitudeSpec: [[Float]],
        nFFT: Int,
        sampleRate: Float,
        refAmplitude: Float = 1.0,
        minDB: Float = -80.0
    ) -> [[Float]] {
        guard !magnitudeSpec.isEmpty, !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count
        let aWeights = weights(nFFT: nFFT, sampleRate: sampleRate)

        guard aWeights.count == nFreqs else { return magnitudeSpec }

        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let epsilon: Float = 1e-10

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                // Convert magnitude to dB (20 * log10 for amplitude)
                let mag = max(magnitudeSpec[freqIdx][frameIdx], epsilon)
                var db = 20.0 * log10(mag / refAmplitude)

                // Apply A-weighting
                db += aWeights[freqIdx]

                // Clip to minimum
                row[frameIdx] = max(db, minDB)
            }

            result.append(row)
        }

        return result
    }

    // MARK: - Private Implementation

    /// Compute A-weighting for a single frequency.
    private static func aWeight(_ f: Float) -> Float {
        // Handle edge cases
        guard f > 0 else { return -Float.infinity }

        // Simplified formula matching librosa implementation
        let fSq = f * f
        let numerator = f4 * f4 * fSq * fSq
        let denom1 = fSq + f1 * f1
        let denom2Sq = (fSq + f2 * f2) * (fSq + f3 * f3)
        let denom2 = sqrt(denom2Sq)
        let denom3 = fSq + f4 * f4

        let raSimple = numerator / (denom1 * denom2 * denom3)

        // Convert to dB and normalize
        let db = 20.0 * log10(raSimple) + aWeightingRef

        return db
    }
}

// MARK: - Other Weighting Curves

extension AWeighting {
    /// Compute C-weighting values for given frequencies.
    ///
    /// C-weighting is flatter than A-weighting and is used for
    /// higher sound pressure levels.
    ///
    /// - Parameter frequencies: Array of frequencies in Hz.
    /// - Returns: C-weighting values in dB.
    public static func cWeights(frequencies: [Float]) -> [Float] {
        guard !frequencies.isEmpty else { return [] }

        var weights = [Float](repeating: 0, count: frequencies.count)

        for (i, f) in frequencies.enumerated() {
            weights[i] = cWeight(f)
        }

        return weights
    }

    /// Compute C-weighting for a single frequency.
    private static func cWeight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        let fSq = f * f
        let numerator = f4 * f4 * fSq
        let denom1 = fSq + f1 * f1
        let denom3 = fSq + f4 * f4

        let rc = numerator / (denom1 * denom3)

        // C-weighting normalization constant
        let cWeightingRef: Float = 0.0619

        return 20.0 * log10(rc) + cWeightingRef
    }
}
