import Accelerate
import Foundation

/// Weighting curve type for perceptual frequency weighting.
public enum WeightingCurve: Sendable {
    /// A-weighting (IEC 61672-1) - approximates human hearing at moderate levels.
    case aWeighting
    /// B-weighting - intermediate between A and C.
    case bWeighting
    /// C-weighting - flatter curve for higher SPL.
    case cWeighting
    /// D-weighting - for aircraft noise.
    case dWeighting
    /// ITU-R 468 - broadcast audio measurement.
    case itu468
    /// Z-weighting (flat) - no frequency weighting.
    case zWeighting
}

/// Perceptual frequency weighting for spectrograms.
///
/// Applies standardized frequency weighting curves to power or magnitude
/// spectrograms, converting them to perceptually-weighted decibel scales.
///
/// ## Example Usage
///
/// ```swift
/// // Apply A-weighting to power spectrogram
/// let weightedDB = PerceptualWeighting.weight(powerSpec, nFFT: 2048, sampleRate: 22050)
///
/// // Use C-weighting for loud signals
/// let cWeightedDB = PerceptualWeighting.weight(
///     powerSpec,
///     nFFT: 2048,
///     sampleRate: 22050,
///     curve: .cWeighting
/// )
///
/// // Get just the weighting curve
/// let curve = PerceptualWeighting.weightingCurve(.aWeighting, nFFT: 2048, sampleRate: 22050)
/// ```
public struct PerceptualWeighting {
    private init() {}

    // IEC 61672-1 reference frequencies
    private static let f1: Float = 20.598997
    private static let f2: Float = 107.65265
    private static let f3: Float = 737.86223
    private static let f4: Float = 12194.217

    /// Apply perceptual weighting to a power spectrogram.
    ///
    /// - Parameters:
    ///   - powerSpec: Power spectrogram of shape (nFreqs, nFrames).
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - sampleRate: Sample rate in Hz.
    ///   - curve: Weighting curve to apply (default: A-weighting).
    ///   - refPower: Reference power for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: Weighted spectrogram in dB of shape (nFreqs, nFrames).
    public static func weight(
        _ powerSpec: [[Float]],
        nFFT: Int,
        sampleRate: Float,
        curve: WeightingCurve = .aWeighting,
        refPower: Float = 1.0,
        minDB: Float = -80.0
    ) -> [[Float]] {
        guard !powerSpec.isEmpty, !powerSpec[0].isEmpty else { return [] }

        let nFreqs = powerSpec.count
        let nFrames = powerSpec[0].count
        let weights = weightingCurve(curve, nFFT: nFFT, sampleRate: sampleRate)

        guard weights.count == nFreqs else { return powerSpec }

        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let epsilon: Float = 1e-10

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                // Convert power to dB
                let power = max(powerSpec[freqIdx][frameIdx], epsilon)
                var db = 10.0 * log10(power / refPower)

                // Apply weighting
                db += weights[freqIdx]

                // Clip to minimum
                row[frameIdx] = max(db, minDB)
            }

            result.append(row)
        }

        return result
    }

    /// Apply perceptual weighting to a magnitude spectrogram.
    ///
    /// - Parameters:
    ///   - magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - nFFT: FFT size used to compute the spectrogram.
    ///   - sampleRate: Sample rate in Hz.
    ///   - curve: Weighting curve to apply (default: A-weighting).
    ///   - refAmplitude: Reference amplitude for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: Weighted spectrogram in dB of shape (nFreqs, nFrames).
    public static func weightMagnitude(
        _ magnitudeSpec: [[Float]],
        nFFT: Int,
        sampleRate: Float,
        curve: WeightingCurve = .aWeighting,
        refAmplitude: Float = 1.0,
        minDB: Float = -80.0
    ) -> [[Float]] {
        guard !magnitudeSpec.isEmpty, !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count
        let weights = weightingCurve(curve, nFFT: nFFT, sampleRate: sampleRate)

        guard weights.count == nFreqs else { return magnitudeSpec }

        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let epsilon: Float = 1e-10

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                // Convert magnitude to dB (20 * log10)
                let mag = max(magnitudeSpec[freqIdx][frameIdx], epsilon)
                var db = 20.0 * log10(mag / refAmplitude)

                // Apply weighting
                db += weights[freqIdx]

                // Clip to minimum
                row[frameIdx] = max(db, minDB)
            }

            result.append(row)
        }

        return result
    }

    /// Compute weighting curve for STFT frequency bins.
    ///
    /// - Parameters:
    ///   - curve: Weighting curve type.
    ///   - nFFT: FFT size.
    ///   - sampleRate: Sample rate in Hz.
    /// - Returns: Weighting values in dB for each frequency bin.
    public static func weightingCurve(
        _ curve: WeightingCurve,
        nFFT: Int,
        sampleRate: Float
    ) -> [Float] {
        let nFreqs = nFFT / 2 + 1
        let binWidth = sampleRate / Float(nFFT)

        var frequencies = [Float](repeating: 0, count: nFreqs)
        for i in 0..<nFreqs {
            frequencies[i] = Float(i) * binWidth
        }

        return weightingCurve(curve, frequencies: frequencies)
    }

    /// Compute weighting curve for arbitrary frequencies.
    ///
    /// - Parameters:
    ///   - curve: Weighting curve type.
    ///   - frequencies: Array of frequencies in Hz.
    /// - Returns: Weighting values in dB.
    public static func weightingCurve(
        _ curve: WeightingCurve,
        frequencies: [Float]
    ) -> [Float] {
        switch curve {
        case .aWeighting:
            return frequencies.map { aWeight($0) }
        case .bWeighting:
            return frequencies.map { bWeight($0) }
        case .cWeighting:
            return frequencies.map { cWeight($0) }
        case .dWeighting:
            return frequencies.map { dWeight($0) }
        case .itu468:
            return frequencies.map { itu468Weight($0) }
        case .zWeighting:
            return [Float](repeating: 0, count: frequencies.count)
        }
    }

    // MARK: - Weighting Implementations

    /// A-weighting per IEC 61672-1
    private static func aWeight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        let fSq = f * f
        let numerator = f4 * f4 * fSq * fSq
        let denom1 = fSq + f1 * f1
        let denom2Sq = (fSq + f2 * f2) * (fSq + f3 * f3)
        let denom2 = sqrt(denom2Sq)
        let denom3 = fSq + f4 * f4

        let ra = numerator / (denom1 * denom2 * denom3)
        return 20.0 * log10(ra) + 1.9997
    }

    /// B-weighting per IEC 61672-1
    private static func bWeight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        let fSq = f * f
        let numerator = f4 * f4 * fSq * f
        let denom1 = fSq + f1 * f1
        let denom2 = sqrt(fSq + f2 * f2)
        let denom3 = fSq + f4 * f4

        let rb = numerator / (denom1 * denom2 * denom3)
        return 20.0 * log10(rb) + 0.17
    }

    /// C-weighting per IEC 61672-1
    private static func cWeight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        let fSq = f * f
        let numerator = f4 * f4 * fSq
        let denom1 = fSq + f1 * f1
        let denom3 = fSq + f4 * f4

        let rc = numerator / (denom1 * denom3)
        return 20.0 * log10(rc) + 0.0619
    }

    /// D-weighting (aircraft noise)
    private static func dWeight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        let fSq = f * f

        // D-weighting uses different constants
        let h = ((1037918.48 - fSq) * (1037918.48 - fSq) + 1080768.16 * fSq) /
                ((9837328.0 - fSq) * (9837328.0 - fSq) + 11723776.0 * fSq)
        let numerator = fSq / ((fSq + 79919.29) * (fSq + 1345600.0))
        let rd = (numerator * h).squareRoot() * 1000

        return 20.0 * log10(rd)
    }

    /// ITU-R 468 noise weighting
    private static func itu468Weight(_ f: Float) -> Float {
        guard f > 0 else { return -Float.infinity }

        // Simplified ITU-R 468 approximation
        let fSq = f * f
        let f4th = fSq * fSq

        // Polynomial approximation
        let h1 = -4.737338981378384e-24 * f4th * fSq +
                  2.043828333606125e-15 * f4th -
                  1.363894795463638e-7 * fSq + 1

        let h2 = 1.306612257412824e-19 * f4th * f -
                 2.118150887518656e-11 * fSq * f +
                 5.559488023498642e-4 * f

        let r468 = 1.246332637532143e-4 * f / sqrt(h1 * h1 + h2 * h2)

        return 20.0 * log10(r468) + 18.2  // Normalize to 0 dB at 2 kHz
    }
}
