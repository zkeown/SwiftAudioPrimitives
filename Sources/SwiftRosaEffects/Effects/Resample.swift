import SwiftRosaCore
import Accelerate
import Foundation

/// Quality level for resampling operations.
public enum ResampleQuality: Sendable {
    /// Fast resampling with 8-tap FIR filter.
    case low

    /// Balanced quality/speed with 16-tap FIR filter.
    case medium

    /// High quality with 32-tap FIR filter.
    case high

    /// Very high quality with 64-tap FIR filter.
    case veryHigh

    /// Filter length for this quality level.
    var filterLength: Int {
        switch self {
        case .low: return 8
        case .medium: return 16
        case .high: return 32
        case .veryHigh: return 64
        }
    }

    /// Kaiser window beta for this quality level.
    var kaiserBeta: Float {
        switch self {
        case .low: return 5.0
        case .medium: return 6.0
        case .high: return 8.0
        case .veryHigh: return 10.0
        }
    }
}

/// Audio signal resampling using polyphase FIR filtering.
///
/// This implementation uses windowed-sinc interpolation with antialiasing
/// to convert audio signals between different sample rates while preserving
/// frequency content below the Nyquist frequency of the lower sample rate.
///
/// ## Example Usage
///
/// ```swift
/// // Resample from 48kHz to 44.1kHz
/// let resampled = await Resample.resample(
///     audio,
///     fromSampleRate: 48000,
///     toSampleRate: 44100,
///     quality: .high
/// )
///
/// // Downsample for model that expects 22.05kHz
/// let downsampled = await Resample.resample(
///     audio,
///     fromSampleRate: 44100,
///     toSampleRate: 22050
/// )
/// ```
public struct Resample: Sendable {
    private init() {}

    // MARK: - Main Resampling Function

    /// Resample an audio signal to a different sample rate.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromSampleRate: Original sample rate in Hz.
    ///   - toSampleRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromSampleRate: Int,
        toSampleRate: Int,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard fromSampleRate != toSampleRate else { return signal }
        guard fromSampleRate > 0 && toSampleRate > 0 else { return signal }

        // Find rational approximation for the resampling ratio
        let (upFactor, downFactor) = rationalApproximation(
            numerator: toSampleRate,
            denominator: fromSampleRate
        )

        // Design antialiasing filter
        // Cutoff should be at the Nyquist of the lower sample rate
        // Normalized to the higher rate (which is where filtering conceptually happens)
        let minRate = Float(min(fromSampleRate, toSampleRate))
        let effectiveRate = Float(fromSampleRate * upFactor)  // Rate after upsampling
        let cutoff = (minRate / 2.0) / (effectiveRate / 2.0)  // Normalized cutoff (0-0.5 range)
        let filterCoeffs = designLowpassFilter(
            cutoff: cutoff * 0.9,  // Slight rolloff to avoid aliasing at band edge
            length: quality.filterLength * upFactor,
            beta: quality.kaiserBeta
        )

        // Perform polyphase resampling
        return polyphaseResample(
            signal: signal,
            upFactor: upFactor,
            downFactor: downFactor,
            filter: filterCoeffs
        )
    }

    /// Resample using simple linear interpolation (fast but lower quality).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromSampleRate: Original sample rate.
    ///   - toSampleRate: Target sample rate.
    /// - Returns: Resampled audio signal.
    public static func resampleLinear(
        _ signal: [Float],
        fromSampleRate: Int,
        toSampleRate: Int
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard fromSampleRate != toSampleRate else { return signal }

        let ratio = Float(fromSampleRate) / Float(toSampleRate)
        let outputLength = Int(Float(signal.count) / ratio)

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcPos = Float(i) * ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)

            if srcIdx + 1 < signal.count {
                output[i] = signal[srcIdx] * (1 - frac) + signal[srcIdx + 1] * frac
            } else if srcIdx < signal.count {
                output[i] = signal[srcIdx]
            }
        }

        return output
    }

    // MARK: - Filter Design

    /// Design a lowpass FIR filter using windowed-sinc method.
    ///
    /// - Parameters:
    ///   - cutoff: Normalized cutoff frequency (0 to 0.5).
    ///   - length: Filter length (should be odd for symmetric filter).
    ///   - beta: Kaiser window beta parameter.
    /// - Returns: FIR filter coefficients.
    internal static func designLowpassFilter(
        cutoff: Float,
        length: Int,
        beta: Float = 6.0
    ) -> [Float] {
        let n = length | 1  // Ensure odd length
        let center = n / 2

        var filter = [Float](repeating: 0, count: n)
        let twoPiCutoff = 2.0 * Float.pi * cutoff

        // Generate sinc function
        for i in 0..<n {
            let x = Float(i - center)
            if x == 0 {
                filter[i] = 2.0 * cutoff
            } else {
                filter[i] = sin(twoPiCutoff * x) / (Float.pi * x)
            }
        }

        // Apply Kaiser window
        let window = kaiserWindow(length: n, beta: beta)
        vDSP_vmul(filter, 1, window, 1, &filter, 1, vDSP_Length(n))

        // Normalize to unit sum
        var sum: Float = 0
        vDSP_sve(filter, 1, &sum, vDSP_Length(n))
        if sum != 0 {
            var invSum = 1.0 / sum
            vDSP_vsmul(filter, 1, &invSum, &filter, 1, vDSP_Length(n))
        }

        return filter
    }

    /// Generate a Kaiser window.
    ///
    /// - Parameters:
    ///   - length: Window length.
    ///   - beta: Shape parameter (larger = narrower main lobe, higher sidelobe attenuation).
    /// - Returns: Kaiser window coefficients.
    private static func kaiserWindow(length: Int, beta: Float) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let center = Float(length - 1) / 2.0
        let betaI0 = besselI0(beta)

        for i in 0..<length {
            let x = (Float(i) - center) / center
            let arg = beta * sqrt(max(0, 1 - x * x))
            window[i] = besselI0(arg) / betaI0
        }

        return window
    }

    /// Modified Bessel function of the first kind, order 0.
    private static func besselI0(_ x: Float) -> Float {
        var sum: Float = 1.0
        var term: Float = 1.0
        let x2 = x * x / 4.0

        for k in 1...25 {
            term *= x2 / Float(k * k)
            sum += term
            if term < 1e-10 { break }
        }

        return sum
    }

    // MARK: - Polyphase Resampling

    /// Perform polyphase resampling using fractional delay approach.
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - upFactor: Upsampling factor L.
    ///   - downFactor: Downsampling factor M.
    ///   - filter: Lowpass filter coefficients.
    /// - Returns: Resampled signal.
    private static func polyphaseResample(
        signal: [Float],
        upFactor: Int,
        downFactor: Int,
        filter: [Float]
    ) -> [Float] {
        let inputLength = signal.count
        let outputLength = (inputLength * upFactor) / downFactor

        guard outputLength > 0 && inputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)
        let filterLen = filter.count
        let filterHalf = filterLen / 2

        // For each output sample
        for n in 0..<outputLength {
            // Position in the upsampled (conceptual) signal
            let m = n * downFactor

            var sum: Float = 0.0

            // Convolution: output[n] = sum_k filter[k] * upsampled[m - k + filterHalf]
            // The upsampled signal is: upsampled[j] = signal[j/L] if j%L==0, else 0
            for k in 0..<filterLen {
                // Position in upsampled domain relative to filter center
                let j = m - k + filterHalf

                // upsampled[j] is non-zero only when j % L == 0
                if j >= 0 && j % upFactor == 0 {
                    let signalIdx = j / upFactor
                    if signalIdx >= 0 && signalIdx < inputLength {
                        sum += signal[signalIdx] * filter[k]
                    }
                }
            }

            // Scale by upFactor to compensate for zero-insertion
            output[n] = sum * Float(upFactor)
        }

        return output
    }

    // MARK: - Utility Functions

    /// Find a rational approximation to a fraction.
    ///
    /// - Parameters:
    ///   - numerator: Target numerator.
    ///   - denominator: Target denominator.
    /// - Returns: Simplified (numerator, denominator) pair.
    private static func rationalApproximation(numerator: Int, denominator: Int) -> (Int, Int) {
        let gcdVal = gcd(numerator, denominator)
        return (numerator / gcdVal, denominator / gcdVal)
    }

    /// Greatest common divisor using Euclidean algorithm.
    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var a = a
        var b = b
        while b != 0 {
            let temp = b
            b = a % b
            a = temp
        }
        return a
    }
}

// MARK: - Convenience Extensions

extension Resample {
    /// Convenience method with alternative parameter names (Int version).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromRate: Original sample rate in Hz.
    ///   - toRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromRate: Int,
        toRate: Int,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        return await resample(signal, fromSampleRate: fromRate, toSampleRate: toRate, quality: quality)
    }

    /// Convenience method with alternative parameter names (Float version).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromRate: Original sample rate in Hz.
    ///   - toRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromRate: Float,
        toRate: Float,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        return await resample(signal, fromSampleRate: Int(fromRate), toSampleRate: Int(toRate), quality: quality)
    }

    /// Common resampling presets.
    public enum Preset: Sendable {
        /// CD quality to Banquet model (44100 → 44100, no change)
        case cdToBanquet

        /// DVD audio to Banquet model (48000 → 44100)
        case dvdToBanquet

        /// High-res audio to Banquet model (96000 → 44100)
        case hiresToBanquet

        /// CD quality to common ML rate (44100 → 22050)
        case cdToML

        /// CD quality to speech rate (44100 → 16000)
        case cdToSpeech

        /// Source and target sample rates.
        var rates: (from: Int, to: Int) {
            switch self {
            case .cdToBanquet: return (44100, 44100)
            case .dvdToBanquet: return (48000, 44100)
            case .hiresToBanquet: return (96000, 44100)
            case .cdToML: return (44100, 22050)
            case .cdToSpeech: return (44100, 16000)
            }
        }
    }

    /// Resample using a preset configuration.
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - preset: Resampling preset.
    ///   - actualInputRate: Actual input sample rate (must match preset.from).
    ///   - quality: Resampling quality.
    /// - Returns: Resampled signal.
    public static func resample(
        _ signal: [Float],
        preset: Preset,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        let rates = preset.rates
        return await resample(
            signal,
            fromSampleRate: rates.from,
            toSampleRate: rates.to,
            quality: quality
        )
    }
}
