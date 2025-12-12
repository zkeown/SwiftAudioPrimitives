import Accelerate
import Foundation

/// Chirp signal type.
public enum ChirpType: Sendable {
    /// Linear frequency sweep (constant rate of change).
    case linear
    /// Exponential/logarithmic frequency sweep (constant octave rate).
    case exponential
    /// Quadratic frequency sweep.
    case quadratic
    /// Hyperbolic frequency sweep.
    case hyperbolic
}

/// Frequency sweep (chirp) generator.
///
/// Generates chirp signals where frequency varies over time from fMin to fMax.
///
/// ## Example Usage
///
/// ```swift
/// // Generate linear chirp from 100 Hz to 1000 Hz over 1 second
/// let chirp = ChirpGenerator.chirp(fMin: 100, fMax: 1000, sampleRate: 22050, duration: 1.0)
///
/// // Generate exponential chirp (logarithmic frequency sweep)
/// let chirp = ChirpGenerator.chirp(
///     fMin: 100,
///     fMax: 10000,
///     sampleRate: 22050,
///     duration: 2.0,
///     type: .exponential
/// )
///
/// // Custom initial phase
/// let chirp = ChirpGenerator.chirp(fMin: 100, fMax: 1000, sampleRate: 22050, duration: 1.0, phase: .pi / 4)
/// ```
public struct ChirpGenerator {
    private init() {}

    /// Generate a frequency sweep (chirp) signal.
    ///
    /// - Parameters:
    ///   - fMin: Start frequency in Hz.
    ///   - fMax: End frequency in Hz.
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - duration: Duration in seconds (optional, specify either duration or length).
    ///   - length: Number of samples (optional, specify either duration or length).
    ///   - type: Chirp type/sweep method (default: exponential).
    ///   - phase: Initial phase in radians (default: 0).
    /// - Returns: Generated chirp signal.
    public static func chirp(
        fMin: Float,
        fMax: Float,
        sampleRate: Float = 22050,
        duration: Float? = nil,
        length: Int? = nil,
        type: ChirpType = .exponential,
        phase: Float = 0
    ) -> [Float] {
        // Determine number of samples
        let numSamples: Int
        if let len = length {
            numSamples = len
        } else if let dur = duration {
            numSamples = Int(dur * sampleRate)
        } else {
            // Default to 1 second
            numSamples = Int(sampleRate)
        }

        guard numSamples > 0 else { return [] }
        guard fMin > 0 && fMax > 0 else { return [] }

        let T = Float(numSamples) / sampleRate  // Total duration in seconds

        var output = [Float](repeating: 0, count: numSamples)

        switch type {
        case .linear:
            output = generateLinearChirp(
                fMin: fMin, fMax: fMax, sampleRate: sampleRate,
                numSamples: numSamples, T: T, phase: phase
            )

        case .exponential:
            output = generateExponentialChirp(
                fMin: fMin, fMax: fMax, sampleRate: sampleRate,
                numSamples: numSamples, T: T, phase: phase
            )

        case .quadratic:
            output = generateQuadraticChirp(
                fMin: fMin, fMax: fMax, sampleRate: sampleRate,
                numSamples: numSamples, T: T, phase: phase
            )

        case .hyperbolic:
            output = generateHyperbolicChirp(
                fMin: fMin, fMax: fMax, sampleRate: sampleRate,
                numSamples: numSamples, T: T, phase: phase
            )
        }

        return output
    }

    // MARK: - Private Implementation

    /// Linear chirp: f(t) = fMin + (fMax - fMin) * t / T
    /// Phase: φ(t) = 2π * (fMin * t + (fMax - fMin) * t² / (2T)) + phase
    private static func generateLinearChirp(
        fMin: Float, fMax: Float, sampleRate: Float,
        numSamples: Int, T: Float, phase: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: numSamples)
        let chirpRate = (fMax - fMin) / T

        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            let phi = 2.0 * Float.pi * (fMin * t + 0.5 * chirpRate * t * t) + phase
            output[i] = sin(phi)
        }

        return output
    }

    /// Exponential chirp: f(t) = fMin * (fMax/fMin)^(t/T)
    /// Phase: φ(t) = 2π * fMin * T * ((fMax/fMin)^(t/T) - 1) / ln(fMax/fMin) + phase
    private static func generateExponentialChirp(
        fMin: Float, fMax: Float, sampleRate: Float,
        numSamples: Int, T: Float, phase: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: numSamples)

        // Handle edge case where fMin == fMax (pure tone)
        if abs(fMax - fMin) < 1e-6 {
            return ToneGenerator.tone(
                frequency: fMin,
                sampleRate: sampleRate,
                length: numSamples,
                phase: phase
            )
        }

        let ratio = fMax / fMin
        let logRatio = log(ratio)

        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            let normalizedTime = t / T

            // Phase integral: 2π * fMin * T * (ratio^(t/T) - 1) / ln(ratio)
            let phi = 2.0 * Float.pi * fMin * T * (pow(ratio, normalizedTime) - 1) / logRatio + phase
            output[i] = sin(phi)
        }

        return output
    }

    /// Quadratic chirp: f(t) = fMin + (fMax - fMin) * (t/T)²
    private static func generateQuadraticChirp(
        fMin: Float, fMax: Float, sampleRate: Float,
        numSamples: Int, T: Float, phase: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: numSamples)
        let freqDiff = fMax - fMin

        for i in 0..<numSamples {
            let t = Float(i) / sampleRate

            // Phase integral of quadratic frequency
            // φ(t) = 2π * (fMin * t + freqDiff * t³ / (3T²)) + phase
            let phi = 2.0 * Float.pi * (fMin * t + freqDiff * t * t * t / (3 * T * T)) + phase
            output[i] = sin(phi)
        }

        return output
    }

    /// Hyperbolic chirp: f(t) = fMin * fMax / (fMax - (fMax - fMin) * t / T)
    private static func generateHyperbolicChirp(
        fMin: Float, fMax: Float, sampleRate: Float,
        numSamples: Int, T: Float, phase: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: numSamples)
        let freqDiff = fMax - fMin

        // Prevent division by zero
        guard abs(freqDiff) > 1e-6 else {
            return ToneGenerator.tone(
                frequency: fMin,
                sampleRate: sampleRate,
                length: numSamples,
                phase: phase
            )
        }

        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            let normalizedTime = t / T

            // f(t) = fMin * fMax / (fMax - freqDiff * t/T)
            let denominator = fMax - freqDiff * normalizedTime
            guard denominator > 0 else { continue }

            // Phase integral
            let phi = 2.0 * Float.pi * fMin * fMax * T / freqDiff * log(fMax / denominator) + phase
            output[i] = sin(phi)
        }

        return output
    }
}

// MARK: - Convenience Methods

extension ChirpGenerator {
    /// Generate a linear frequency sweep.
    public static func linearChirp(
        fMin: Float,
        fMax: Float,
        sampleRate: Float = 22050,
        duration: Float
    ) -> [Float] {
        return chirp(fMin: fMin, fMax: fMax, sampleRate: sampleRate, duration: duration, type: .linear)
    }

    /// Generate an exponential (logarithmic) frequency sweep.
    public static func exponentialChirp(
        fMin: Float,
        fMax: Float,
        sampleRate: Float = 22050,
        duration: Float
    ) -> [Float] {
        return chirp(fMin: fMin, fMax: fMax, sampleRate: sampleRate, duration: duration, type: .exponential)
    }
}
