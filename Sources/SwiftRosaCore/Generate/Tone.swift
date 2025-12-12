import Accelerate
import Foundation

/// Pure tone (sinusoid) generator.
///
/// Generates pure sine wave tones at specified frequencies.
///
/// ## Example Usage
///
/// ```swift
/// // Generate 1 second of 440 Hz (A4) at 22050 Hz sample rate
/// let tone = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 1.0)
///
/// // Generate by sample count
/// let tone = ToneGenerator.tone(frequency: 440, sampleRate: 22050, length: 22050)
///
/// // Generate with custom phase offset
/// let tone = ToneGenerator.tone(frequency: 440, sampleRate: 22050, duration: 1.0, phase: .pi / 2)
/// ```
public struct ToneGenerator {
    private init() {}

    /// Generate a pure sine tone.
    ///
    /// - Parameters:
    ///   - frequency: Tone frequency in Hz.
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - duration: Duration in seconds (optional, specify either duration or length).
    ///   - length: Number of samples (optional, specify either duration or length).
    ///   - phase: Initial phase in radians (default: 0).
    ///   - amplitude: Amplitude scaling (default: 1.0).
    /// - Returns: Generated tone samples.
    public static func tone(
        frequency: Float,
        sampleRate: Float = 22050,
        duration: Float? = nil,
        length: Int? = nil,
        phase: Float = 0,
        amplitude: Float = 1.0
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

        var output = [Float](repeating: 0, count: numSamples)

        // Generate phase ramp: 2 * pi * frequency * t + phase
        // where t = [0, 1, 2, ..., numSamples-1] / sampleRate
        let phaseIncrement = 2.0 * Float.pi * frequency / sampleRate

        // Use vDSP for efficient generation
        // First generate linear ramp [0, 1, 2, ..., numSamples-1]
        var ramp = [Float](repeating: 0, count: numSamples)
        var zero: Float = 0
        var one: Float = 1
        vDSP_vramp(&zero, &one, &ramp, 1, vDSP_Length(numSamples))

        // Scale by phase increment
        var scaledPhaseInc = phaseIncrement
        vDSP_vsmul(ramp, 1, &scaledPhaseInc, &ramp, 1, vDSP_Length(numSamples))

        // Add initial phase
        var phaseOffset = phase
        vDSP_vsadd(ramp, 1, &phaseOffset, &ramp, 1, vDSP_Length(numSamples))

        // Compute sine using vForce
        var n = Int32(numSamples)
        vvsinf(&output, ramp, &n)

        // Scale by amplitude
        if amplitude != 1.0 {
            var amp = amplitude
            vDSP_vsmul(output, 1, &amp, &output, 1, vDSP_Length(numSamples))
        }

        return output
    }

    /// Generate a multi-tone signal (sum of sinusoids).
    ///
    /// - Parameters:
    ///   - frequencies: Array of frequencies in Hz.
    ///   - amplitudes: Amplitudes for each frequency (optional, defaults to equal amplitude).
    ///   - phases: Initial phases for each frequency (optional, defaults to 0).
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - duration: Duration in seconds.
    /// - Returns: Generated multi-tone signal.
    public static func multiTone(
        frequencies: [Float],
        amplitudes: [Float]? = nil,
        phases: [Float]? = nil,
        sampleRate: Float = 22050,
        duration: Float
    ) -> [Float] {
        guard !frequencies.isEmpty else { return [] }

        let numSamples = Int(duration * sampleRate)
        guard numSamples > 0 else { return [] }

        // Default amplitudes: equal amplitude summing to 1
        let amps = amplitudes ?? [Float](repeating: 1.0 / Float(frequencies.count), count: frequencies.count)

        // Default phases: all zero
        let phs = phases ?? [Float](repeating: 0, count: frequencies.count)

        var output = [Float](repeating: 0, count: numSamples)

        for (i, freq) in frequencies.enumerated() {
            let amp = i < amps.count ? amps[i] : 1.0 / Float(frequencies.count)
            let ph = i < phs.count ? phs[i] : 0

            let component = tone(
                frequency: freq,
                sampleRate: sampleRate,
                length: numSamples,
                phase: ph,
                amplitude: amp
            )

            // Add to output
            vDSP_vadd(output, 1, component, 1, &output, 1, vDSP_Length(numSamples))
        }

        return output
    }
}
