import SwiftRosaCore
import SwiftRosaEffects
import Accelerate
import Foundation

/// Noise types for audio augmentation.
public enum NoiseType: Sendable {
    /// White noise (flat spectrum).
    case white
    /// Pink noise (1/f spectrum).
    case pink
    /// Brown noise (1/f^2 spectrum).
    case brown
    /// Custom noise samples.
    case custom([Float])
}

/// Audio augmentation utilities.
///
/// Provides various audio augmentation techniques commonly used for
/// training robust audio/speech recognition models.
///
/// ## Example Usage
///
/// ```swift
/// // Time-stretch without pitch change
/// let stretched = await AudioAugmentation.timeStretch(audio, rate: 1.2)
///
/// // Pitch-shift without duration change
/// let shifted = await AudioAugmentation.pitchShift(audio, semitones: 2)
///
/// // Add noise at specific SNR
/// let noisy = AudioAugmentation.addNoise(audio, noise: .white, snrDb: 10)
/// ```
public struct AudioAugmentation: Sendable {
    private init() {}

    // MARK: - Time-Domain Augmentations

    /// Time-stretch audio without changing pitch.
    ///
    /// Uses phase vocoder technique to stretch or compress audio duration.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - rate: Stretch rate (< 1 = slower, > 1 = faster).
    ///   - stftConfig: STFT configuration (default: 2048, 512 hop).
    /// - Returns: Time-stretched audio.
    public static func timeStretch(
        _ signal: [Float],
        rate: Float,
        stftConfig: STFTConfig = STFTConfig(nFFT: 2048, hopLength: 512)
    ) async -> [Float] {
        guard rate > 0 && rate != 1.0 else { return signal }
        guard !signal.isEmpty else { return [] }

        let stft = STFT(config: stftConfig)
        let istft = ISTFT(config: stftConfig)

        // Compute STFT
        let spectrogram = await stft.transform(signal)

        // Phase vocoder stretching
        let stretchedSpec = phaseVocoderStretch(spectrogram, rate: rate, hopLength: stftConfig.hopLength)

        // Reconstruct
        return await istft.transform(stretchedSpec)
    }

    /// Pitch-shift audio without changing duration.
    ///
    /// Combines time-stretching with resampling.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - semitones: Pitch shift in semitones (positive = higher, negative = lower).
    ///   - sampleRate: Sample rate in Hz.
    ///   - stftConfig: STFT configuration.
    /// - Returns: Pitch-shifted audio.
    public static func pitchShift(
        _ signal: [Float],
        semitones: Float,
        sampleRate: Float = 22050,
        stftConfig: STFTConfig = STFTConfig(nFFT: 2048, hopLength: 512)
    ) async -> [Float] {
        guard semitones != 0 else { return signal }
        guard !signal.isEmpty else { return [] }

        // Pitch shift ratio
        let ratio = pow(2.0, semitones / 12.0)

        // Time-stretch by inverse ratio
        let stretched = await timeStretch(signal, rate: 1.0 / ratio, stftConfig: stftConfig)

        // Resample to original length
        let resampled = await Resample.resample(
            stretched,
            fromRate: sampleRate,
            toRate: sampleRate * ratio,
            quality: .high
        )

        // Trim or pad to original length
        if resampled.count > signal.count {
            return Array(resampled.prefix(signal.count))
        } else if resampled.count < signal.count {
            var padded = resampled
            padded.append(contentsOf: [Float](repeating: 0, count: signal.count - resampled.count))
            return padded
        }

        return resampled
    }

    /// Add noise at specified SNR.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - noise: Type of noise to add.
    ///   - snrDb: Target signal-to-noise ratio in dB.
    /// - Returns: Signal with added noise.
    public static func addNoise(
        _ signal: [Float],
        noise: NoiseType,
        snrDb: Float
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }

        // Generate noise
        var noiseSignal: [Float]
        switch noise {
        case .white:
            noiseSignal = generateWhiteNoise(length: signal.count)
        case .pink:
            noiseSignal = generatePinkNoise(length: signal.count)
        case .brown:
            noiseSignal = generateBrownNoise(length: signal.count)
        case .custom(let customNoise):
            // Loop custom noise to match signal length
            noiseSignal = [Float](repeating: 0, count: signal.count)
            for i in 0..<signal.count {
                noiseSignal[i] = customNoise[i % customNoise.count]
            }
        }

        // Compute signal energy
        var signalEnergy: Float = 0
        vDSP_svesq(signal, 1, &signalEnergy, vDSP_Length(signal.count))

        // Compute noise energy
        var noiseEnergy: Float = 0
        vDSP_svesq(noiseSignal, 1, &noiseEnergy, vDSP_Length(noiseSignal.count))

        // Compute scaling factor for target SNR
        // SNR_dB = 10 * log10(signal_energy / noise_energy)
        // noise_energy_scaled = signal_energy / 10^(SNR_dB/10)
        let targetNoiseEnergy = signalEnergy / pow(10.0, snrDb / 10.0)
        let scale = noiseEnergy > 0 ? sqrt(targetNoiseEnergy / noiseEnergy) : 0

        // Add scaled noise to signal
        var result = [Float](repeating: 0, count: signal.count)
        for i in 0..<signal.count {
            result[i] = signal[i] + noiseSignal[i] * scale
        }

        return result
    }

    /// Apply random gain adjustment.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - minDb: Minimum gain in dB.
    ///   - maxDb: Maximum gain in dB.
    /// - Returns: Gain-adjusted signal.
    public static func randomGain(
        _ signal: [Float],
        minDb: Float = -6,
        maxDb: Float = 6
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }

        let gainDb = Float.random(in: minDb...maxDb)
        let gainLinear = pow(10.0, gainDb / 20.0)

        var result = [Float](repeating: 0, count: signal.count)
        var gain = gainLinear
        vDSP_vsmul(signal, 1, &gain, &result, 1, vDSP_Length(signal.count))

        return result
    }

    /// Apply time shift with optional wrap-around.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - shiftSamples: Number of samples to shift (positive = delay).
    ///   - wrap: Wrap shifted samples (true) or zero-pad (false).
    /// - Returns: Time-shifted signal.
    public static func timeShift(
        _ signal: [Float],
        shiftSamples: Int,
        wrap: Bool = false
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }

        let n = signal.count
        let shift = ((shiftSamples % n) + n) % n  // Normalize to positive

        var result = [Float](repeating: 0, count: n)

        if wrap {
            for i in 0..<n {
                result[(i + shift) % n] = signal[i]
            }
        } else {
            if shift > 0 {
                // Delay: copy from beginning, zero at start
                for i in shift..<n {
                    result[i] = signal[i - shift]
                }
            } else {
                // Advance: copy to beginning, zero at end
                for i in 0..<(n + shift) {
                    result[i] = signal[i - shift]
                }
            }
        }

        return result
    }

    /// Trim or pad audio to target length.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - targetLength: Target length in samples.
    ///   - padMode: Padding mode if signal is shorter.
    /// - Returns: Audio of exactly targetLength samples.
    public static func trimOrPad(
        _ signal: [Float],
        targetLength: Int,
        padMode: PadMode = .constant(0)
    ) -> [Float] {
        if signal.count == targetLength {
            return signal
        } else if signal.count > targetLength {
            // Trim from center
            let start = (signal.count - targetLength) / 2
            return Array(signal[start..<(start + targetLength)])
        } else {
            // Pad
            let padTotal = targetLength - signal.count
            let padLeft = padTotal / 2
            let padRight = padTotal - padLeft

            var result = [Float]()
            result.reserveCapacity(targetLength)

            switch padMode {
            case .constant(let value):
                result.append(contentsOf: [Float](repeating: value, count: padLeft))
                result.append(contentsOf: signal)
                result.append(contentsOf: [Float](repeating: value, count: padRight))

            case .reflect:
                for i in (1...padLeft).reversed() {
                    let idx = i % signal.count
                    result.append(signal[idx])
                }
                result.append(contentsOf: signal)
                for i in 1...padRight {
                    let idx = (signal.count - 1) - (i % signal.count)
                    result.append(signal[max(0, idx)])
                }

            case .edge:
                result.append(contentsOf: [Float](repeating: signal.first ?? 0, count: padLeft))
                result.append(contentsOf: signal)
                result.append(contentsOf: [Float](repeating: signal.last ?? 0, count: padRight))
            }

            return result
        }
    }

    // MARK: - Private Helpers

    private static func generateWhiteNoise(length: Int) -> [Float] {
        var noise = [Float](repeating: 0, count: length)
        for i in 0..<length {
            noise[i] = Float.random(in: -1...1)
        }
        return noise
    }

    private static func generatePinkNoise(length: Int) -> [Float] {
        // Pink noise using Paul Kellet's refined method
        var noise = [Float](repeating: 0, count: length)
        var b0: Float = 0, b1: Float = 0, b2: Float = 0
        var b3: Float = 0, b4: Float = 0, b5: Float = 0, b6: Float = 0

        for i in 0..<length {
            let white = Float.random(in: -1...1)

            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759
            b2 = 0.96900 * b2 + white * 0.1538520
            b3 = 0.86650 * b3 + white * 0.3104856
            b4 = 0.55000 * b4 + white * 0.5329522
            b5 = -0.7616 * b5 - white * 0.0168980

            noise[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) / 7.0
            b6 = white * 0.115926
        }

        return noise
    }

    private static func generateBrownNoise(length: Int) -> [Float] {
        // Brown noise (random walk)
        var noise = [Float](repeating: 0, count: length)
        var lastOut: Float = 0

        for i in 0..<length {
            let white = Float.random(in: -1...1)
            lastOut = (lastOut + 0.02 * white) / 1.02
            noise[i] = lastOut * 3.5  // Scale to reasonable range
        }

        return noise
    }

    /// Simple phase vocoder stretching.
    private static func phaseVocoderStretch(
        _ spectrogram: ComplexMatrix,
        rate: Float,
        hopLength: Int
    ) -> ComplexMatrix {
        guard rate > 0 else { return spectrogram }

        let nFreqs = spectrogram.rows
        let nFramesIn = spectrogram.cols
        let nFramesOut = Int(Float(nFramesIn) / rate)

        guard nFramesOut > 0 else {
            return ComplexMatrix(real: [], imag: [])
        }

        var realOut = [[Float]](repeating: [Float](repeating: 0, count: nFramesOut), count: nFreqs)
        var imagOut = [[Float]](repeating: [Float](repeating: 0, count: nFramesOut), count: nFreqs)

        // Phase accumulator for each frequency
        var phaseAccum = [Float](repeating: 0, count: nFreqs)

        // Expected phase advance per frame
        let hopRatio = Float(hopLength)
        var expectedPhaseAdvance = [Float](repeating: 0, count: nFreqs)
        for f in 0..<nFreqs {
            expectedPhaseAdvance[f] = 2.0 * Float.pi * Float(f) * hopRatio / Float((nFreqs - 1) * 2)
        }

        for tOut in 0..<nFramesOut {
            // Map output frame to input frame (linear interpolation)
            let tInFloat = Float(tOut) * rate
            let tIn0 = Int(tInFloat)
            let tIn1 = min(tIn0 + 1, nFramesIn - 1)
            let alpha = tInFloat - Float(tIn0)

            for f in 0..<nFreqs {
                // Interpolate magnitude
                let mag0 = sqrt(spectrogram.real[f][tIn0] * spectrogram.real[f][tIn0] +
                               spectrogram.imag[f][tIn0] * spectrogram.imag[f][tIn0])
                let mag1 = sqrt(spectrogram.real[f][tIn1] * spectrogram.real[f][tIn1] +
                               spectrogram.imag[f][tIn1] * spectrogram.imag[f][tIn1])
                let mag = mag0 * (1 - alpha) + mag1 * alpha

                // Get input phases
                let phase0 = atan2(spectrogram.imag[f][tIn0], spectrogram.real[f][tIn0])
                let phase1 = atan2(spectrogram.imag[f][tIn1], spectrogram.real[f][tIn1])

                // Compute phase difference
                var phaseDiff = phase1 - phase0 - expectedPhaseAdvance[f]

                // Wrap to [-pi, pi]
                while phaseDiff > Float.pi { phaseDiff -= 2 * Float.pi }
                while phaseDiff < -Float.pi { phaseDiff += 2 * Float.pi }

                // Accumulate phase
                if tOut == 0 {
                    phaseAccum[f] = phase0 * (1 - alpha) + phase1 * alpha
                } else {
                    phaseAccum[f] += expectedPhaseAdvance[f] + phaseDiff
                }

                // Convert back to complex
                realOut[f][tOut] = mag * cos(phaseAccum[f])
                imagOut[f][tOut] = mag * sin(phaseAccum[f])
            }
        }

        return ComplexMatrix(real: realOut, imag: imagOut)
    }
}
