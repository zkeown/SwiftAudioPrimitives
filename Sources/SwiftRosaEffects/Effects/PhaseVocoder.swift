import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for phase vocoder.
public struct PhaseVocoderConfig: Sendable {
    /// FFT size.
    public let nFFT: Int

    /// Hop length.
    public let hopLength: Int

    /// Window type.
    public let windowType: WindowType

    /// Create phase vocoder configuration.
    ///
    /// - Parameters:
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop length (default: 512).
    ///   - windowType: Window type (default: hann).
    public init(
        nFFT: Int = 2048,
        hopLength: Int = 512,
        windowType: WindowType = .hann
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.windowType = windowType
    }
}

/// Phase vocoder for time-stretching and pitch-shifting.
///
/// The phase vocoder is a technique for modifying the time scale or pitch of
/// audio signals while preserving quality. It works by manipulating the phase
/// relationships between STFT frames.
///
/// ## Example Usage
///
/// ```swift
/// let vocoder = PhaseVocoder()
///
/// // Slow down by 20% (rate = 0.8)
/// let slowed = await vocoder.timeStretch(audio, rate: 0.8)
///
/// // Speed up by 50% (rate = 1.5)
/// let sped = await vocoder.timeStretch(audio, rate: 1.5)
///
/// // Shift pitch up by 2 semitones
/// let higher = await vocoder.pitchShift(audio, semitones: 2, sampleRate: 22050)
/// ```
public struct PhaseVocoder: Sendable {
    /// Configuration.
    public let config: PhaseVocoderConfig

    private let stft: STFT
    private let istft: ISTFT

    /// Create a phase vocoder.
    ///
    /// - Parameter config: Configuration.
    public init(config: PhaseVocoderConfig = PhaseVocoderConfig()) {
        self.config = config

        let stftConfig = STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.nFFT,
            windowType: config.windowType,
            center: true,
            padMode: .constant(0)  // Match librosa's default (zero padding)
        )

        self.stft = STFT(config: stftConfig)
        self.istft = ISTFT(config: stftConfig)
    }

    /// Time-stretch audio without changing pitch.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - rate: Stretch rate (< 1 = slower, > 1 = faster).
    /// - Returns: Time-stretched audio.
    public func timeStretch(
        _ signal: [Float],
        rate: Float
    ) async -> [Float] {
        guard rate > 0 && rate != 1.0 else { return signal }
        guard !signal.isEmpty else { return [] }

        // Compute STFT
        let spectrogram = await stft.transform(signal)

        // Stretch spectrogram
        let stretchedSpec = stretchSpectrogram(spectrogram, rate: rate)

        // Expected output length (matches librosa's calculation)
        let expectedLength = Int(round(Float(signal.count) / rate))

        // Reconstruct with expected length
        return await istft.transform(stretchedSpec, length: expectedLength)
    }

    /// Pitch-shift audio without changing duration.
    ///
    /// Combines time-stretching with resampling.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - semitones: Pitch shift in semitones.
    ///   - sampleRate: Sample rate in Hz.
    /// - Returns: Pitch-shifted audio.
    public func pitchShift(
        _ signal: [Float],
        semitones: Float,
        sampleRate: Float
    ) async -> [Float] {
        guard semitones != 0 else { return signal }
        guard !signal.isEmpty else { return [] }

        // librosa's algorithm:
        // rate = 2^(-n_steps/12)  (note the negative sign!)
        // time_stretch(rate) then resample from sr/rate to sr
        let rate = pow(2.0, -semitones / 12.0)

        // Time-stretch by rate
        let stretched = await timeStretch(signal, rate: rate)

        // Resample from (sampleRate / rate) to sampleRate
        // This matches librosa's: resample(stretched, orig_sr=sr/rate, target_sr=sr)
        let resampled = await Resample.resample(
            stretched,
            fromRate: sampleRate / rate,
            toRate: sampleRate,
            quality: .high
        )

        // Trim or pad to original length
        return trimOrPad(resampled, toLength: signal.count)
    }

    /// Time-stretch spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Input complex spectrogram.
    ///   - rate: Stretch rate.
    /// - Returns: Stretched complex spectrogram.
    public func stretchSpectrogram(
        _ spectrogram: ComplexMatrix,
        rate: Float
    ) -> ComplexMatrix {
        guard rate > 0 else { return spectrogram }

        let nFreqs = spectrogram.rows
        let nFramesIn = spectrogram.cols
        let nFramesOut = max(1, Int(round(Float(nFramesIn) / rate)))

        var realOut = [[Float]](repeating: [Float](repeating: 0, count: nFramesOut), count: nFreqs)
        var imagOut = [[Float]](repeating: [Float](repeating: 0, count: nFramesOut), count: nFreqs)

        // Phase accumulator for each frequency bin
        var phaseAccum = [Float](repeating: 0, count: nFreqs)

        // Expected phase advance per input frame
        let hopRatio = Float(config.hopLength) / Float(config.nFFT)
        var expectedPhaseAdvance = [Float](repeating: 0, count: nFreqs)
        for f in 0..<nFreqs {
            expectedPhaseAdvance[f] = 2.0 * Float.pi * Float(f) * hopRatio
        }

        // Get initial phases
        for f in 0..<nFreqs {
            phaseAccum[f] = atan2(spectrogram.imag[f][0], spectrogram.real[f][0])
        }

        for tOut in 0..<nFramesOut {
            // Map output frame to input frame position
            // librosa: time_steps = np.arange(0, D.shape[-1], rate)
            // So time_steps[0] = 0, time_steps[1] = rate, time_steps[2] = 2*rate, etc.
            let tInFloat = Float(tOut) * rate
            let tIn0 = min(Int(tInFloat), nFramesIn - 1)
            let tIn1 = min(tIn0 + 1, nFramesIn - 1)
            let alpha = tInFloat - Float(tIn0)

            for f in 0..<nFreqs {
                // Interpolate magnitude from current input frames
                let mag0 = sqrt(spectrogram.real[f][tIn0] * spectrogram.real[f][tIn0] +
                               spectrogram.imag[f][tIn0] * spectrogram.imag[f][tIn0])
                let mag1 = sqrt(spectrogram.real[f][tIn1] * spectrogram.real[f][tIn1] +
                               spectrogram.imag[f][tIn1] * spectrogram.imag[f][tIn1])
                let mag = mag0 * (1 - alpha) + mag1 * alpha

                // OUTPUT FIRST using current phaseAccum (before updating)
                // This matches librosa's: d_stretch[..., t] = phasor(phase_acc, mag=mag)
                realOut[f][tOut] = mag * cos(phaseAccum[f])
                imagOut[f][tOut] = mag * sin(phaseAccum[f])

                // THEN update phaseAccum for the next iteration
                // Compute phase deviation between current pair of input frames
                let phase0 = atan2(spectrogram.imag[f][tIn0], spectrogram.real[f][tIn0])
                let phase1 = atan2(spectrogram.imag[f][tIn1], spectrogram.real[f][tIn1])

                // Phase difference (unwrapped deviation from expected)
                var dphase = phase1 - phase0 - expectedPhaseAdvance[f]

                // Wrap to [-pi, pi]
                while dphase > Float.pi { dphase -= 2 * Float.pi }
                while dphase < -Float.pi { dphase += 2 * Float.pi }

                // Accumulate for next iteration
                // This matches librosa's: phase_acc += phi_advance + dphase
                phaseAccum[f] += expectedPhaseAdvance[f] + dphase
            }
        }

        return ComplexMatrix(real: realOut, imag: imagOut)
    }

    // MARK: - Private Helpers

    private func trimOrPad(_ signal: [Float], toLength targetLength: Int) -> [Float] {
        if signal.count == targetLength {
            return signal
        } else if signal.count > targetLength {
            return Array(signal.prefix(targetLength))
        } else {
            var padded = signal
            padded.append(contentsOf: [Float](repeating: 0, count: targetLength - signal.count))
            return padded
        }
    }
}

// MARK: - Presets

extension PhaseVocoder {
    /// Standard configuration for general use.
    public static var standard: PhaseVocoder {
        PhaseVocoder(config: PhaseVocoderConfig(
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann
        ))
    }

    /// High-quality configuration with larger FFT.
    public static var highQuality: PhaseVocoder {
        PhaseVocoder(config: PhaseVocoderConfig(
            nFFT: 4096,
            hopLength: 1024,
            windowType: .hann
        ))
    }

    /// Fast configuration with smaller FFT.
    public static var fast: PhaseVocoder {
        PhaseVocoder(config: PhaseVocoderConfig(
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann
        ))
    }
}
