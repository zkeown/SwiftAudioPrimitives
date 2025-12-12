import SwiftRosaCore
import Accelerate
import Foundation

/// Onset detection method.
public enum OnsetMethod: Sendable {
    /// Energy-based onset detection.
    case energy
    /// Spectral flux (change in magnitude spectrum).
    case spectralFlux
    /// Complex domain (magnitude and phase changes).
    case complexDomain
    /// High-frequency content.
    case highFrequencyContent
    /// RMS energy.
    case rms
}

/// Configuration for onset detection.
public struct OnsetConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// FFT size.
    public let nFFT: Int

    /// Detection method.
    public let method: OnsetMethod

    /// Peak picking threshold.
    public let peakThreshold: Float

    /// Pre-maximum waiting period (frames).
    public let preMax: Int

    /// Post-maximum waiting period (frames).
    public let postMax: Int

    /// Pre-average context (frames).
    public let preAvg: Int

    /// Post-average context (frames).
    public let postAvg: Int

    /// Threshold offset.
    public let delta: Float

    /// Minimum wait between onsets (frames).
    public let wait: Int

    /// Create onset detection configuration.
    ///
    /// Default values match librosa's `onset_detect` function.
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        nFFT: Int = 2048,
        method: OnsetMethod = .spectralFlux,
        peakThreshold: Float = 0.3,
        preMax: Int = 1,
        postMax: Int = 1,
        preAvg: Int = 1,
        postAvg: Int = 1,
        delta: Float = 0.0,
        wait: Int = 1
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.nFFT = nFFT
        self.method = method
        self.peakThreshold = peakThreshold
        self.preMax = preMax
        self.postMax = postMax
        self.preAvg = preAvg
        self.postAvg = postAvg
        self.delta = delta
        self.wait = wait
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }
}

/// Onset detector.
///
/// Detects note onsets (attacks, transients) in audio signals.
/// Useful for beat tracking, music transcription, and segmentation.
///
/// ## Example Usage
///
/// ```swift
/// let detector = OnsetDetector()
///
/// // Get onset times
/// let onsetTimes = await detector.detectTimes(audioSignal)
///
/// // Get onset strength envelope
/// let envelope = await detector.onsetStrength(audioSignal)
/// ```
public struct OnsetDetector: Sendable {
    /// Configuration.
    public let config: OnsetConfig

    private let stft: STFT

    /// Create an onset detector.
    ///
    /// - Parameter config: Configuration.
    public init(config: OnsetConfig = OnsetConfig()) {
        self.config = config
        self.stft = STFT(config: STFTConfig(
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.nFFT,
            windowType: .hann,
            center: true,
            padMode: .constant(0)  // Match librosa default
        ))
    }

    /// Compute onset strength envelope.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Onset strength per frame.
    public func onsetStrength(_ signal: [Float]) async -> [Float] {
        let spectrogram = await stft.transform(signal)

        switch config.method {
        case .energy:
            return computeEnergyOnset(spectrogram)
        case .spectralFlux:
            return computeSpectralFlux(spectrogram)
        case .complexDomain:
            return computeComplexDomain(spectrogram)
        case .highFrequencyContent:
            return computeHFC(spectrogram)
        case .rms:
            return computeRMSOnset(spectrogram)
        }
    }

    /// Detect onset frame indices.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of frame indices where onsets occur.
    public func detect(_ signal: [Float]) async -> [Int] {
        guard signal.count >= config.nFFT else { return [] }
        let envelope = await onsetStrength(signal)
        return pickPeaks(envelope)
    }

    /// Detect onset times in seconds.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of onset times in seconds.
    public func detectTimes(_ signal: [Float]) async -> [Float] {
        let frameIndices = await detect(signal)
        return frameIndices.map { Float($0) / config.framesPerSecond }
    }

    /// Pick peaks from onset envelope.
    ///
    /// - Parameter onsetEnvelope: Onset strength envelope.
    /// - Returns: Indices of detected peaks.
    public func pickPeaks(_ onsetEnvelope: [Float]) -> [Int] {
        guard !onsetEnvelope.isEmpty else { return [] }

        var peaks = [Int]()
        let n = onsetEnvelope.count

        // Ensure we have enough samples for peak picking
        guard n > config.preMax + config.postMax else { return [] }

        // Normalize envelope
        var maxVal: Float = 0
        vDSP_maxv(onsetEnvelope, 1, &maxVal, vDSP_Length(n))

        // If the envelope maximum is below a meaningful threshold,
        // there's no significant onset activity (just numerical noise)
        let minMeaningfulOnset: Float = 1e-6
        guard maxVal > minMeaningfulOnset else { return [] }

        var normalizedEnvelope = [Float](repeating: 0, count: n)
        var scale = 1.0 / maxVal
        vDSP_vsmul(onsetEnvelope, 1, &scale, &normalizedEnvelope, 1, vDSP_Length(n))

        var lastOnset = -config.wait

        for i in config.preMax..<(n - config.postMax) {
            // Check if this is a local maximum
            var isMax = true
            for j in (i - config.preMax)...(i + config.postMax) {
                if j != i && normalizedEnvelope[j] >= normalizedEnvelope[i] {
                    isMax = false
                    break
                }
            }

            guard isMax else { continue }

            // Compute local average
            let avgStart = max(0, i - config.preAvg)
            let avgEnd = min(n, i + config.postAvg + 1)
            var localAvg: Float = 0
            for j in avgStart..<avgEnd {
                localAvg += normalizedEnvelope[j]
            }
            localAvg /= Float(avgEnd - avgStart)

            // Check threshold
            let threshold = localAvg + config.delta
            if normalizedEnvelope[i] >= threshold && normalizedEnvelope[i] >= config.peakThreshold {
                // Check wait constraint
                if i - lastOnset >= config.wait {
                    peaks.append(i)
                    lastOnset = i
                }
            }
        }

        return peaks
    }

    // MARK: - Onset Detection Methods

    private func computeEnergyOnset(_ spectrogram: ComplexMatrix) -> [Float] {
        let mag = spectrogram.magnitude
        guard !mag.isEmpty && !mag[0].isEmpty else { return [] }

        let nFreqs = mag.count
        let nFrames = mag[0].count

        var envelope = [Float](repeating: 0, count: nFrames)

        // Sum squared magnitudes per frame
        for t in 0..<nFrames {
            var sum: Float = 0
            for f in 0..<nFreqs {
                sum += mag[f][t] * mag[f][t]
            }
            envelope[t] = sum
        }

        // Compute difference (onset = energy increase)
        return computeHalfWaveDiff(envelope)
    }

    private func computeSpectralFlux(_ spectrogram: ComplexMatrix) -> [Float] {
        let mag = spectrogram.magnitude
        guard !mag.isEmpty && !mag[0].isEmpty else { return [] }

        let nFreqs = mag.count
        let nFrames = mag[0].count

        var envelope = [Float](repeating: 0, count: nFrames)

        for t in 1..<nFrames {
            var flux: Float = 0
            for f in 0..<nFreqs {
                // Half-wave rectified difference
                let diff = mag[f][t] - mag[f][t - 1]
                flux += max(0, diff)
            }
            envelope[t] = flux
        }

        return envelope
    }

    private func computeComplexDomain(_ spectrogram: ComplexMatrix) -> [Float] {
        guard !spectrogram.real.isEmpty && !spectrogram.real[0].isEmpty else { return [] }

        let nFreqs = spectrogram.rows
        let nFrames = spectrogram.cols

        var envelope = [Float](repeating: 0, count: nFrames)

        // Need at least 2 frames for prediction
        guard nFrames >= 2 else { return envelope }

        for t in 2..<nFrames {
            var deviation: Float = 0

            for f in 0..<nFreqs {
                // Current complex value
                let real = spectrogram.real[f][t]
                let imag = spectrogram.imag[f][t]

                // Previous complex value
                let prevReal = spectrogram.real[f][t - 1]
                let prevImag = spectrogram.imag[f][t - 1]

                // Predicted magnitude (assume constant)
                let prevMag = sqrt(prevReal * prevReal + prevImag * prevImag)

                // Predicted phase (linear extrapolation)
                let prevPhase = atan2(prevImag, prevReal)
                let prevPrevPhase = atan2(spectrogram.imag[f][t - 2], spectrogram.real[f][t - 2])
                let predictedPhase = 2 * prevPhase - prevPrevPhase

                // Predicted complex value
                let predictedReal = prevMag * cos(predictedPhase)
                let predictedImag = prevMag * sin(predictedPhase)

                // Deviation from prediction
                let diffReal = real - predictedReal
                let diffImag = imag - predictedImag
                deviation += sqrt(diffReal * diffReal + diffImag * diffImag)
            }

            envelope[t] = deviation
        }

        return envelope
    }

    private func computeHFC(_ spectrogram: ComplexMatrix) -> [Float] {
        let mag = spectrogram.magnitude
        guard !mag.isEmpty && !mag[0].isEmpty else { return [] }

        let nFreqs = mag.count
        let nFrames = mag[0].count

        var envelope = [Float](repeating: 0, count: nFrames)

        // Weight by frequency bin index (emphasize high frequencies)
        for t in 0..<nFrames {
            var hfc: Float = 0
            for f in 0..<nFreqs {
                hfc += Float(f) * mag[f][t] * mag[f][t]
            }
            envelope[t] = hfc
        }

        // Compute difference
        return computeHalfWaveDiff(envelope)
    }

    private func computeRMSOnset(_ spectrogram: ComplexMatrix) -> [Float] {
        let mag = spectrogram.magnitude
        guard !mag.isEmpty && !mag[0].isEmpty else { return [] }

        let nFreqs = mag.count
        let nFrames = mag[0].count

        var envelope = [Float](repeating: 0, count: nFrames)

        for t in 0..<nFrames {
            var sumSquares: Float = 0
            for f in 0..<nFreqs {
                sumSquares += mag[f][t] * mag[f][t]
            }
            envelope[t] = sqrt(sumSquares / Float(nFreqs))
        }

        return computeHalfWaveDiff(envelope)
    }

    private func computeHalfWaveDiff(_ signal: [Float]) -> [Float] {
        guard !signal.isEmpty else { return [] }

        var diff = [Float](repeating: 0, count: signal.count)

        for i in 1..<signal.count {
            diff[i] = max(0, signal[i] - signal[i - 1])
        }

        return diff
    }
}

// MARK: - Presets

extension OnsetDetector {
    /// Standard configuration for general use (matches librosa defaults).
    public static var standard: OnsetDetector {
        OnsetDetector(config: OnsetConfig(
            sampleRate: 22050,
            hopLength: 512,
            nFFT: 2048,
            method: .spectralFlux,
            peakThreshold: 0.3,
            preMax: 1,
            postMax: 1,
            preAvg: 1,
            postAvg: 1,
            delta: 0.0,
            wait: 1
        ))
    }

    /// Optimized for percussive sounds.
    public static var percussive: OnsetDetector {
        OnsetDetector(config: OnsetConfig(
            sampleRate: 22050,
            hopLength: 256,
            nFFT: 1024,
            method: .highFrequencyContent,
            peakThreshold: 0.2,
            preMax: 1,
            postMax: 1,
            preAvg: 1,
            postAvg: 1,
            delta: 0.0,
            wait: 1
        ))
    }

    /// Optimized for melodic instruments.
    public static var melodic: OnsetDetector {
        OnsetDetector(config: OnsetConfig(
            sampleRate: 22050,
            hopLength: 512,
            nFFT: 2048,
            method: .complexDomain,
            peakThreshold: 0.3,
            preMax: 3,
            postMax: 3,
            preAvg: 3,
            postAvg: 3,
            delta: 0.05,
            wait: 1
        ))
    }
}
