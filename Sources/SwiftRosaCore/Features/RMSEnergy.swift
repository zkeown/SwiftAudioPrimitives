import Accelerate
import Foundation

/// Configuration for RMS energy computation.
public struct RMSEnergyConfig: Sendable {
    /// Frame length in samples.
    public let frameLength: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Whether to center the signal by padding.
    public let center: Bool

    /// Create RMS energy configuration.
    ///
    /// - Parameters:
    ///   - frameLength: Frame length in samples (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - center: Whether to center frames by padding (default: true).
    public init(
        frameLength: Int = 2048,
        hopLength: Int = 512,
        center: Bool = true
    ) {
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.center = center
    }
}

/// Root-mean-square energy feature extractor.
///
/// Computes the RMS energy per frame, which represents the short-term
/// power of the signal. This is useful for voice activity detection,
/// audio segmentation, and amplitude analysis.
///
/// ## Example Usage
///
/// ```swift
/// let rms = RMSEnergy(config: RMSEnergyConfig(frameLength: 2048, hopLength: 512))
///
/// // From raw signal
/// let energy = rms.transform(audioSignal)
///
/// // From pre-computed magnitude spectrogram (more efficient if STFT already computed)
/// let energyFromSpec = rms.transform(magnitudeSpec: spectrogram)
/// ```
///
/// ## Formula
///
/// For each frame: `RMS[t] = sqrt(mean(frame[t]²))`
///
/// For magnitude spectrogram: `RMS = sqrt(mean(S², axis=0))` where S is the magnitude.
public struct RMSEnergy: Sendable {
    /// Configuration.
    public let config: RMSEnergyConfig

    /// Create RMS energy extractor.
    ///
    /// - Parameter config: Configuration (default: standard settings).
    public init(config: RMSEnergyConfig = RMSEnergyConfig()) {
        self.config = config
    }

    /// Compute RMS energy from raw audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: RMS energy per frame.
    public func transform(_ signal: [Float]) -> [Float] {
        guard !signal.isEmpty else { return [] }

        // Pad signal if centering
        let paddedSignal: [Float]
        if config.center {
            let padLength = config.frameLength / 2
            paddedSignal = padSignal(signal, padLength: padLength)
        } else {
            paddedSignal = signal
        }

        // Compute number of frames
        let nFrames = max(0, 1 + (paddedSignal.count - config.frameLength) / config.hopLength)
        guard nFrames > 0 else { return [] }

        var rmsValues = [Float](repeating: 0, count: nFrames)

        // Process each frame
        for frameIdx in 0..<nFrames {
            let start = frameIdx * config.hopLength
            let end = min(start + config.frameLength, paddedSignal.count)
            let frameLength = end - start

            guard frameLength > 0 else { continue }

            // Compute sum of squares using vDSP
            var sumSquares: Float = 0
            paddedSignal.withUnsafeBufferPointer { ptr in
                vDSP_svesq(ptr.baseAddress! + start, 1, &sumSquares, vDSP_Length(frameLength))
            }

            // Compute RMS: sqrt(mean(x²))
            rmsValues[frameIdx] = sqrt(sumSquares / Float(frameLength))
        }

        return rmsValues
    }

    /// Compute RMS energy from pre-computed magnitude spectrogram.
    ///
    /// This is more efficient when you already have the magnitude spectrogram
    /// computed from STFT, as it avoids redundant FFT computation.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    /// - Returns: RMS energy per frame.
    public func transform(magnitudeSpec: [[Float]]) -> [Float] {
        guard !magnitudeSpec.isEmpty, !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        var rmsValues = [Float](repeating: 0, count: nFrames)

        // For each frame, compute RMS across frequency bins
        // RMS = sqrt(mean(S²)) = sqrt(sum(S²) / nFreqs)
        for frameIdx in 0..<nFrames {
            var sumSquares: Float = 0

            for freqIdx in 0..<nFreqs {
                let mag = magnitudeSpec[freqIdx][frameIdx]
                sumSquares += mag * mag
            }

            rmsValues[frameIdx] = sqrt(sumSquares / Float(nFreqs))
        }

        return rmsValues
    }

    /// Compute RMS energy in decibels.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - refValue: Reference value for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: RMS energy in dB per frame.
    public func transformDB(
        _ signal: [Float],
        refValue: Float = 1.0,
        minDB: Float = -80.0
    ) -> [Float] {
        let rms = transform(signal)
        return rmsToDecibels(rms, refValue: refValue, minDB: minDB)
    }

    /// Compute RMS energy in decibels from magnitude spectrogram.
    ///
    /// - Parameters:
    ///   - magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    ///   - refValue: Reference value for dB conversion (default: 1.0).
    ///   - minDB: Minimum dB value to clip to (default: -80).
    /// - Returns: RMS energy in dB per frame.
    public func transformDB(
        magnitudeSpec: [[Float]],
        refValue: Float = 1.0,
        minDB: Float = -80.0
    ) -> [Float] {
        let rms = transform(magnitudeSpec: magnitudeSpec)
        return rmsToDecibels(rms, refValue: refValue, minDB: minDB)
    }

    // MARK: - Private Implementation

    private func padSignal(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var result = [Float]()
        result.reserveCapacity(signal.count + 2 * padLength)

        // Use constant (zero) padding to match librosa default for RMS
        // librosa.feature.rms uses pad_mode='constant' by default
        result.append(contentsOf: [Float](repeating: 0, count: padLength))

        // Original signal
        result.append(contentsOf: signal)

        // Right padding with zeros
        result.append(contentsOf: [Float](repeating: 0, count: padLength))

        return result
    }

    private func rmsToDecibels(
        _ rms: [Float],
        refValue: Float,
        minDB: Float
    ) -> [Float] {
        guard !rms.isEmpty else { return [] }

        var dbValues = [Float](repeating: 0, count: rms.count)
        let epsilon: Float = 1e-10

        for i in 0..<rms.count {
            // dB = 20 * log10(rms / refValue)
            let ratio = max(rms[i], epsilon) / refValue
            let db = 20.0 * log10(ratio)
            dbValues[i] = max(db, minDB)
        }

        return dbValues
    }
}

// MARK: - Presets

extension RMSEnergy {
    /// Standard configuration matching librosa defaults.
    public static var standard: RMSEnergy {
        RMSEnergy(config: RMSEnergyConfig(
            frameLength: 2048,
            hopLength: 512,
            center: true
        ))
    }

    /// Configuration optimized for speech analysis.
    public static var speech: RMSEnergy {
        RMSEnergy(config: RMSEnergyConfig(
            frameLength: 512,
            hopLength: 160,
            center: true
        ))
    }

    /// Configuration for music analysis with larger frames.
    public static var music: RMSEnergy {
        RMSEnergy(config: RMSEnergyConfig(
            frameLength: 4096,
            hopLength: 1024,
            center: true
        ))
    }
}
