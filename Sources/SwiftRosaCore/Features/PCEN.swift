import Accelerate
import Foundation

/// Configuration for PCEN (Per-Channel Energy Normalization).
public struct PCENConfig: Sendable {
    /// Gain factor (alpha). Controls compression amount. Default: 0.98.
    public let gain: Float

    /// Bias term (delta). Adds stability. Default: 2.0.
    public let bias: Float

    /// Power exponent (r). Controls output compression. Default: 0.5.
    public let power: Float

    /// Time constant in seconds for IIR smoothing. Default: 0.025.
    public let timeConstant: Float

    /// Sample rate in Hz (for computing IIR coefficient from time constant).
    public let sampleRate: Float

    /// Hop length in samples (for computing IIR coefficient from time constant).
    public let hopLength: Int

    /// Small constant for numerical stability. Default: 1e-6.
    public let eps: Float

    /// IIR coefficient (b). If nil, computed from timeConstant.
    public let b: Float?

    /// Trainable PCEN parameters (for differentiable learning).
    public let trainable: Bool

    /// Create PCEN configuration.
    ///
    /// - Parameters:
    ///   - gain: Gain factor alpha (default: 0.98).
    ///   - bias: Bias term delta (default: 2.0).
    ///   - power: Power exponent r (default: 0.5).
    ///   - timeConstant: Time constant in seconds (default: 0.025).
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length in samples (default: 512).
    ///   - eps: Numerical stability constant (default: 1e-6).
    ///   - b: Manual IIR coefficient (optional, overrides timeConstant).
    ///   - trainable: Whether parameters are trainable (default: false).
    public init(
        gain: Float = 0.98,
        bias: Float = 2.0,
        power: Float = 0.5,
        timeConstant: Float = 0.025,
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        eps: Float = 1e-6,
        b: Float? = nil,
        trainable: Bool = false
    ) {
        self.gain = gain
        self.bias = bias
        self.power = power
        self.timeConstant = timeConstant
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.eps = eps
        self.b = b
        self.trainable = trainable
    }

    /// Compute IIR smoothing coefficient from time constant.
    public var smoothingCoeff: Float {
        if let manualB = b {
            return manualB
        }

        // s = time constant in frames
        let s = timeConstant * sampleRate / Float(hopLength)

        // b = (sqrt(1 + 4*s²) - 1) / (2*s²)
        let s2 = s * s
        return (sqrt(1 + 4 * s2) - 1) / (2 * s2)
    }
}

/// Per-Channel Energy Normalization.
///
/// PCEN is an adaptive gain control mechanism that normalizes energy
/// per frequency channel, providing robustness to background noise
/// and channel variations.
///
/// ## Example Usage
///
/// ```swift
/// let pcen = PCEN(config: PCENConfig(gain: 0.98, bias: 2.0, power: 0.5))
///
/// // Apply PCEN to mel spectrogram
/// let normalized = pcen.transform(melSpectrogram)
///
/// // Or with custom smoothing
/// let normalized = pcen.transform(melSpectrogram, initialM: previousM)
/// ```
///
/// ## Formula
///
/// ```
/// output = (E / (eps + M)^alpha + delta)^r - delta^r
/// ```
///
/// Where:
/// - E = input energy (spectrogram)
/// - M = smoothed energy via IIR filter: M[t] = (1-b) * M[t-1] + b * E[t]
/// - alpha = gain (compression exponent)
/// - delta = bias (for stability and floor)
/// - r = power (output compression)
public struct PCEN: Sendable {
    /// Configuration.
    public let config: PCENConfig

    /// Computed smoothing coefficient.
    public var smoothingCoeff: Float { config.smoothingCoeff }

    /// Create PCEN processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: PCENConfig = PCENConfig()) {
        self.config = config
    }

    /// Apply PCEN to a spectrogram.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram of shape (nFreqs, nFrames).
    ///   - initialM: Initial smoothed energy state (optional, for streaming).
    /// - Returns: PCEN-normalized spectrogram of shape (nFreqs, nFrames).
    public func transform(
        _ spectrogram: [[Float]],
        initialM: [Float]? = nil
    ) -> [[Float]] {
        guard !spectrogram.isEmpty, !spectrogram[0].isEmpty else { return [] }

        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        // Compute smoothed energy M via IIR filter
        let M = computeSmoothed(spectrogram, initialM: initialM)

        // Apply PCEN formula
        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let alpha = config.gain
        let delta = config.bias
        let r = config.power
        let eps = config.eps

        // Pre-compute delta^r for offset
        let deltaR = pow(delta, r)

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                let E = spectrogram[freqIdx][frameIdx]
                let m = M[freqIdx][frameIdx]

                // AGC: E / (eps + M)^alpha
                let agc = E / pow(eps + m, alpha)

                // Output: (agc + delta)^r - delta^r
                row[frameIdx] = pow(agc + delta, r) - deltaR
            }

            result.append(row)
        }

        return result
    }

    /// Apply PCEN to a spectrogram, returning both output and final M state.
    ///
    /// Useful for streaming applications where you need to maintain state.
    ///
    /// - Parameters:
    ///   - spectrogram: Input spectrogram of shape (nFreqs, nFrames).
    ///   - initialM: Initial smoothed energy state (optional).
    /// - Returns: Tuple of (PCEN output, final M state per frequency).
    public func transformWithState(
        _ spectrogram: [[Float]],
        initialM: [Float]? = nil
    ) -> (output: [[Float]], finalM: [Float]) {
        guard !spectrogram.isEmpty, !spectrogram[0].isEmpty else { return ([], []) }

        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count

        // Compute smoothed energy M via IIR filter
        let M = computeSmoothed(spectrogram, initialM: initialM)

        // Extract final M state
        var finalM = [Float](repeating: 0, count: nFreqs)
        for freqIdx in 0..<nFreqs {
            finalM[freqIdx] = M[freqIdx][nFrames - 1]
        }

        // Apply PCEN formula
        var result = [[Float]]()
        result.reserveCapacity(nFreqs)

        let alpha = config.gain
        let delta = config.bias
        let r = config.power
        let eps = config.eps
        let deltaR = pow(delta, r)

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                let E = spectrogram[freqIdx][frameIdx]
                let m = M[freqIdx][frameIdx]
                let agc = E / pow(eps + m, alpha)
                row[frameIdx] = pow(agc + delta, r) - deltaR
            }

            result.append(row)
        }

        return (result, finalM)
    }

    // MARK: - Private Implementation

    /// Compute smoothed energy via IIR low-pass filter.
    private func computeSmoothed(
        _ spectrogram: [[Float]],
        initialM: [Float]?
    ) -> [[Float]] {
        let nFreqs = spectrogram.count
        let nFrames = spectrogram[0].count
        let b = smoothingCoeff
        let oneMinusB = 1.0 - b

        var M = [[Float]]()
        M.reserveCapacity(nFreqs)

        for freqIdx in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)

            // Initialize M[0]
            let m0 = initialM?[freqIdx] ?? spectrogram[freqIdx][0]
            row[0] = oneMinusB * m0 + b * spectrogram[freqIdx][0]

            // IIR filter: M[t] = (1-b) * M[t-1] + b * E[t]
            for frameIdx in 1..<nFrames {
                row[frameIdx] = oneMinusB * row[frameIdx - 1] + b * spectrogram[freqIdx][frameIdx]
            }

            M.append(row)
        }

        return M
    }
}

// MARK: - Presets

extension PCEN {
    /// Standard PCEN configuration matching librosa defaults.
    public static var standard: PCEN {
        PCEN(config: PCENConfig(
            gain: 0.98,
            bias: 2.0,
            power: 0.5,
            timeConstant: 0.025
        ))
    }

    /// PCEN configuration for speech processing.
    public static var speech: PCEN {
        PCEN(config: PCENConfig(
            gain: 0.98,
            bias: 2.0,
            power: 0.5,
            timeConstant: 0.04,
            sampleRate: 16000,
            hopLength: 160
        ))
    }

    /// PCEN configuration with stronger compression.
    public static var strongCompression: PCEN {
        PCEN(config: PCENConfig(
            gain: 0.8,
            bias: 10.0,
            power: 0.25,
            timeConstant: 0.06
        ))
    }
}
