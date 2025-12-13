import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for Predominant Local Pulse (PLP) detection.
public struct PLPConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Minimum tempo in BPM.
    public let tempoMin: Float

    /// Maximum tempo in BPM.
    public let tempoMax: Float

    /// Prior tempo (mode of the tempo distribution).
    public let prior: Float?

    /// Create PLP configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - tempoMin: Minimum tempo in BPM (default: 30).
    ///   - tempoMax: Maximum tempo in BPM (default: 300).
    ///   - prior: Prior tempo estimate (default: nil).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        tempoMin: Float = 30,
        tempoMax: Float = 300,
        prior: Float? = nil
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.tempoMin = tempoMin
        self.tempoMax = tempoMax
        self.prior = prior
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }
}

/// Predominant Local Pulse (PLP) detector.
///
/// Computes a pulse curve that represents the predominant periodicity
/// at each time frame. This is useful for beat tracking and tempo analysis
/// where the tempo may vary over time.
///
/// ## Example Usage
///
/// ```swift
/// let plp = PLP()
/// let pulse = await plp.compute(audioSignal)
/// // pulse[t] represents the local pulse strength at frame t
/// ```
///
/// - Note: Matches `librosa.beat.plp(y, sr)` behavior.
public struct PLP: Sendable {
    /// Configuration.
    public let config: PLPConfig

    /// Onset detector for computing onset strength.
    private let onsetDetector: OnsetDetector

    /// Create a PLP detector.
    ///
    /// - Parameter config: Configuration.
    public init(config: PLPConfig = PLPConfig()) {
        self.config = config
        self.onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: config.sampleRate,
            hopLength: config.hopLength,
            nFFT: 2048,
            method: .librosa
        ))
    }

    /// Compute predominant local pulse.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Pulse curve of shape (nFrames,).
    public func compute(_ signal: [Float]) async -> [Float] {
        // Get onset strength envelope
        let onset = await onsetDetector.onsetStrength(signal)
        guard !onset.isEmpty else { return [] }

        return computeFromOnset(onset)
    }

    /// Compute PLP from pre-computed onset strength.
    ///
    /// - Parameter onset: Onset strength envelope.
    /// - Returns: Pulse curve.
    public func computeFromOnset(_ onset: [Float]) -> [Float] {
        guard !onset.isEmpty else { return [] }

        let nFrames = onset.count

        // Convert tempo range to period range (in frames)
        let fps = config.framesPerSecond
        let periodMin = Int(60.0 / config.tempoMax * fps)  // Faster tempo = shorter period
        let periodMax = Int(60.0 / config.tempoMin * fps)  // Slower tempo = longer period

        guard periodMin > 0, periodMax > periodMin else { return onset }

        // Compute local autocorrelation for each frame
        var pulse = [Float](repeating: 0, count: nFrames)

        // Window size for local analysis (should cover ~2 periods max)
        let windowSize = min(periodMax * 4, nFrames)
        let halfWindow = windowSize / 2

        for t in 0..<nFrames {
            // Extract local window centered at t
            let startIdx = max(0, t - halfWindow)
            let endIdx = min(nFrames, t + halfWindow)
            let localWindow = Array(onset[startIdx..<endIdx])

            guard localWindow.count > periodMin else {
                pulse[t] = onset[t]
                continue
            }

            // Compute autocorrelation for periods in valid range
            var maxCorr: Float = 0
            var bestPeriod = periodMin

            let periodUpperBound = min(periodMax, localWindow.count / 2)
            guard periodUpperBound >= periodMin else {
                pulse[t] = onset[t]
                continue
            }

            for period in periodMin...periodUpperBound {
                let corr = autocorrelation(localWindow, lag: period)

                // Weight by prior if provided
                var weightedCorr = corr
                if let prior = config.prior {
                    let priorPeriod = 60.0 / prior * fps
                    let priorWeight = exp(-pow(Float(period) - priorPeriod, 2) / (2 * 100))
                    weightedCorr *= (1 + priorWeight)
                }

                if weightedCorr > maxCorr {
                    maxCorr = weightedCorr
                    bestPeriod = period
                }
            }

            // The pulse value is the onset strength weighted by periodicity confidence
            let periodicityConfidence = maxCorr / (localWindow.reduce(0) { $0 + $1 * $1 } + 1e-10)
            pulse[t] = onset[t] * (1 + periodicityConfidence)
        }

        // Normalize pulse curve
        var maxPulse: Float = 0
        vDSP_maxv(pulse, 1, &maxPulse, vDSP_Length(nFrames))
        if maxPulse > 0 {
            var scale = 1.0 / maxPulse
            vDSP_vsmul(pulse, 1, &scale, &pulse, 1, vDSP_Length(nFrames))
        }

        return pulse
    }

    /// Compute normalized autocorrelation at a specific lag.
    private func autocorrelation(_ signal: [Float], lag: Int) -> Float {
        guard lag < signal.count else { return 0 }

        var sum: Float = 0
        var norm1: Float = 0
        var norm2: Float = 0

        for i in 0..<(signal.count - lag) {
            sum += signal[i] * signal[i + lag]
            norm1 += signal[i] * signal[i]
            norm2 += signal[i + lag] * signal[i + lag]
        }

        let normProduct = sqrt(norm1 * norm2)
        return normProduct > 0 ? sum / normProduct : 0
    }
}

// MARK: - Presets

extension PLP {
    /// Standard configuration for music analysis.
    public static var standard: PLP {
        PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 30,
            tempoMax: 300,
            prior: nil
        ))
    }

    /// Configuration optimized for dance music (faster tempo).
    public static var danceMusic: PLP {
        PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 100,
            tempoMax: 200,
            prior: 128  // Common dance music tempo
        ))
    }

    /// Configuration for slow music.
    public static var slowMusic: PLP {
        PLP(config: PLPConfig(
            sampleRate: 22050,
            hopLength: 512,
            tempoMin: 40,
            tempoMax: 120,
            prior: 80
        ))
    }
}
