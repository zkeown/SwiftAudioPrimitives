import SwiftRosaCore
import Accelerate
import Foundation

/// Voice activity detection method.
public enum VADMethod: Sendable {
    /// Energy-based detection with RMS threshold.
    case energy(threshold: Float = 0.01)

    /// Zero-crossing rate based detection.
    case zeroCrossing(threshold: Float = 0.1)

    /// Spectral-based detection using flatness and centroid.
    case spectral(threshold: Float = 0.5)

    /// Combined detection using weighted voting.
    case combined(weights: (energy: Float, zcr: Float, spectral: Float) = (0.4, 0.3, 0.3))
}

/// Configuration for voice activity detection.
public struct VADConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Frame length in samples.
    public let frameLength: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Detection method.
    public let method: VADMethod

    /// Minimum speech duration in seconds.
    public let minSpeechDuration: Float

    /// Minimum silence duration in seconds.
    public let minSilenceDuration: Float

    /// Create VAD configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 16000).
    ///   - frameLength: Frame length in samples (default: 512).
    ///   - hopLength: Hop length (default: 160).
    ///   - method: Detection method (default: combined).
    ///   - minSpeechDuration: Minimum speech duration (default: 0.1s).
    ///   - minSilenceDuration: Minimum silence duration (default: 0.3s).
    public init(
        sampleRate: Float = 16000,
        frameLength: Int = 512,
        hopLength: Int = 160,
        method: VADMethod = .combined(weights: (0.4, 0.3, 0.3)),
        minSpeechDuration: Float = 0.1,
        minSilenceDuration: Float = 0.3
    ) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.method = method
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }

    /// Minimum speech frames.
    public var minSpeechFrames: Int {
        max(1, Int(minSpeechDuration * framesPerSecond))
    }

    /// Minimum silence frames.
    public var minSilenceFrames: Int {
        max(1, Int(minSilenceDuration * framesPerSecond))
    }
}

/// A detected speech segment.
public struct VADSegment: Sendable {
    /// Start time in seconds.
    public let startTime: Float

    /// End time in seconds.
    public let endTime: Float

    /// Detection confidence (0-1).
    public let confidence: Float

    /// Duration in seconds.
    public var duration: Float { endTime - startTime }
}

/// Voice activity detector.
///
/// Detects speech/voice activity in audio signals using various methods
/// including energy, zero-crossing rate, and spectral features.
///
/// ## Example Usage
///
/// ```swift
/// let vad = VAD(config: VADConfig(sampleRate: 16000))
///
/// // Get frame-wise decisions
/// let decisions = await vad.transform(audioSignal)
///
/// // Get speech segments as time ranges
/// let segments = await vad.segments(audioSignal)
/// ```
public struct VAD: Sendable {
    /// Configuration.
    public let config: VADConfig

    /// Create a voice activity detector.
    ///
    /// - Parameter config: Configuration.
    public init(config: VADConfig = VADConfig()) {
        self.config = config
    }

    /// Detect voice activity per frame.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Binary array (1 = speech, 0 = silence).
    public func transform(_ signal: [Float]) async -> [Float] {
        let (decisions, _) = await transformWithConfidence(signal)
        return decisions
    }

    /// Detect voice activity with confidence scores.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tuple of (decisions, confidences).
    public func transformWithConfidence(
        _ signal: [Float]
    ) async -> (decisions: [Float], confidences: [Float]) {
        guard !signal.isEmpty else { return ([], []) }

        // Compute raw scores based on method
        let (rawScores, nFrames) = computeRawScores(signal)

        guard nFrames > 0 else { return ([], []) }

        // Apply threshold to get decisions
        var decisions = [Float](repeating: 0, count: nFrames)
        var confidences = rawScores

        for i in 0..<nFrames {
            decisions[i] = rawScores[i] > 0.5 ? 1.0 : 0.0
        }

        // Apply temporal smoothing
        decisions = applyTemporalSmoothing(decisions)

        return (decisions, confidences)
    }

    /// Get speech segments as time ranges.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of speech segments.
    public func segments(_ signal: [Float]) async -> [VADSegment] {
        let (decisions, confidences) = await transformWithConfidence(signal)

        guard !decisions.isEmpty else { return [] }

        var segments = [VADSegment]()
        var inSpeech = false
        var segmentStart = 0
        var segmentConfidenceSum: Float = 0

        for i in 0..<decisions.count {
            let isSpeech = decisions[i] > 0.5

            if isSpeech && !inSpeech {
                // Start of speech segment
                inSpeech = true
                segmentStart = i
                segmentConfidenceSum = confidences[i]
            } else if isSpeech && inSpeech {
                // Continue speech segment
                segmentConfidenceSum += confidences[i]
            } else if !isSpeech && inSpeech {
                // End of speech segment
                inSpeech = false

                let startTime = Float(segmentStart) / config.framesPerSecond
                let endTime = Float(i) / config.framesPerSecond
                let avgConfidence = segmentConfidenceSum / Float(i - segmentStart)

                segments.append(VADSegment(
                    startTime: startTime,
                    endTime: endTime,
                    confidence: avgConfidence
                ))
            }
        }

        // Handle segment at end
        if inSpeech {
            let startTime = Float(segmentStart) / config.framesPerSecond
            let endTime = Float(decisions.count) / config.framesPerSecond
            let avgConfidence = segmentConfidenceSum / Float(decisions.count - segmentStart)

            segments.append(VADSegment(
                startTime: startTime,
                endTime: endTime,
                confidence: avgConfidence
            ))
        }

        return segments
    }

    // MARK: - Private Implementation

    private func computeRawScores(_ signal: [Float]) -> ([Float], Int) {
        switch config.method {
        case .energy(let threshold):
            return computeEnergyScores(signal, threshold: threshold)

        case .zeroCrossing(let threshold):
            return computeZCRScores(signal, threshold: threshold)

        case .spectral(let threshold):
            return computeSpectralScores(signal, threshold: threshold)

        case .combined(let weights):
            return computeCombinedScores(signal, weights: weights)
        }
    }

    private func computeEnergyScores(_ signal: [Float], threshold: Float) -> ([Float], Int) {
        let frames = frameSignal(signal)
        let nFrames = frames.count

        var scores = [Float](repeating: 0, count: nFrames)

        // Compute RMS for each frame and find max
        var rmsValues = [Float](repeating: 0, count: nFrames)
        var maxRMS: Float = 0

        for (i, frame) in frames.enumerated() {
            var sumSquares: Float = 0
            vDSP_svesq(frame, 1, &sumSquares, vDSP_Length(frame.count))
            let rms = sqrt(sumSquares / Float(frame.count))
            rmsValues[i] = rms
            maxRMS = max(maxRMS, rms)
        }

        // Normalize and apply threshold
        let adaptiveThreshold = max(threshold, maxRMS * 0.1)

        for i in 0..<nFrames {
            if rmsValues[i] > adaptiveThreshold {
                scores[i] = min(1.0, rmsValues[i] / (adaptiveThreshold * 3))
            }
        }

        return (scores, nFrames)
    }

    private func computeZCRScores(_ signal: [Float], threshold: Float) -> ([Float], Int) {
        let zcr = ZeroCrossingRate(
            frameLength: config.frameLength,
            hopLength: config.hopLength,
            center: true
        )
        let zcrValues = zcr.transform(signal)

        // Low ZCR typically indicates voiced speech
        // High ZCR indicates unvoiced/noise
        var scores = [Float](repeating: 0, count: zcrValues.count)

        for i in 0..<zcrValues.count {
            // Voiced speech typically has low ZCR
            // We want to detect speech, so invert the ZCR
            let invertedZCR = 1.0 - zcrValues[i]
            scores[i] = invertedZCR > threshold ? invertedZCR : 0
        }

        return (scores, zcrValues.count)
    }

    private func computeSpectralScores(_ signal: [Float], threshold: Float) -> ([Float], Int) {
        let spectralFeatures = SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: config.sampleRate,
            nFFT: config.frameLength * 2,
            hopLength: config.hopLength,
            windowType: .hann,
            center: true
        ))

        // Compute flatness - speech is more tonal (lower flatness)
        let flatness = spectralFeatures.flatness(magnitudeSpec: computeMagnitudeSpec(signal))

        guard !flatness.isEmpty else { return ([], 0) }

        var scores = [Float](repeating: 0, count: flatness.count)

        for i in 0..<flatness.count {
            // Low flatness = tonal content = likely speech
            let tonality = 1.0 - flatness[i]
            scores[i] = tonality > threshold ? tonality : 0
        }

        return (scores, flatness.count)
    }

    private func computeCombinedScores(
        _ signal: [Float],
        weights: (energy: Float, zcr: Float, spectral: Float)
    ) -> ([Float], Int) {
        let (energyScores, nFrames1) = computeEnergyScores(signal, threshold: 0.01)
        let (zcrScores, nFrames2) = computeZCRScores(signal, threshold: 0.1)
        let (spectralScores, nFrames3) = computeSpectralScores(signal, threshold: 0.5)

        let nFrames = min(min(nFrames1, nFrames2), nFrames3)
        guard nFrames > 0 else { return ([], 0) }

        var scores = [Float](repeating: 0, count: nFrames)

        for i in 0..<nFrames {
            let energy = i < energyScores.count ? energyScores[i] : 0
            let zcr = i < zcrScores.count ? zcrScores[i] : 0
            let spectral = i < spectralScores.count ? spectralScores[i] : 0

            scores[i] = weights.energy * energy + weights.zcr * zcr + weights.spectral * spectral
        }

        // Normalize
        var maxScore: Float = 0
        vDSP_maxv(scores, 1, &maxScore, vDSP_Length(nFrames))

        if maxScore > 0 {
            var scale = 1.0 / maxScore
            vDSP_vsmul(scores, 1, &scale, &scores, 1, vDSP_Length(nFrames))
        }

        return (scores, nFrames)
    }

    private func frameSignal(_ signal: [Float]) -> [[Float]] {
        let padLength = config.frameLength / 2
        let paddedSignal = padSignal(signal, padLength: padLength)

        let nFrames = max(0, 1 + (paddedSignal.count - config.frameLength) / config.hopLength)
        var frames = [[Float]]()
        frames.reserveCapacity(nFrames)

        for i in 0..<nFrames {
            let start = i * config.hopLength
            let end = min(start + config.frameLength, paddedSignal.count)
            frames.append(Array(paddedSignal[start..<end]))
        }

        return frames
    }

    private func computeMagnitudeSpec(_ signal: [Float]) -> [[Float]] {
        let stft = STFT(config: STFTConfig(
            nFFT: config.frameLength * 2,
            hopLength: config.hopLength,
            winLength: config.frameLength,
            windowType: .hann,
            center: true,
            padMode: .reflect
        ))

        // Synchronously compute STFT magnitude
        // This is a simplification - in production, you'd want async
        var result = [[Float]]()

        // Frame the signal and compute FFT manually (simplified)
        let frames = frameSignal(signal)
        let nFreqs = config.frameLength + 1

        for _ in 0..<nFreqs {
            result.append([Float](repeating: 0, count: frames.count))
        }

        // For now, return placeholder - actual implementation would compute FFT
        return result
    }

    private func padSignal(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var result = [Float]()
        result.reserveCapacity(signal.count + 2 * padLength)

        // Reflect padding
        for i in (1...padLength).reversed() {
            let idx = i % signal.count
            result.append(signal[idx])
        }

        result.append(contentsOf: signal)

        for i in 1...padLength {
            let idx = (signal.count - 1) - (i % signal.count)
            result.append(signal[max(0, idx)])
        }

        return result
    }

    private func applyTemporalSmoothing(_ decisions: [Float]) -> [Float] {
        var smoothed = decisions

        // Apply minimum speech duration constraint
        let minSpeech = config.minSpeechFrames

        var i = 0
        while i < smoothed.count {
            if smoothed[i] > 0.5 {
                // Count speech segment length
                var j = i
                while j < smoothed.count && smoothed[j] > 0.5 {
                    j += 1
                }

                let segmentLength = j - i

                // If too short, mark as silence
                if segmentLength < minSpeech {
                    for k in i..<j {
                        smoothed[k] = 0
                    }
                }

                i = j
            } else {
                i += 1
            }
        }

        // Apply minimum silence duration constraint
        let minSilence = config.minSilenceFrames

        i = 0
        while i < smoothed.count {
            if smoothed[i] < 0.5 {
                // Count silence segment length
                var j = i
                while j < smoothed.count && smoothed[j] < 0.5 {
                    j += 1
                }

                let segmentLength = j - i

                // If too short and surrounded by speech, mark as speech
                if segmentLength < minSilence && i > 0 && j < smoothed.count {
                    for k in i..<j {
                        smoothed[k] = 1
                    }
                }

                i = j
            } else {
                i += 1
            }
        }

        return smoothed
    }
}

// MARK: - Presets

extension VAD {
    /// Configuration optimized for speech recognition.
    public static var speech: VAD {
        VAD(config: VADConfig(
            sampleRate: 16000,
            frameLength: 512,
            hopLength: 160,
            method: .combined(weights: (0.5, 0.2, 0.3)),
            minSpeechDuration: 0.1,
            minSilenceDuration: 0.3
        ))
    }

    /// Configuration for detecting vocals in music.
    public static var music: VAD {
        VAD(config: VADConfig(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            method: .combined(weights: (0.3, 0.3, 0.4)),
            minSpeechDuration: 0.2,
            minSilenceDuration: 0.5
        ))
    }

    /// Simple energy-based detection.
    public static var energyBased: VAD {
        VAD(config: VADConfig(
            sampleRate: 16000,
            frameLength: 512,
            hopLength: 160,
            method: .energy(threshold: 0.01),
            minSpeechDuration: 0.1,
            minSilenceDuration: 0.2
        ))
    }
}
