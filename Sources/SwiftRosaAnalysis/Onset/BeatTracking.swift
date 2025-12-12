import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for beat tracking.
public struct BeatTrackerConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Starting BPM estimate.
    public let startBPM: Float

    /// BPM search range.
    public let bpmRange: (min: Float, max: Float)

    /// Tightness of tempo constraint (higher = stricter).
    public let tightness: Float

    /// Create beat tracker configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - startBPM: Initial BPM estimate (default: 120).
    ///   - bpmRange: BPM search range (default: 60-240).
    ///   - tightness: Tempo tightness (default: 100).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        startBPM: Float = 120.0,
        bpmRange: (min: Float, max: Float) = (60.0, 240.0),
        tightness: Float = 100.0
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.startBPM = startBPM
        self.bpmRange = bpmRange
        self.tightness = tightness
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }

    /// Convert BPM to frames per beat.
    public func bpmToFrames(_ bpm: Float) -> Float {
        60.0 * framesPerSecond / bpm
    }

    /// Convert frames per beat to BPM.
    public func framesToBPM(_ frames: Float) -> Float {
        60.0 * framesPerSecond / frames
    }
}

/// Simple beat tracker.
///
/// Estimates tempo and tracks beat positions in audio signals.
/// Uses onset detection and autocorrelation-based tempo estimation.
///
/// ## Example Usage
///
/// ```swift
/// let tracker = BeatTracker()
///
/// // Estimate tempo
/// let bpm = await tracker.estimateTempo(audioSignal)
///
/// // Track beats
/// let beatTimes = await tracker.track(audioSignal)
/// ```
public struct BeatTracker: Sendable {
    /// Configuration.
    public let config: BeatTrackerConfig

    private let onsetDetector: OnsetDetector

    /// Create a beat tracker.
    ///
    /// - Parameter config: Configuration.
    public init(config: BeatTrackerConfig = BeatTrackerConfig()) {
        self.config = config
        self.onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: config.sampleRate,
            hopLength: config.hopLength,
            nFFT: 2048,
            method: .spectralFlux,
            peakThreshold: 0.3,
            preMax: 3,
            postMax: 3,
            preAvg: 3,
            postAvg: 3,
            delta: 0.05,
            wait: 10
        ))
    }

    /// Estimate tempo in BPM.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Estimated tempo in beats per minute.
    public func estimateTempo(_ signal: [Float]) async -> Float {
        // Compute onset strength
        let onsetEnvelope = await onsetDetector.onsetStrength(signal)
        guard !onsetEnvelope.isEmpty else { return config.startBPM }

        // Compute autocorrelation
        let autocorr = computeAutocorrelation(onsetEnvelope)

        // Find peaks in tempo range
        let minLag = Int(config.bpmToFrames(config.bpmRange.max))
        let maxLag = Int(config.bpmToFrames(config.bpmRange.min))

        var bestLag = Int(config.bpmToFrames(config.startBPM))
        var bestValue: Float = 0

        for lag in minLag..<min(maxLag, autocorr.count) {
            if autocorr[lag] > bestValue {
                bestValue = autocorr[lag]
                bestLag = lag
            }
        }

        // Convert lag to BPM
        return config.framesToBPM(Float(bestLag))
    }

    /// Track beats in audio.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of beat times in seconds.
    public func track(_ signal: [Float]) async -> [Float] {
        // Get onset envelope
        let onsetEnvelope = await onsetDetector.onsetStrength(signal)
        guard !onsetEnvelope.isEmpty else { return [] }

        // Estimate tempo
        let tempo = await estimateTempo(signal)
        let periodFrames = config.bpmToFrames(tempo)

        // Dynamic programming to find optimal beat sequence
        let beats = trackBeats(onsetEnvelope, periodFrames: periodFrames)

        // Convert to times
        return beats.map { Float($0) / config.framesPerSecond }
    }

    /// Get tempo in BPM and beat times together.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tuple of (tempo in BPM, beat times in seconds).
    public func analyze(_ signal: [Float]) async -> (tempo: Float, beats: [Float]) {
        let tempo = await estimateTempo(signal)
        let beats = await track(signal)
        return (tempo, beats)
    }

    // MARK: - Private Implementation

    private func computeAutocorrelation(_ signal: [Float]) -> [Float] {
        let n = signal.count
        let maxLag = min(n, Int(config.bpmToFrames(config.bpmRange.min)) + 100)

        var autocorr = [Float](repeating: 0, count: maxLag)

        // Compute autocorrelation for each lag
        for lag in 0..<maxLag {
            var sum: Float = 0
            for i in 0..<(n - lag) {
                sum += signal[i] * signal[i + lag]
            }
            autocorr[lag] = sum / Float(n - lag)
        }

        // Normalize
        if autocorr[0] > 0 {
            let scale = 1.0 / autocorr[0]
            for i in 0..<maxLag {
                autocorr[i] *= scale
            }
        }

        return autocorr
    }

    private func trackBeats(_ onsetEnvelope: [Float], periodFrames: Float) -> [Int] {
        let n = onsetEnvelope.count
        guard n > 0 else { return [] }

        // Normalize onset envelope
        var maxVal: Float = 0
        vDSP_maxv(onsetEnvelope, 1, &maxVal, vDSP_Length(n))
        var normalizedEnvelope = onsetEnvelope
        if maxVal > 0 {
            var scale = 1.0 / maxVal
            vDSP_vsmul(normalizedEnvelope, 1, &scale, &normalizedEnvelope, 1, vDSP_Length(n))
        }

        // Simple peak-picking with period constraint
        var beats = [Int]()
        let period = Int(periodFrames)
        let tolerance = period / 4

        // Find initial strong beat
        var bestStart = 0
        var bestStartValue: Float = 0
        for i in 0..<min(period * 2, n) {
            if normalizedEnvelope[i] > bestStartValue {
                bestStartValue = normalizedEnvelope[i]
                bestStart = i
            }
        }

        beats.append(bestStart)
        var lastBeat = bestStart

        // Track subsequent beats
        while lastBeat + period < n {
            let searchStart = lastBeat + period - tolerance
            let searchEnd = min(lastBeat + period + tolerance, n)

            // Find peak in search window
            var bestPos = lastBeat + period
            var bestValue: Float = 0

            for i in searchStart..<searchEnd {
                if normalizedEnvelope[i] > bestValue {
                    bestValue = normalizedEnvelope[i]
                    bestPos = i
                }
            }

            // Accept if above threshold or use predicted position
            if bestValue > 0.1 {
                beats.append(bestPos)
                lastBeat = bestPos
            } else {
                // Use predicted position
                let predicted = lastBeat + period
                if predicted < n {
                    beats.append(predicted)
                    lastBeat = predicted
                } else {
                    break
                }
            }
        }

        return beats
    }
}

// MARK: - Presets

extension BeatTracker {
    /// Standard configuration for general music.
    public static var standard: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 512,
            startBPM: 120.0,
            bpmRange: (60.0, 240.0),
            tightness: 100.0
        ))
    }

    /// Configuration for electronic/dance music (higher tempo range).
    public static var electronic: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 256,
            startBPM: 130.0,
            bpmRange: (100.0, 180.0),
            tightness: 150.0
        ))
    }

    /// Configuration for classical/slower music.
    public static var classical: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 512,
            startBPM: 80.0,
            bpmRange: (40.0, 160.0),
            tightness: 80.0
        ))
    }
}

// MARK: - Utility Methods

extension BeatTracker {
    /// Compute inter-beat intervals from beat times.
    ///
    /// - Parameter beatTimes: Beat times in seconds.
    /// - Returns: Intervals between consecutive beats.
    public static func interBeatIntervals(_ beatTimes: [Float]) -> [Float] {
        guard beatTimes.count > 1 else { return [] }

        var intervals = [Float]()
        intervals.reserveCapacity(beatTimes.count - 1)

        for i in 1..<beatTimes.count {
            intervals.append(beatTimes[i] - beatTimes[i - 1])
        }

        return intervals
    }

    /// Convert beat times to a click track.
    ///
    /// - Parameters:
    ///   - beatTimes: Beat times in seconds.
    ///   - sampleRate: Output sample rate.
    ///   - duration: Total duration in seconds.
    ///   - clickFrequency: Click tone frequency (default: 1000 Hz).
    ///   - clickDuration: Click duration in seconds (default: 0.02).
    /// - Returns: Click track audio signal.
    public static func generateClickTrack(
        beatTimes: [Float],
        sampleRate: Float,
        duration: Float,
        clickFrequency: Float = 1000.0,
        clickDuration: Float = 0.02
    ) -> [Float] {
        let totalSamples = Int(duration * sampleRate)
        var clickTrack = [Float](repeating: 0, count: totalSamples)

        let clickSamples = Int(clickDuration * sampleRate)

        for beatTime in beatTimes {
            let startSample = Int(beatTime * sampleRate)

            for i in 0..<clickSamples {
                let sampleIndex = startSample + i
                guard sampleIndex < totalSamples else { break }

                let phase = 2.0 * Float.pi * clickFrequency * Float(i) / sampleRate
                let envelope = 1.0 - Float(i) / Float(clickSamples)  // Decay
                clickTrack[sampleIndex] = sin(phase) * envelope * 0.5
            }
        }

        return clickTrack
    }
}
