import Accelerate
import Foundation

/// Errors that can occur during pitch detection.
public enum PitchDetectorError: Error, Sendable {
    case invalidParameters(String)
    case analysisError(String)
}

/// Configuration for pitch detection.
public struct PitchDetectorConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Frame length in samples.
    public let frameLength: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Minimum detectable frequency (Hz).
    public let fMin: Float

    /// Maximum detectable frequency (Hz).
    public let fMax: Float

    /// Detection threshold (0-1). Lower = more sensitive.
    public let threshold: Float

    /// Create pitch detector configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - frameLength: Frame length in samples (default: 2048).
    ///   - hopLength: Hop length (default: 512).
    ///   - fMin: Minimum frequency (default: 65 Hz, ~C2).
    ///   - fMax: Maximum frequency (default: 2093 Hz, ~C7).
    ///   - threshold: Detection threshold (default: 0.1).
    public init(
        sampleRate: Float = 22050,
        frameLength: Int = 2048,
        hopLength: Int = 512,
        fMin: Float = 65.0,
        fMax: Float = 2093.0,
        threshold: Float = 0.1
    ) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.fMin = fMin
        self.fMax = max(fMin + 1, min(fMax, sampleRate / 2))
        self.threshold = threshold
    }

    /// Minimum lag for period search.
    public var minLag: Int {
        max(2, Int(sampleRate / fMax))
    }

    /// Maximum lag for period search.
    public var maxLag: Int {
        min(frameLength / 2, Int(sampleRate / fMin))
    }
}

/// YIN pitch detection algorithm.
///
/// Implements the YIN algorithm for fundamental frequency (F0) estimation.
/// YIN is a widely-used pitch detection method that works well for both
/// speech and music.
///
/// ## Algorithm Overview
///
/// 1. Compute difference function for each lag
/// 2. Compute cumulative mean normalized difference
/// 3. Find first minimum below threshold
/// 4. Apply parabolic interpolation for sub-sample accuracy
/// 5. Convert period to frequency
///
/// ## Example Usage
///
/// ```swift
/// let detector = PitchDetector(config: PitchDetectorConfig(sampleRate: 16000))
///
/// // Detect pitch per frame
/// let pitches = await detector.transform(audioSignal)
///
/// // Detect pitch with confidence values
/// let (pitches, confidences) = await detector.transformWithConfidence(audioSignal)
/// ```
///
/// ## References
///
/// A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator
/// for speech and music," J. Acoust. Soc. Am., vol. 111, no. 4, 2002.
public struct PitchDetector: Sendable {
    /// Configuration.
    public let config: PitchDetectorConfig

    /// Create a pitch detector.
    ///
    /// - Parameter config: Configuration (default: music-optimized).
    public init(config: PitchDetectorConfig = PitchDetectorConfig()) {
        self.config = config
    }

    /// Detect pitch (F0) for each frame.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Fundamental frequency per frame in Hz.
    ///   0.0 indicates unvoiced/undetected.
    public func transform(_ signal: [Float]) async -> [Float] {
        let (pitches, _) = await transformWithConfidence(signal)
        return pitches
    }

    /// Detect pitch with confidence values.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tuple of (pitches, confidences).
    ///   Pitches in Hz (0 = unvoiced), confidences 0-1.
    public func transformWithConfidence(
        _ signal: [Float]
    ) async -> (pitches: [Float], confidences: [Float]) {
        guard signal.count >= config.frameLength else {
            return ([], [])
        }

        // Pad signal for centered frames
        let padLength = config.frameLength / 2
        let paddedSignal = padSignal(signal, padLength: padLength)

        let nFrames = 1 + (paddedSignal.count - config.frameLength) / config.hopLength

        var pitches = [Float](repeating: 0, count: nFrames)
        var confidences = [Float](repeating: 0, count: nFrames)

        for frameIdx in 0..<nFrames {
            let start = frameIdx * config.hopLength
            let end = min(start + config.frameLength, paddedSignal.count)

            guard end - start >= config.frameLength else { continue }

            let frame = Array(paddedSignal[start..<end])
            let (pitch, confidence) = detectFrame(frame)

            pitches[frameIdx] = pitch
            confidences[frameIdx] = confidence
        }

        return (pitches, confidences)
    }

    /// Detect pitch for a single frame.
    ///
    /// - Parameter frame: Audio frame (should be frameLength samples).
    /// - Returns: Tuple of (pitch in Hz, confidence 0-1).
    public func detectFrame(_ frame: [Float]) -> (pitch: Float, confidence: Float) {
        let minLag = config.minLag
        let maxLag = min(config.maxLag, frame.count / 2)

        guard maxLag > minLag else {
            return (0, 0)
        }

        // Step 1: Compute difference function
        let diff = computeDifferenceFunction(frame, maxLag: maxLag)

        // Step 2: Compute cumulative mean normalized difference function
        let cmndf = computeCMNDF(diff)

        // Step 3: Find first minimum below threshold
        guard let (lagIndex, minValue) = findPeriod(cmndf, minLag: minLag, maxLag: maxLag) else {
            return (0, 0)
        }

        // Step 4: Parabolic interpolation for sub-sample accuracy
        let refinedLag = parabolicInterpolation(cmndf, index: lagIndex)

        // Step 5: Convert lag to frequency
        let pitch = config.sampleRate / refinedLag

        // Confidence is inverse of CMNDF value at minimum
        let confidence = 1.0 - minValue

        // Validate pitch is within range
        if pitch >= config.fMin && pitch <= config.fMax {
            return (pitch, confidence)
        } else {
            return (0, 0)
        }
    }

    // MARK: - Private Implementation

    /// Compute difference function d(tau).
    private func computeDifferenceFunction(_ frame: [Float], maxLag: Int) -> [Float] {
        let n = frame.count
        var diff = [Float](repeating: 0, count: maxLag)

        // d(tau) = sum_j (x[j] - x[j + tau])^2
        for tau in 1..<maxLag {
            var sum: Float = 0
            for j in 0..<(n - tau) {
                let delta = frame[j] - frame[j + tau]
                sum += delta * delta
            }
            diff[tau] = sum
        }

        return diff
    }

    /// Compute cumulative mean normalized difference function d'(tau).
    private func computeCMNDF(_ diff: [Float]) -> [Float] {
        var cmndf = [Float](repeating: 0, count: diff.count)

        // d'(0) = 1
        cmndf[0] = 1

        var cumulativeSum: Float = 0
        for tau in 1..<diff.count {
            cumulativeSum += diff[tau]

            if cumulativeSum > 0 {
                // d'(tau) = d(tau) / ((1/tau) * sum_{j=1}^{tau} d(j))
                cmndf[tau] = diff[tau] * Float(tau) / cumulativeSum
            } else {
                cmndf[tau] = 1
            }
        }

        return cmndf
    }

    /// Find first minimum in CMNDF below threshold.
    private func findPeriod(_ cmndf: [Float], minLag: Int, maxLag: Int) -> (Int, Float)? {
        // Find first local minimum below threshold
        for tau in minLag..<maxLag {
            if cmndf[tau] < config.threshold {
                // Check if it's a local minimum
                if tau + 1 < cmndf.count && cmndf[tau] <= cmndf[tau + 1] {
                    return (tau, cmndf[tau])
                }
            }
        }

        // If no minimum below threshold, find global minimum
        var minValue: Float = Float.infinity
        var minIndex = minLag

        for tau in minLag..<min(maxLag, cmndf.count) {
            if cmndf[tau] < minValue {
                minValue = cmndf[tau]
                minIndex = tau
            }
        }

        // Only return if reasonably confident
        if minValue < 0.5 {
            return (minIndex, minValue)
        }

        return nil
    }

    /// Apply parabolic interpolation for sub-sample accuracy.
    private func parabolicInterpolation(_ data: [Float], index: Int) -> Float {
        guard index > 0 && index < data.count - 1 else {
            return Float(index)
        }

        let alpha = data[index - 1]
        let beta = data[index]
        let gamma = data[index + 1]

        let denominator = alpha - 2 * beta + gamma
        guard abs(denominator) > 1e-10 else {
            return Float(index)
        }

        let interpolatedOffset = 0.5 * (alpha - gamma) / denominator
        return Float(index) + interpolatedOffset
    }

    /// Pad signal for centered frame analysis.
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
}

// MARK: - Utility Methods

extension PitchDetector {
    /// Convert frequency to MIDI note number.
    ///
    /// - Parameter frequency: Frequency in Hz.
    /// - Returns: MIDI note number (69 = A4 = 440 Hz).
    public static func frequencyToMIDI(_ frequency: Float) -> Float {
        guard frequency > 0 else { return 0 }
        return 69.0 + 12.0 * log2(frequency / 440.0)
    }

    /// Convert MIDI note number to frequency.
    ///
    /// - Parameter midi: MIDI note number.
    /// - Returns: Frequency in Hz.
    public static func midiToFrequency(_ midi: Float) -> Float {
        return 440.0 * pow(2.0, (midi - 69.0) / 12.0)
    }

    /// Convert frequency to note name.
    ///
    /// - Parameter frequency: Frequency in Hz.
    /// - Returns: Note name (e.g., "A4", "C#5").
    public static func frequencyToNoteName(_ frequency: Float) -> String {
        guard frequency > 0 else { return "-" }

        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let midi = frequencyToMIDI(frequency)
        let noteIndex = Int(round(midi)) % 12
        let octave = Int(round(midi)) / 12 - 1

        return "\(noteNames[noteIndex])\(octave)"
    }

    /// Median filter for pitch smoothing.
    ///
    /// - Parameters:
    ///   - pitches: Raw pitch values.
    ///   - windowSize: Filter window size (must be odd).
    /// - Returns: Smoothed pitch values.
    public static func medianFilter(_ pitches: [Float], windowSize: Int = 5) -> [Float] {
        guard !pitches.isEmpty else { return [] }

        let halfWindow = windowSize / 2
        var result = [Float](repeating: 0, count: pitches.count)

        for i in 0..<pitches.count {
            var window = [Float]()

            for j in max(0, i - halfWindow)...min(pitches.count - 1, i + halfWindow) {
                window.append(pitches[j])
            }

            window.sort()
            result[i] = window[window.count / 2]
        }

        return result
    }
}

// MARK: - Presets

extension PitchDetector {
    /// Configuration optimized for speech (80-400 Hz).
    public static var voice: PitchDetector {
        PitchDetector(config: PitchDetectorConfig(
            sampleRate: 16000,
            frameLength: 1024,
            hopLength: 160,
            fMin: 80,
            fMax: 400,
            threshold: 0.1
        ))
    }

    /// Configuration for music with wide frequency range.
    public static var music: PitchDetector {
        PitchDetector(config: PitchDetectorConfig(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65,
            fMax: 2093,
            threshold: 0.1
        ))
    }

    /// Configuration for singing voice (80-1000 Hz).
    public static var singing: PitchDetector {
        PitchDetector(config: PitchDetectorConfig(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 80,
            fMax: 1000,
            threshold: 0.15
        ))
    }

    /// Configuration matching librosa.yin defaults.
    public static var librosaDefault: PitchDetector {
        PitchDetector(config: PitchDetectorConfig(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65,
            fMax: 2093,
            threshold: 0.1
        ))
    }
}
