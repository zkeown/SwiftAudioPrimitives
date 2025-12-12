import SwiftRosaCore
import SwiftRosaAnalysis
import Accelerate
import Foundation

/// Real-time streaming pitch detector using YIN algorithm.
///
/// `StreamingPitchDetector` maintains an internal ring buffer and detects
/// fundamental frequency (F0) as audio arrives. Push audio samples as they
/// arrive, and pop pitch values as they become available.
///
/// This is suitable for:
/// - Real-time pitch tracking
/// - Live audio tuning applications
/// - Streaming music information retrieval
///
/// ## Example Usage
///
/// ```swift
/// let streaming = await StreamingPitchDetector(sampleRate: 16000)
///
/// // Audio callback
/// func audioCallback(buffer: [Float]) async {
///     await streaming.push(buffer)
///
///     // Get pitch values
///     let pitches = await streaming.popPitches()
///     for pitch in pitches {
///         if pitch > 0 {
///             print("Detected pitch: \(pitch) Hz")
///         }
///     }
/// }
///
/// // Get pitches with confidence
/// let results = await streaming.popPitchesWithConfidence()
/// for (pitch, confidence) in zip(results.pitches, results.confidences) {
///     print("Pitch: \(pitch) Hz, Confidence: \(confidence)")
/// }
/// ```
public actor StreamingPitchDetector {
    /// Ring buffer for incoming samples.
    private var ringBuffer: [Float]

    /// Current write position in ring buffer.
    private var writePosition: Int = 0

    /// Number of samples currently in buffer.
    private var samplesInBuffer: Int = 0

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

    /// Detection threshold.
    public let threshold: Float

    /// Minimum lag for period search.
    private let minLag: Int

    /// Maximum lag for period search.
    private let maxLag: Int

    /// Recent pitch history for median filtering.
    private var pitchHistory: [Float] = []

    /// Median filter window size.
    private let medianFilterSize: Int

    /// Statistics.
    private var totalSamplesReceived: Int = 0
    private var totalFramesOutput: Int = 0

    // MARK: - Initialization

    /// Create a streaming pitch detector.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - frameLength: Frame length in samples (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - fMin: Minimum detectable frequency (default: 65 Hz).
    ///   - fMax: Maximum detectable frequency (default: 2093 Hz).
    ///   - threshold: Detection threshold (default: 0.1).
    ///   - medianFilterSize: Size of median filter for smoothing (default: 5, 0 = disabled).
    public init(
        sampleRate: Float = 22050,
        frameLength: Int = 2048,
        hopLength: Int = 512,
        fMin: Float = 65.0,
        fMax: Float = 2093.0,
        threshold: Float = 0.1,
        medianFilterSize: Int = 5
    ) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.fMin = fMin
        self.fMax = max(fMin + 1, min(fMax, sampleRate / 2))
        self.threshold = threshold
        self.medianFilterSize = medianFilterSize

        self.minLag = max(2, Int(sampleRate / self.fMax))
        self.maxLag = min(frameLength / 2, Int(sampleRate / fMin))

        // Ring buffer sized for at least 4 frame lengths
        let bufferSize = frameLength * 4
        self.ringBuffer = [Float](repeating: 0, count: bufferSize)
    }

    // MARK: - Public Interface

    /// Push new audio samples into the buffer.
    ///
    /// - Parameter samples: Audio samples to add.
    public func push(_ samples: [Float]) {
        for sample in samples {
            ringBuffer[writePosition] = sample
            writePosition = (writePosition + 1) % ringBuffer.count
            samplesInBuffer = min(samplesInBuffer + 1, ringBuffer.count)
        }
        totalSamplesReceived += samples.count
    }

    /// Pop all available pitch values.
    ///
    /// - Returns: Array of pitch values in Hz (0 = unvoiced/undetected).
    public func popPitches() -> [Float] {
        let results = popPitchesWithConfidence()
        return results.pitches
    }

    /// Pop all available pitch values with confidence scores.
    ///
    /// - Returns: Named tuple with pitches and confidences arrays.
    public func popPitchesWithConfidence() -> (pitches: [Float], confidences: [Float]) {
        var pitches: [Float] = []
        var confidences: [Float] = []

        while samplesInBuffer >= frameLength {
            let (pitch, confidence) = computeNextPitch()
            pitches.append(pitch)
            confidences.append(confidence)

            // Store for median filtering
            if medianFilterSize > 0 {
                pitchHistory.append(pitch)
                if pitchHistory.count > medianFilterSize * 2 {
                    pitchHistory.removeFirst()
                }
            }

            totalFramesOutput += 1
            samplesInBuffer -= hopLength
        }

        return (pitches, confidences)
    }

    /// Pop pitches with median filtering applied.
    ///
    /// - Returns: Smoothed pitch values.
    public func popPitchesSmoothed() -> [Float] {
        let rawPitches = popPitches()
        guard medianFilterSize > 0 && !rawPitches.isEmpty else {
            return rawPitches
        }

        return applyMedianFilter(rawPitches)
    }

    /// Pop pitch with MIDI note information.
    ///
    /// - Returns: Array of tuples (pitch, midiNote, noteName, confidence).
    public func popPitchesWithNotes() -> [(pitch: Float, midiNote: Float, noteName: String, confidence: Float)] {
        let results = popPitchesWithConfidence()
        var output: [(pitch: Float, midiNote: Float, noteName: String, confidence: Float)] = []

        for (pitch, confidence) in zip(results.pitches, results.confidences) {
            let midiNote = PitchDetector.frequencyToMIDI(pitch)
            let noteName = PitchDetector.frequencyToNoteName(pitch)
            output.append((pitch, midiNote, noteName, confidence))
        }

        return output
    }

    /// Get the most recent pitch value (for display).
    ///
    /// - Returns: Most recent pitch in Hz, or nil if none available.
    public func currentPitch() -> Float? {
        guard samplesInBuffer >= frameLength else { return nil }

        let frame = extractSamples(count: frameLength)
        let (pitch, _) = detectPitch(frame)

        return pitch > 0 ? pitch : nil
    }

    /// Get current pitch with note name.
    ///
    /// - Returns: Tuple of (pitch, noteName) or nil.
    public func currentNote() -> (pitch: Float, noteName: String)? {
        guard let pitch = currentPitch(), pitch > 0 else { return nil }
        return (pitch, PitchDetector.frequencyToNoteName(pitch))
    }

    /// Number of complete frames available.
    public var availableFrameCount: Int {
        max(0, (samplesInBuffer - frameLength) / hopLength + 1)
    }

    /// Number of samples currently buffered.
    public var bufferedSamples: Int {
        samplesInBuffer
    }

    /// Minimum samples needed before a pitch can be detected.
    public var samplesNeededForFrame: Int {
        max(0, frameLength - samplesInBuffer)
    }

    /// Reset internal state.
    public func reset() {
        ringBuffer = [Float](repeating: 0, count: ringBuffer.count)
        writePosition = 0
        samplesInBuffer = 0
        totalSamplesReceived = 0
        totalFramesOutput = 0
        pitchHistory = []
    }

    /// Flush remaining samples with zero-padding.
    ///
    /// - Returns: Array of remaining pitch values.
    public func flush() -> [Float] {
        var pitches: [Float] = []

        // Process complete frames
        pitches.append(contentsOf: popPitches())

        // Process remaining samples with zero-padding
        if samplesInBuffer > 0 {
            var remaining = extractSamples(count: samplesInBuffer)

            if remaining.count < frameLength {
                remaining.append(contentsOf: [Float](repeating: 0, count: frameLength - remaining.count))
            }

            let (pitch, _) = detectPitch(remaining)
            pitches.append(pitch)
            samplesInBuffer = 0
        }

        return pitches
    }

    // MARK: - Statistics

    /// Statistics about the streaming processor.
    public struct Stats: Sendable {
        public let totalSamplesReceived: Int
        public let totalFramesOutput: Int
        public let currentBufferLevel: Int
        public let bufferCapacity: Int
    }

    /// Get current statistics.
    public var stats: Stats {
        Stats(
            totalSamplesReceived: totalSamplesReceived,
            totalFramesOutput: totalFramesOutput,
            currentBufferLevel: samplesInBuffer,
            bufferCapacity: ringBuffer.count
        )
    }

    // MARK: - Private Implementation

    /// Compute the next pitch from the buffer.
    private func computeNextPitch() -> (Float, Float) {
        let frame = extractSamples(count: frameLength)
        return detectPitch(frame)
    }

    /// Extract samples from ring buffer (oldest first).
    private func extractSamples(count: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: count)
        let readPosition = (writePosition - samplesInBuffer + ringBuffer.count) % ringBuffer.count

        for i in 0..<count {
            let idx = (readPosition + i) % ringBuffer.count
            samples[i] = ringBuffer[idx]
        }

        return samples
    }

    /// Detect pitch for a single frame using YIN algorithm.
    private func detectPitch(_ frame: [Float]) -> (pitch: Float, confidence: Float) {
        guard maxLag > minLag else {
            return (0, 0)
        }

        // Step 1: Compute difference function
        let diff = computeDifferenceFunction(frame)

        // Step 2: Compute cumulative mean normalized difference function
        let cmndf = computeCMNDF(diff)

        // Step 3: Find first minimum below threshold
        guard let (lagIndex, minValue) = findPeriod(cmndf) else {
            return (0, 0)
        }

        // Step 4: Parabolic interpolation for sub-sample accuracy
        let refinedLag = parabolicInterpolation(cmndf, index: lagIndex)

        // Step 5: Convert lag to frequency
        let pitch = sampleRate / refinedLag
        let confidence = 1.0 - minValue

        // Validate pitch is within range
        if pitch >= fMin && pitch <= fMax {
            return (pitch, confidence)
        } else {
            return (0, 0)
        }
    }

    /// Compute difference function d(tau).
    private func computeDifferenceFunction(_ frame: [Float]) -> [Float] {
        let n = frame.count
        var diff = [Float](repeating: 0, count: maxLag)

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
        cmndf[0] = 1

        var cumulativeSum: Float = 0
        for tau in 1..<diff.count {
            cumulativeSum += diff[tau]

            if cumulativeSum > 0 {
                cmndf[tau] = diff[tau] * Float(tau) / cumulativeSum
            } else {
                cmndf[tau] = 1
            }
        }

        return cmndf
    }

    /// Find first minimum in CMNDF below threshold.
    private func findPeriod(_ cmndf: [Float]) -> (Int, Float)? {
        // Find first local minimum below threshold
        for tau in minLag..<min(maxLag, cmndf.count) {
            if cmndf[tau] < threshold {
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

    /// Apply median filter to pitches.
    private func applyMedianFilter(_ pitches: [Float]) -> [Float] {
        guard medianFilterSize > 0 else { return pitches }

        let halfWindow = medianFilterSize / 2
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

extension StreamingPitchDetector {
    /// Configuration optimized for speech (80-400 Hz).
    public static func voice() async -> StreamingPitchDetector {
        StreamingPitchDetector(
            sampleRate: 16000,
            frameLength: 1024,
            hopLength: 160,
            fMin: 80,
            fMax: 400,
            threshold: 0.1,
            medianFilterSize: 5
        )
    }

    /// Configuration for music with wide frequency range.
    public static func music() async -> StreamingPitchDetector {
        StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65,
            fMax: 2093,
            threshold: 0.1,
            medianFilterSize: 5
        )
    }

    /// Configuration for singing voice (80-1000 Hz).
    public static func singing() async -> StreamingPitchDetector {
        StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 80,
            fMax: 1000,
            threshold: 0.15,
            medianFilterSize: 5
        )
    }

    /// Low-latency configuration for real-time tuners.
    public static func tuner() async -> StreamingPitchDetector {
        StreamingPitchDetector(
            sampleRate: 44100,
            frameLength: 1024,
            hopLength: 256,
            fMin: 65,
            fMax: 1000,
            threshold: 0.1,
            medianFilterSize: 3
        )
    }

    /// Process a complete signal (for testing).
    public func processComplete(_ signal: [Float]) -> [Float] {
        reset()
        push(signal)
        return flush()
    }
}
