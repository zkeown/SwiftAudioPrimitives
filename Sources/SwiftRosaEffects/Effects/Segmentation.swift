import Foundation
import SwiftRosaCore

/// Audio segmentation utilities for trimming silence and splitting audio.
///
/// These functions detect non-silent regions based on RMS energy thresholds,
/// matching librosa's `effects.trim` and `effects.split` behavior.
///
/// ## Example Usage
///
/// ```swift
/// // Trim leading/trailing silence
/// let (trimmed, indices) = Segmentation.trim(audio, topDb: 20)
/// print("Audio starts at sample \(indices.0), ends at \(indices.1)")
///
/// // Split into non-silent segments
/// let segments = Segmentation.split(audio, topDb: 30)
/// for (start, end) in segments {
///     let segment = Array(audio[start..<end])
///     // Process each segment...
/// }
/// ```
public struct Segmentation: Sendable {
    private init() {}

    // MARK: - Trim

    /// Trim leading and trailing silence from an audio signal.
    ///
    /// Removes samples from the beginning and end of the signal where the
    /// energy is below a threshold (relative to the peak energy).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - topDb: Threshold in decibels below peak energy (default: 60).
    ///            Higher values = more aggressive trimming.
    ///   - frameLength: Frame length for energy computation (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - ref: Reference value for dB computation. If nil, uses max(abs(signal)).
    /// - Returns: Tuple of (trimmed signal, (start index, end index)).
    ///
    /// - Note: Matches `librosa.effects.trim(y, top_db=60)`
    public static func trim(
        _ signal: [Float],
        topDb: Float = 60,
        frameLength: Int = 2048,
        hopLength: Int = 512,
        ref: Float? = nil
    ) -> (signal: [Float], indices: (Int, Int)) {
        guard !signal.isEmpty else {
            return ([], (0, 0))
        }

        // Compute non-silent intervals
        let intervals = nonSilentIntervals(
            signal,
            topDb: topDb,
            frameLength: frameLength,
            hopLength: hopLength,
            ref: ref
        )

        // If no non-silent regions, return empty
        guard let first = intervals.first, let last = intervals.last else {
            return ([], (0, 0))
        }

        // Get start and end sample indices
        let startSample = first.start
        let endSample = last.end

        // Extract trimmed signal
        let trimmed = Array(signal[startSample..<endSample])

        return (trimmed, (startSample, endSample))
    }

    // MARK: - Split

    /// Split an audio signal into non-silent intervals.
    ///
    /// Returns the sample indices of regions where energy exceeds the threshold.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - topDb: Threshold in decibels below peak energy (default: 60).
    ///   - frameLength: Frame length for energy computation (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - ref: Reference value for dB computation. If nil, uses max(abs(signal)).
    /// - Returns: Array of (start, end) sample index tuples for non-silent regions.
    ///
    /// - Note: Matches `librosa.effects.split(y, top_db=60)`
    public static func split(
        _ signal: [Float],
        topDb: Float = 60,
        frameLength: Int = 2048,
        hopLength: Int = 512,
        ref: Float? = nil
    ) -> [(start: Int, end: Int)] {
        return nonSilentIntervals(
            signal,
            topDb: topDb,
            frameLength: frameLength,
            hopLength: hopLength,
            ref: ref
        )
    }

    // MARK: - Private Helpers

    /// Compute non-silent intervals in a signal.
    private static func nonSilentIntervals(
        _ signal: [Float],
        topDb: Float,
        frameLength: Int,
        hopLength: Int,
        ref: Float?
    ) -> [(start: Int, end: Int)] {
        guard !signal.isEmpty else { return [] }

        // Compute RMS energy for each frame
        let rmsValues = computeRMS(signal, frameLength: frameLength, hopLength: hopLength)

        guard !rmsValues.isEmpty else { return [] }

        // Compute reference (peak RMS or provided ref)
        let reference = ref ?? rmsValues.max() ?? 1.0

        guard reference > 0 else { return [] }

        // Compute threshold
        // In dB: threshold = 20 * log10(rms / ref) >= -topDb
        // Linear: rms >= ref * 10^(-topDb/20)
        let thresholdLinear = reference * pow(10.0, -topDb / 20.0)

        // Find frames above threshold
        var nonSilentFrames = [Bool](repeating: false, count: rmsValues.count)
        for (i, rms) in rmsValues.enumerated() {
            nonSilentFrames[i] = rms >= thresholdLinear
        }

        // Convert frame indices to sample intervals
        var intervals: [(start: Int, end: Int)] = []
        var inInterval = false
        var intervalStart = 0

        for (frameIdx, isNonSilent) in nonSilentFrames.enumerated() {
            if isNonSilent && !inInterval {
                // Start of non-silent interval
                intervalStart = frameIdx * hopLength
                inInterval = true
            } else if !isNonSilent && inInterval {
                // End of non-silent interval
                let intervalEnd = min(frameIdx * hopLength + frameLength, signal.count)
                intervals.append((start: intervalStart, end: intervalEnd))
                inInterval = false
            }
        }

        // Handle case where signal ends in non-silent region
        if inInterval {
            intervals.append((start: intervalStart, end: signal.count))
        }

        // Merge adjacent intervals (within one hop)
        return mergeIntervals(intervals, minGap: hopLength)
    }

    /// Compute RMS energy for each frame.
    private static func computeRMS(
        _ signal: [Float],
        frameLength: Int,
        hopLength: Int
    ) -> [Float] {
        let nFrames = max(1, (signal.count - frameLength) / hopLength + 1)
        var rmsValues = [Float](repeating: 0, count: nFrames)

        for frame in 0..<nFrames {
            let start = frame * hopLength
            let end = min(start + frameLength, signal.count)

            var sumSquares: Float = 0
            for i in start..<end {
                sumSquares += signal[i] * signal[i]
            }

            let frameSize = Float(end - start)
            rmsValues[frame] = sqrt(sumSquares / frameSize)
        }

        return rmsValues
    }

    /// Merge intervals that are within minGap samples of each other.
    private static func mergeIntervals(
        _ intervals: [(start: Int, end: Int)],
        minGap: Int
    ) -> [(start: Int, end: Int)] {
        guard !intervals.isEmpty else { return [] }

        var merged: [(start: Int, end: Int)] = []
        var current = intervals[0]

        for i in 1..<intervals.count {
            let next = intervals[i]

            if next.start - current.end <= minGap {
                // Merge: extend current interval
                current = (start: current.start, end: next.end)
            } else {
                // Gap too large: save current, start new
                merged.append(current)
                current = next
            }
        }

        // Add final interval
        merged.append(current)

        return merged
    }
}

// MARK: - Convenience Extensions

public extension Array where Element == Float {
    /// Trim leading and trailing silence.
    ///
    /// - Parameter topDb: Threshold in dB below peak (default: 60).
    /// - Returns: Trimmed signal.
    func trimmed(topDb: Float = 60) -> [Float] {
        return Segmentation.trim(self, topDb: topDb).signal
    }

    /// Split into non-silent segments.
    ///
    /// - Parameter topDb: Threshold in dB below peak (default: 60).
    /// - Returns: Array of non-silent signal segments.
    func splitOnSilence(topDb: Float = 60) -> [[Float]] {
        let intervals = Segmentation.split(self, topDb: topDb)
        return intervals.map { Array(self[$0.start..<$0.end]) }
    }
}
