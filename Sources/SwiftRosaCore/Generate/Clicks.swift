import Accelerate
import Foundation

/// Click track generator.
///
/// Generates click sounds at specified times or frame positions.
/// Useful for beat tracking visualization and metronome generation.
///
/// ## Example Usage
///
/// ```swift
/// // Generate clicks at specific times
/// let clicks = ClickGenerator.clicks(times: [0.5, 1.0, 1.5, 2.0], sampleRate: 22050, length: 44100)
///
/// // Generate clicks at frame positions
/// let clicks = ClickGenerator.clicks(frames: [0, 100, 200], sampleRate: 22050, hopLength: 512, length: 22050)
///
/// // Customize click sound
/// let clicks = ClickGenerator.clicks(
///     times: beatTimes,
///     sampleRate: 22050,
///     length: totalSamples,
///     clickFrequency: 880,
///     clickDuration: 0.05
/// )
/// ```
public struct ClickGenerator {
    private init() {}

    /// Generate clicks at specified times.
    ///
    /// - Parameters:
    ///   - times: Click times in seconds (optional, specify either times or frames).
    ///   - frames: Click frame indices (optional, specify either times or frames).
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length for frame-to-sample conversion (default: 512).
    ///   - length: Total output length in samples (optional, auto-computed if nil).
    ///   - clickFrequency: Frequency of click tone in Hz (default: 1000).
    ///   - clickDuration: Duration of each click in seconds (default: 0.1).
    ///   - gain: Click amplitude (default: 1.0).
    /// - Returns: Generated click track.
    public static func clicks(
        times: [Float]? = nil,
        frames: [Int]? = nil,
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        length: Int? = nil,
        clickFrequency: Float = 1000,
        clickDuration: Float = 0.1,
        gain: Float = 1.0
    ) -> [Float] {
        // Convert times or frames to sample positions
        let clickPositions: [Int]

        if let t = times {
            clickPositions = t.map { Int($0 * sampleRate) }
        } else if let f = frames {
            clickPositions = f.map { $0 * hopLength }
        } else {
            return []
        }

        guard !clickPositions.isEmpty else { return [] }

        // Determine output length
        let outputLength: Int
        if let len = length {
            outputLength = len
        } else {
            // Auto-compute: last click position + click duration
            let clickSamples = Int(clickDuration * sampleRate)
            outputLength = (clickPositions.max() ?? 0) + clickSamples
        }

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        // Generate single click template
        let clickSamples = Int(clickDuration * sampleRate)
        let clickTemplate = generateClick(
            frequency: clickFrequency,
            sampleRate: sampleRate,
            length: clickSamples,
            gain: gain
        )

        // Place clicks at each position
        for pos in clickPositions {
            guard pos >= 0 && pos < outputLength else { continue }

            let endPos = min(pos + clickSamples, outputLength)
            let actualLength = endPos - pos

            // Add click to output (may be truncated at end)
            for i in 0..<actualLength {
                output[pos + i] += clickTemplate[i]
            }
        }

        // Clip to prevent overflow when clicks overlap
        var minVal: Float = -1.0
        var maxVal: Float = 1.0
        vDSP_vclip(output, 1, &minVal, &maxVal, &output, 1, vDSP_Length(outputLength))

        return output
    }

    /// Generate a metronome click track at a given BPM.
    ///
    /// - Parameters:
    ///   - bpm: Beats per minute.
    ///   - duration: Duration in seconds.
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - clickFrequency: Frequency of click tone in Hz (default: 1000).
    ///   - clickDuration: Duration of each click in seconds (default: 0.05).
    ///   - accentFirst: Accent the first beat of each measure (default: false).
    ///   - beatsPerMeasure: Beats per measure for accenting (default: 4).
    /// - Returns: Generated metronome track.
    public static func metronome(
        bpm: Float,
        duration: Float,
        sampleRate: Float = 22050,
        clickFrequency: Float = 1000,
        clickDuration: Float = 0.05,
        accentFirst: Bool = false,
        beatsPerMeasure: Int = 4
    ) -> [Float] {
        let beatInterval = 60.0 / bpm  // seconds per beat
        let numBeats = Int(duration / beatInterval) + 1

        var times = [Float]()
        times.reserveCapacity(numBeats)

        for i in 0..<numBeats {
            let time = Float(i) * beatInterval
            if time < duration {
                times.append(time)
            }
        }

        let outputLength = Int(duration * sampleRate)

        if accentFirst {
            // Generate with accents
            var output = [Float](repeating: 0, count: outputLength)
            let clickSamples = Int(clickDuration * sampleRate)

            for (i, time) in times.enumerated() {
                let pos = Int(time * sampleRate)
                guard pos >= 0 && pos < outputLength else { continue }

                let isAccent = (i % beatsPerMeasure) == 0
                let freq = isAccent ? clickFrequency * 2 : clickFrequency
                let gain: Float = isAccent ? 1.0 : 0.7

                let click = generateClick(
                    frequency: freq,
                    sampleRate: sampleRate,
                    length: clickSamples,
                    gain: gain
                )

                let endPos = min(pos + clickSamples, outputLength)
                for j in 0..<(endPos - pos) {
                    output[pos + j] += click[j]
                }
            }

            return output
        } else {
            return clicks(
                times: times,
                sampleRate: sampleRate,
                length: outputLength,
                clickFrequency: clickFrequency,
                clickDuration: clickDuration
            )
        }
    }

    // MARK: - Private Implementation

    private static func generateClick(
        frequency: Float,
        sampleRate: Float,
        length: Int,
        gain: Float
    ) -> [Float] {
        guard length > 0 else { return [] }

        var click = [Float](repeating: 0, count: length)

        // Generate decaying sine burst
        let phaseIncrement = 2.0 * Float.pi * frequency / sampleRate

        for i in 0..<length {
            let phase = Float(i) * phaseIncrement
            let envelope = 1.0 - Float(i) / Float(length)  // Linear decay
            click[i] = gain * envelope * sin(phase)
        }

        return click
    }
}
