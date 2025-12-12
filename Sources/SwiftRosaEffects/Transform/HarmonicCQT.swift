import Accelerate
import Foundation
import SwiftRosaCore

/// Configuration for Harmonic CQT.
public struct HarmonicCQTConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Minimum frequency (or note name like "C1").
    public let fMin: Float

    /// Number of frequency bins.
    public let nBins: Int

    /// Bins per octave.
    public let binsPerOctave: Int

    /// Number of harmonics to stack.
    public let nHarmonics: Int

    /// Optional weights per harmonic (nil = equal weights).
    public let harmonicWeights: [Float]?

    /// Filter scale for CQT.
    public let filterScale: Float

    /// Create Harmonic CQT configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - fMin: Minimum frequency (default: 32.70, C1).
    ///   - nBins: Number of frequency bins (default: 84).
    ///   - binsPerOctave: Bins per octave (default: 12).
    ///   - nHarmonics: Number of harmonics to stack (default: 5).
    ///   - harmonicWeights: Optional weights per harmonic.
    ///   - filterScale: Filter scale (default: 1.0).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        nHarmonics: Int = 5,
        harmonicWeights: [Float]? = nil,
        filterScale: Float = 1.0
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.fMin = fMin
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.nHarmonics = max(1, nHarmonics)
        self.harmonicWeights = harmonicWeights
        self.filterScale = filterScale
    }

    /// Create configuration from note name.
    ///
    /// - Parameters:
    ///   - fMinNote: Note name (e.g., "C1", "A4").
    ///   - other: Other parameters as above.
    public init(
        fMinNote: String,
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        nHarmonics: Int = 5,
        harmonicWeights: [Float]? = nil,
        filterScale: Float = 1.0
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.fMin = HarmonicCQTConfig.noteToFrequency(fMinNote)
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.nHarmonics = max(1, nHarmonics)
        self.harmonicWeights = harmonicWeights
        self.filterScale = filterScale
    }

    /// Convert note name to frequency.
    public static func noteToFrequency(_ note: String) -> Float {
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let alternateNames = ["Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"]

        var noteStr = note.uppercased()

        // Handle flats
        for (flat, sharp) in alternateNames {
            if noteStr.hasPrefix(flat) {
                noteStr = noteStr.replacingOccurrences(of: flat, with: sharp)
            }
        }

        // Parse note name and octave
        var noteName = ""
        var octaveStr = ""

        for char in noteStr {
            if char.isLetter || char == "#" {
                noteName.append(char)
            } else if char.isNumber || char == "-" {
                octaveStr.append(char)
            }
        }

        guard let noteIndex = noteNames.firstIndex(of: noteName) else {
            return 32.70  // Default to C1
        }

        let octave = Int(octaveStr) ?? 4

        // A4 = 440 Hz reference
        // MIDI note number: noteIndex + 12 * (octave + 1)
        // C1 = MIDI 24
        let midiNote = noteIndex + 12 * (octave + 1)
        let a4MidiNote = 69  // A4 = MIDI 69

        return 440.0 * pow(2.0, Float(midiNote - a4MidiNote) / 12.0)
    }
}

/// Harmonic Constant-Q Transform.
///
/// Computes multiple CQTs at harmonic intervals and stacks them,
/// providing a multi-resolution view of harmonic content.
///
/// ## Example Usage
///
/// ```swift
/// let hcqt = HarmonicCQT(config: HarmonicCQTConfig(nHarmonics: 6))
///
/// // Get 3D tensor output (harmonics, bins, frames)
/// let tensor = await hcqt.transform(audioSignal)
///
/// // Get flattened 2D output (harmonics*bins, frames)
/// let stacked = await hcqt.stackedMagnitude(audioSignal)
/// ```
public struct HarmonicCQT: Sendable {
    /// Configuration.
    public let config: HarmonicCQTConfig

    /// CQT processors for each harmonic.
    private let harmonicCQTs: [CQT]

    /// Effective weights for each harmonic.
    public let weights: [Float]

    /// Create Harmonic CQT processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: HarmonicCQTConfig = HarmonicCQTConfig()) {
        self.config = config

        // Compute weights
        if let w = config.harmonicWeights, w.count >= config.nHarmonics {
            self.weights = Array(w.prefix(config.nHarmonics))
        } else {
            // Default: equal weights
            self.weights = [Float](repeating: 1.0 / Float(config.nHarmonics), count: config.nHarmonics)
        }

        // Create CQT for each harmonic
        var cqts = [CQT]()
        cqts.reserveCapacity(config.nHarmonics)

        for h in 1...config.nHarmonics {
            let harmonicFMin = config.fMin * Float(h)
            let cqtConfig = CQTConfig(
                sampleRate: config.sampleRate,
                hopLength: config.hopLength,
                fMin: harmonicFMin,
                nBins: config.nBins,
                binsPerOctave: config.binsPerOctave,
                filterScale: config.filterScale
            )
            cqts.append(CQT(config: cqtConfig))
        }

        self.harmonicCQTs = cqts
    }

    /// Compute Harmonic CQT transform.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of CQT results, one per harmonic.
    public func transform(_ signal: [Float]) async -> [ComplexMatrix] {
        var results = [ComplexMatrix]()
        results.reserveCapacity(config.nHarmonics)

        // Compute CQT for each harmonic
        // Note: Could be parallelized with TaskGroup
        for cqt in harmonicCQTs {
            let result = await cqt.transform(signal)
            results.append(result)
        }

        return results
    }

    /// Compute Harmonic CQT magnitude.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of magnitude spectrograms, one per harmonic.
    public func magnitude(_ signal: [Float]) async -> [[[Float]]] {
        let transforms = await transform(signal)
        return transforms.map { $0.magnitude }
    }

    /// Compute weighted sum of harmonic magnitudes.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Weighted magnitude spectrogram of shape (nBins, nFrames).
    public func weightedMagnitude(_ signal: [Float]) async -> [[Float]] {
        let magnitudes = await magnitude(signal)

        guard !magnitudes.isEmpty, !magnitudes[0].isEmpty else { return [] }

        let nBins = magnitudes[0].count
        let nFrames = magnitudes[0][0].count

        var result = [[Float]]()
        result.reserveCapacity(nBins)

        for bin in 0..<nBins {
            var row = [Float](repeating: 0, count: nFrames)

            for (h, mag) in magnitudes.enumerated() {
                let weight = weights[h]
                for frame in 0..<nFrames {
                    row[frame] += weight * mag[bin][frame]
                }
            }

            result.append(row)
        }

        return result
    }

    /// Compute stacked harmonic magnitudes (flattened).
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Stacked magnitude of shape (nHarmonics * nBins, nFrames).
    public func stackedMagnitude(_ signal: [Float]) async -> [[Float]] {
        let magnitudes = await magnitude(signal)

        var stacked = [[Float]]()
        stacked.reserveCapacity(config.nHarmonics * config.nBins)

        for mag in magnitudes {
            stacked.append(contentsOf: mag)
        }

        return stacked
    }

    /// Get center frequencies for the fundamental CQT.
    public var fundamentalFrequencies: [Float] {
        return harmonicCQTs.first?.centerFrequencies ?? []
    }
}

// MARK: - Presets

extension HarmonicCQT {
    /// Standard 5-harmonic CQT.
    public static var standard: HarmonicCQT {
        HarmonicCQT(config: HarmonicCQTConfig(
            sampleRate: 22050,
            fMin: 32.70,  // C1
            nBins: 84,
            binsPerOctave: 12,
            nHarmonics: 5
        ))
    }

    /// 6-harmonic CQT for music transcription.
    public static var transcription: HarmonicCQT {
        HarmonicCQT(config: HarmonicCQTConfig(
            sampleRate: 22050,
            fMin: 27.5,  // A0
            nBins: 88,   // Full piano range
            binsPerOctave: 12,
            nHarmonics: 6,
            harmonicWeights: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]  // Decay
        ))
    }
}
