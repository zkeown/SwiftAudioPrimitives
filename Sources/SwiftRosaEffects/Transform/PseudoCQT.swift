import Accelerate
import Foundation
import SwiftRosaCore

/// Configuration for Pseudo-CQT.
public struct PseudoCQTConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// FFT size (larger = better frequency resolution for low frequencies).
    public let nFFT: Int

    /// Minimum frequency in Hz.
    public let fMin: Float

    /// Number of frequency bins.
    public let nBins: Int

    /// Bins per octave.
    public let binsPerOctave: Int

    /// Window type.
    public let windowType: WindowType

    /// Create Pseudo-CQT configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - nFFT: FFT size (default: 4096 for good low-frequency resolution).
    ///   - fMin: Minimum frequency (default: 32.70, C1).
    ///   - nBins: Number of frequency bins (default: 84, 7 octaves).
    ///   - binsPerOctave: Bins per octave (default: 12, semitones).
    ///   - windowType: Window type (default: hann).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        nFFT: Int = 4096,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        windowType: WindowType = .hann
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.nFFT = nFFT
        self.fMin = fMin
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.windowType = windowType
    }

    /// Maximum frequency based on nBins and binsPerOctave.
    public var fMax: Float {
        fMin * pow(2.0, Float(nBins) / Float(binsPerOctave))
    }
}

/// STFT-based approximation of Constant-Q Transform.
///
/// Pseudo-CQT computes a logarithmically-spaced frequency representation
/// by mapping STFT bins to CQT bins. It's faster than true CQT but less
/// accurate for low frequencies where the STFT has limited resolution.
///
/// ## Example Usage
///
/// ```swift
/// let pcqt = PseudoCQT(config: PseudoCQTConfig(nFFT: 8192))
///
/// // Get CQT-like representation
/// let cqtMagnitude = await pcqt.magnitude(audioSignal)
/// ```
///
/// ## Advantages over True CQT
///
/// - Faster computation (single FFT size)
/// - Lower memory usage
/// - Better for high frequencies
///
/// ## Disadvantages
///
/// - Less accurate for low frequencies
/// - Fixed time resolution (unlike true CQT's variable resolution)
public struct PseudoCQT: Sendable {
    /// Configuration.
    public let config: PseudoCQTConfig

    /// STFT processor.
    private let stft: STFT

    /// Pre-computed bin mapping from STFT to CQT.
    /// Each entry: (startBin, endBin, weights)
    private let binMapping: [(start: Int, end: Int, weights: [Float])]

    /// Center frequencies for each CQT bin.
    public let centerFrequencies: [Float]

    /// Create Pseudo-CQT processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: PseudoCQTConfig = PseudoCQTConfig()) {
        self.config = config

        // Create STFT with specified configuration
        self.stft = STFT(config: STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.nFFT,
            windowType: config.windowType,
            center: true,
            padMode: .reflect
        ))

        // Compute center frequencies
        var freqs = [Float](repeating: 0, count: config.nBins)
        for bin in 0..<config.nBins {
            freqs[bin] = config.fMin * pow(2.0, Float(bin) / Float(config.binsPerOctave))
        }
        self.centerFrequencies = freqs

        // Compute bin mapping
        self.binMapping = PseudoCQT.computeBinMapping(config: config, centerFrequencies: freqs)
    }

    /// Compute Pseudo-CQT transform.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex CQT representation of shape (nBins, nFrames).
    public func transform(_ signal: [Float]) async -> ComplexMatrix {
        let stftResult = await stft.transform(signal)

        guard !stftResult.real.isEmpty, !stftResult.real[0].isEmpty else {
            return ComplexMatrix(real: [], imag: [])
        }

        let nFrames = stftResult.real[0].count

        var realResult = [[Float]]()
        var imagResult = [[Float]]()
        realResult.reserveCapacity(config.nBins)
        imagResult.reserveCapacity(config.nBins)

        // Map STFT bins to CQT bins
        for binIdx in 0..<config.nBins {
            var realRow = [Float](repeating: 0, count: nFrames)
            var imagRow = [Float](repeating: 0, count: nFrames)

            let mapping = binMapping[binIdx]

            for frameIdx in 0..<nFrames {
                var sumReal: Float = 0
                var sumImag: Float = 0
                var weightIdx = 0

                for stftBin in mapping.start..<mapping.end {
                    if stftBin < stftResult.real.count {
                        sumReal += mapping.weights[weightIdx] * stftResult.real[stftBin][frameIdx]
                        sumImag += mapping.weights[weightIdx] * stftResult.imag[stftBin][frameIdx]
                    }
                    weightIdx += 1
                }

                realRow[frameIdx] = sumReal
                imagRow[frameIdx] = sumImag
            }

            realResult.append(realRow)
            imagResult.append(imagRow)
        }

        return ComplexMatrix(real: realResult, imag: imagResult)
    }

    /// Compute Pseudo-CQT magnitude.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nBins, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let cqt = await transform(signal)
        return cqt.magnitude
    }

    /// Compute Pseudo-CQT power (magnitude squared).
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram of shape (nBins, nFrames).
    public func power(_ signal: [Float]) async -> [[Float]] {
        let mag = await magnitude(signal)

        return mag.map { row in
            var squared = [Float](repeating: 0, count: row.count)
            vDSP_vsq(row, 1, &squared, 1, vDSP_Length(row.count))
            return squared
        }
    }

    /// Get the frequency for a specific CQT bin.
    ///
    /// - Parameter bin: CQT bin index.
    /// - Returns: Center frequency in Hz.
    public func frequency(forBin bin: Int) -> Float {
        guard bin >= 0 && bin < config.nBins else { return 0 }
        return centerFrequencies[bin]
    }

    /// Get the note name for a specific CQT bin.
    ///
    /// - Parameter bin: CQT bin index.
    /// - Returns: Note name (e.g., "C4", "A#3").
    public func noteName(forBin bin: Int) -> String {
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let noteInOctave = bin % config.binsPerOctave
        let octave = bin / config.binsPerOctave + 1  // Assuming fMin is C1
        return "\(noteNames[noteInOctave])\(octave)"
    }

    // MARK: - Private Implementation

    /// Compute mapping from STFT bins to CQT bins.
    private static func computeBinMapping(
        config: PseudoCQTConfig,
        centerFrequencies: [Float]
    ) -> [(start: Int, end: Int, weights: [Float])] {
        let binWidth = config.sampleRate / Float(config.nFFT)
        let nSTFTBins = config.nFFT / 2 + 1

        var mapping = [(start: Int, end: Int, weights: [Float])]()
        mapping.reserveCapacity(config.nBins)

        for (_, centerFreq) in centerFrequencies.enumerated() {
            // Compute Q factor and bandwidth
            let Q = Float(config.binsPerOctave) / (pow(2.0, 1.0 / Float(config.binsPerOctave)) - 1)
            let bandwidth = centerFreq / Q

            // Find STFT bins within the bandwidth
            let lowerFreq = centerFreq - bandwidth / 2
            let upperFreq = centerFreq + bandwidth / 2

            let startBin = max(0, Int(floor(lowerFreq / binWidth)))
            let endBin = min(nSTFTBins, Int(ceil(upperFreq / binWidth)) + 1)

            // Compute triangular weights
            var weights = [Float]()
            weights.reserveCapacity(endBin - startBin)

            for stftBin in startBin..<endBin {
                let stftFreq = Float(stftBin) * binWidth
                let distance = abs(stftFreq - centerFreq)
                let weight = max(0, 1.0 - distance / (bandwidth / 2))
                weights.append(weight)
            }

            // Normalize weights
            let weightSum = weights.reduce(0, +)
            if weightSum > 0 {
                for i in 0..<weights.count {
                    weights[i] /= weightSum
                }
            }

            mapping.append((start: startBin, end: endBin, weights: weights))
        }

        return mapping
    }
}

// MARK: - Presets

extension PseudoCQT {
    /// Standard Pseudo-CQT configuration.
    public static var standard: PseudoCQT {
        PseudoCQT(config: PseudoCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nFFT: 4096,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12
        ))
    }

    /// High-resolution Pseudo-CQT for better low-frequency accuracy.
    public static var highResolution: PseudoCQT {
        PseudoCQT(config: PseudoCQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nFFT: 8192,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12
        ))
    }
}
