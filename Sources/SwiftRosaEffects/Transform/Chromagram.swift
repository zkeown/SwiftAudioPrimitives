import SwiftRosaCore
import Accelerate
import Foundation

/// Normalization mode for chromagram.
public enum ChromaNorm: Sendable {
    /// No normalization.
    case none
    /// L1 normalization (sum to 1).
    case l1
    /// L2 normalization (unit norm).
    case l2
    /// Max normalization (max = 1).
    case max
}

/// Configuration for chromagram extraction.
public struct ChromagramConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Number of chroma bins (default: 12 for Western music).
    public let nChroma: Int

    /// Use CQT-based chromagram (true) or STFT-based (false).
    public let useCQT: Bool

    /// Normalization mode.
    public let norm: ChromaNorm

    /// Tuning offset in fractional bins.
    public let tuning: Float

    // STFT-based parameters
    /// FFT size (for STFT-based chromagram).
    public let nFFT: Int

    /// Window type.
    public let windowType: WindowType

    // CQT-based parameters
    /// Number of octaves (for CQT-based chromagram).
    public let nOctaves: Int

    /// Bins per octave (for CQT-based chromagram).
    public let binsPerOctave: Int

    /// Create chromagram configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - nChroma: Number of chroma bins (default: 12).
    ///   - useCQT: Use CQT-based extraction (default: true).
    ///   - norm: Normalization mode (default: l2).
    ///   - tuning: Tuning offset (default: 0).
    ///   - nFFT: FFT size for STFT (default: 2048).
    ///   - windowType: Window type (default: hann).
    ///   - nOctaves: Octaves for CQT (default: 7).
    ///   - binsPerOctave: Bins per octave for CQT (default: 36).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        nChroma: Int = 12,
        useCQT: Bool = true,
        norm: ChromaNorm = .l2,
        tuning: Float = 0.0,
        nFFT: Int = 2048,
        windowType: WindowType = .hann,
        nOctaves: Int = 7,
        binsPerOctave: Int = 36
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.nChroma = nChroma
        self.useCQT = useCQT
        self.norm = norm
        self.tuning = tuning
        self.nFFT = nFFT
        self.windowType = windowType
        self.nOctaves = nOctaves
        self.binsPerOctave = binsPerOctave
    }
}

/// Chromagram (pitch-class profile) extractor.
///
/// A chromagram represents the intensity of each of the 12 pitch classes
/// (C, C#, D, ..., B) over time, regardless of octave. This makes it useful
/// for chord recognition, key detection, and cover song identification.
///
/// ## Example Usage
///
/// ```swift
/// let chromagram = Chromagram(config: ChromagramConfig())
///
/// // Extract chromagram
/// let chroma = await chromagram.transform(audioSignal)
/// // chroma has shape (12, nFrames)
///
/// // From pre-computed CQT
/// let chromaFromCQT = chromagram.fromCQT(cqtMagnitude)
/// ```
public struct Chromagram: Sendable {
    /// Configuration.
    public let config: ChromagramConfig

    /// CQT processor (for CQT-based chromagram).
    private let cqt: CQT?

    /// STFT processor (for STFT-based chromagram).
    private let stft: STFT?

    /// Chroma filterbank for STFT-based extraction.
    private let chromaFilterbank: [[Float]]?

    /// Pre-flattened filterbank for vDSP_mmul (nChroma × nFreqs row-major).
    private let flatFilterbank: [Float]?

    /// Number of frequency bins for STFT-based chromagram.
    private let nFreqs: Int

    /// Create a chromagram extractor.
    ///
    /// - Parameter config: Configuration.
    public init(config: ChromagramConfig = ChromagramConfig()) {
        self.config = config

        if config.useCQT {
            // CQT-based chromagram
            self.cqt = CQT(config: CQTConfig(
                sampleRate: config.sampleRate,
                hopLength: config.hopLength,
                fMin: 32.70,  // C1
                nBins: config.nOctaves * config.binsPerOctave,
                binsPerOctave: config.binsPerOctave,
                windowType: config.windowType,
                filterScale: 1.0,
                sparsity: 0.01
            ))
            self.stft = nil
            self.chromaFilterbank = nil
            self.flatFilterbank = nil
            self.nFreqs = 0
        } else {
            // STFT-based chromagram
            self.cqt = nil
            let nFreqBins = config.nFFT / 2 + 1
            self.nFreqs = nFreqBins
            self.stft = STFT(config: STFTConfig(
                uncheckedNFFT: config.nFFT,
                hopLength: config.hopLength,
                winLength: config.nFFT,
                windowType: config.windowType,
                center: true,
                padMode: .reflect
            ))
            let filterbank = Self.computeChromaFilterbank(
                sampleRate: config.sampleRate,
                nFFT: config.nFFT,
                nChroma: config.nChroma,
                tuning: config.tuning
            )
            self.chromaFilterbank = filterbank

            // Pre-flatten filterbank for vDSP_mmul: (nChroma × nFreqs) row-major
            var flat = [Float](repeating: 0, count: config.nChroma * nFreqBins)
            for c in 0..<config.nChroma {
                for f in 0..<min(nFreqBins, filterbank[c].count) {
                    flat[c * nFreqBins + f] = filterbank[c][f]
                }
            }
            self.flatFilterbank = flat
        }
    }

    /// Compute chromagram from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Chromagram of shape (nChroma, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        if config.useCQT, let cqt = cqt {
            let cqtMag = await cqt.magnitude(signal)
            return fromCQT(cqtMag)
        } else if let stft = stft {
            let stftMag = await stft.magnitude(signal)
            return fromSTFT(stftMag)
        } else {
            return []
        }
    }

    /// Compute chromagram from pre-computed CQT magnitude.
    ///
    /// Uses the same CQT-to-chroma mapping as librosa.filters.cq_to_chroma:
    /// - Merges `binsPerOctave / nChroma` CQT bins per chroma bin
    /// - Applies proper centering and rotation based on fmin
    ///
    /// - Parameter cqtMagnitude: CQT magnitude of shape (nBins, nFrames).
    /// - Returns: Chromagram of shape (nChroma, nFrames).
    public func fromCQT(_ cqtMagnitude: [[Float]]) -> [[Float]] {
        guard !cqtMagnitude.isEmpty && !cqtMagnitude[0].isEmpty else { return [] }

        let nBins = cqtMagnitude.count
        let nFrames = cqtMagnitude[0].count

        // Build the CQT-to-chroma mapping matrix (matches librosa.filters.cq_to_chroma)
        let cqToChroma = Self.computeCQToChromaMatrix(
            nInput: nBins,
            binsPerOctave: config.binsPerOctave,
            nChroma: config.nChroma,
            fmin: 32.70  // C1, the default in Swift CQT
        )

        // Initialize chroma
        var chroma = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nChroma)

        // Apply the transformation: chroma = cqToChroma @ cqtMagnitude
        for chromaIdx in 0..<config.nChroma {
            for frame in 0..<nFrames {
                var sum: Float = 0
                for bin in 0..<nBins {
                    sum += cqToChroma[chromaIdx][bin] * cqtMagnitude[bin][frame]
                }
                chroma[chromaIdx][frame] = sum
            }
        }

        // Normalize
        return normalizeChroma(chroma)
    }

    /// Compute the CQT-to-chroma transformation matrix.
    ///
    /// Matches librosa.filters.cq_to_chroma exactly:
    /// 1. Create identity matrix repeated for each merge group
    /// 2. Roll to center on target bin
    /// 3. Tile for all octaves
    /// 4. Rotate based on fmin (so C1 maps to chroma bin 0 = C)
    ///
    /// - Parameters:
    ///   - nInput: Number of CQT bins
    ///   - binsPerOctave: CQT bins per octave
    ///   - nChroma: Number of output chroma bins
    ///   - fmin: Minimum frequency of CQT (default C1 = 32.70 Hz)
    /// - Returns: Transformation matrix of shape (nChroma, nInput)
    private static func computeCQToChromaMatrix(
        nInput: Int,
        binsPerOctave: Int,
        nChroma: Int,
        fmin: Float
    ) -> [[Float]] {
        // How many CQT bins merge into each chroma bin
        let nMerge = binsPerOctave / nChroma

        // Build one octave of the mapping: (nChroma x binsPerOctave)
        // Start with identity repeated for each merge group
        var oneOctave = [[Float]](
            repeating: [Float](repeating: 0, count: binsPerOctave),
            count: nChroma
        )

        for chromaIdx in 0..<nChroma {
            for mergeIdx in 0..<nMerge {
                let binIdx = chromaIdx * nMerge + mergeIdx
                if binIdx < binsPerOctave {
                    oneOctave[chromaIdx][binIdx] = 1.0
                }
            }
        }

        // Roll left by nMerge/2 to center on target bin
        let rollAmount = nMerge / 2
        for chromaIdx in 0..<nChroma {
            var rolled = [Float](repeating: 0, count: binsPerOctave)
            for i in 0..<binsPerOctave {
                let srcIdx = (i + rollAmount) % binsPerOctave
                rolled[i] = oneOctave[chromaIdx][srcIdx]
            }
            oneOctave[chromaIdx] = rolled
        }

        // Compute number of octaves and tile
        let nOctaves = Int(ceil(Float(nInput) / Float(binsPerOctave)))

        // Build full matrix by tiling and trimming
        var fullMatrix = [[Float]](
            repeating: [Float](repeating: 0, count: nInput),
            count: nChroma
        )

        for chromaIdx in 0..<nChroma {
            for octave in 0..<nOctaves {
                for binInOctave in 0..<binsPerOctave {
                    let globalBin = octave * binsPerOctave + binInOctave
                    if globalBin < nInput {
                        fullMatrix[chromaIdx][globalBin] = oneOctave[chromaIdx][binInOctave]
                    }
                }
            }
        }

        // Compute rotation based on fmin
        // MIDI note number mod 12 tells us which chroma bin fmin corresponds to
        let midiNote = 12.0 * log2(fmin / 440.0) + 69.0
        let midi0 = midiNote.truncatingRemainder(dividingBy: 12.0)
        let roll = Int(round(midi0 * Float(nChroma) / 12.0)) % nChroma

        // Apply the roll to rows (rotate which chroma bin each CQT bin maps to)
        if roll != 0 {
            var rotated = [[Float]](
                repeating: [Float](repeating: 0, count: nInput),
                count: nChroma
            )
            for chromaIdx in 0..<nChroma {
                let srcIdx = (chromaIdx - roll + nChroma) % nChroma
                rotated[chromaIdx] = fullMatrix[srcIdx]
            }
            fullMatrix = rotated
        }

        return fullMatrix
    }

    /// Compute chromagram from pre-computed STFT magnitude.
    ///
    /// - Parameter stftMagnitude: STFT magnitude of shape (nFreqs, nFrames).
    /// - Returns: Chromagram of shape (nChroma, nFrames).
    public func fromSTFT(_ stftMagnitude: [[Float]]) -> [[Float]] {
        guard !stftMagnitude.isEmpty && !stftMagnitude[0].isEmpty else { return [] }
        guard let flatFB = flatFilterbank else { return [] }

        let inputNFreqs = stftMagnitude.count
        let nFrames = stftMagnitude[0].count
        let nChroma = config.nChroma

        // OPTIMIZATION: Use vDSP_mmul for matrix multiplication instead of triple loop
        // chroma (nChroma × nFrames) = filterbank (nChroma × nFreqs) × magnitude (nFreqs × nFrames)

        // Flatten magnitude to (nFreqs × nFrames) row-major
        var flatMag = [Float](repeating: 0, count: inputNFreqs * nFrames)
        for f in 0..<inputNFreqs {
            let row = stftMagnitude[f]
            let destOffset = f * nFrames
            row.withUnsafeBufferPointer { rowPtr in
                memcpy(&flatMag[destOffset], rowPtr.baseAddress!, min(nFrames, row.count) * MemoryLayout<Float>.size)
            }
        }

        // Output: (nChroma × nFrames) row-major
        var flatResult = [Float](repeating: 0, count: nChroma * nFrames)

        // Matrix multiply: chroma = filterbank × magnitude
        // filterbank: (nChroma × nFreqs), magnitude: (nFreqs × nFrames) -> result: (nChroma × nFrames)
        flatFB.withUnsafeBufferPointer { fbPtr in
            flatMag.withUnsafeBufferPointer { magPtr in
                flatResult.withUnsafeMutableBufferPointer { resultPtr in
                    vDSP_mmul(
                        fbPtr.baseAddress!, 1,      // A: filterbank (nChroma × nFreqs)
                        magPtr.baseAddress!, 1,     // B: magnitude (nFreqs × nFrames)
                        resultPtr.baseAddress!, 1,  // C: result (nChroma × nFrames)
                        vDSP_Length(nChroma),       // M: rows of A and C
                        vDSP_Length(nFrames),       // N: cols of B and C
                        vDSP_Length(nFreqs)         // K: cols of A, rows of B
                    )
                }
            }
        }

        // Reshape to [[Float]]
        var chroma = [[Float]]()
        chroma.reserveCapacity(nChroma)
        for c in 0..<nChroma {
            let start = c * nFrames
            chroma.append(Array(flatResult[start..<(start + nFrames)]))
        }

        // Normalize
        return normalizeChroma(chroma)
    }

    // MARK: - Private Implementation

    private static func computeChromaFilterbank(
        sampleRate: Float,
        nFFT: Int,
        nChroma: Int,
        tuning: Float
    ) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1

        // Frequency for each FFT bin
        var frequencies = [Float](repeating: 0, count: nFreqs)
        let binWidth = sampleRate / Float(nFFT)
        for i in 0..<nFreqs {
            frequencies[i] = Float(i) * binWidth
        }

        // Initialize filterbank
        var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nChroma)

        // Map each frequency bin to chroma
        let fRef: Float = 261.63  // C4 reference frequency

        for freq in 1..<nFreqs {  // Skip DC
            let f = frequencies[freq]

            // Convert frequency to chroma bin (with tuning offset)
            let chromaFloat = Float(nChroma) * log2(f / fRef) + tuning
            let chromaIdx = Int(chromaFloat.truncatingRemainder(dividingBy: Float(nChroma)))

            let validIdx = chromaIdx >= 0 ? chromaIdx : chromaIdx + nChroma

            if validIdx >= 0 && validIdx < nChroma {
                // Simple assignment (could use interpolation for smoother results)
                filterbank[validIdx][freq] = 1.0
            }
        }

        return filterbank
    }

    private func normalizeChroma(_ chroma: [[Float]]) -> [[Float]] {
        guard !chroma.isEmpty && !chroma[0].isEmpty else { return chroma }

        let nFrames = chroma[0].count
        var normalized = chroma

        switch config.norm {
        case .none:
            return chroma

        case .l1:
            // Normalize each frame to sum to 1
            for frame in 0..<nFrames {
                var sum: Float = 0
                for chromaIdx in 0..<config.nChroma {
                    sum += normalized[chromaIdx][frame]
                }
                if sum > 0 {
                    for chromaIdx in 0..<config.nChroma {
                        normalized[chromaIdx][frame] /= sum
                    }
                }
            }

        case .l2:
            // Normalize each frame to unit L2 norm
            for frame in 0..<nFrames {
                var sumSquares: Float = 0
                for chromaIdx in 0..<config.nChroma {
                    sumSquares += normalized[chromaIdx][frame] * normalized[chromaIdx][frame]
                }
                let norm = sqrt(sumSquares)
                if norm > 0 {
                    for chromaIdx in 0..<config.nChroma {
                        normalized[chromaIdx][frame] /= norm
                    }
                }
            }

        case .max:
            // Normalize each frame so max = 1
            for frame in 0..<nFrames {
                var maxVal: Float = 0
                for chromaIdx in 0..<config.nChroma {
                    maxVal = max(maxVal, normalized[chromaIdx][frame])
                }
                if maxVal > 0 {
                    for chromaIdx in 0..<config.nChroma {
                        normalized[chromaIdx][frame] /= maxVal
                    }
                }
            }
        }

        return normalized
    }
}

// MARK: - Presets

extension Chromagram {
    /// Configuration for music analysis.
    public static var musicAnalysis: Chromagram {
        Chromagram(config: ChromagramConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12,
            useCQT: true,
            norm: .l2,
            tuning: 0.0,
            nOctaves: 7,
            binsPerOctave: 36
        ))
    }

    /// Configuration for chord recognition.
    public static var chordRecognition: Chromagram {
        Chromagram(config: ChromagramConfig(
            sampleRate: 22050,
            hopLength: 2048,  // Larger hop for chord-level features
            nChroma: 12,
            useCQT: true,
            norm: .l2,
            tuning: 0.0,
            nOctaves: 6,
            binsPerOctave: 36
        ))
    }

    /// STFT-based chromagram (faster but less accurate).
    public static var stftBased: Chromagram {
        Chromagram(config: ChromagramConfig(
            sampleRate: 22050,
            hopLength: 512,
            nChroma: 12,
            useCQT: false,
            norm: .l2,
            tuning: 0.0,
            nFFT: 4096,
            windowType: .hann
        ))
    }
}

// MARK: - CENS (Chroma Energy Normalized Statistics)

extension Chromagram {
    /// Compute Chroma Energy Normalized Statistics (CENS).
    ///
    /// CENS is a variant of chroma features that applies:
    /// 1. L1 normalization per frame
    /// 2. Quantization to discrete levels (optional)
    /// 3. Temporal smoothing with a window function
    /// 4. Final L2 normalization
    ///
    /// CENS features are robust to local tempo variations and are useful for
    /// music retrieval, cover song identification, and audio matching.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - winLenSmooth: Smoothing window length in frames (default: 41).
    ///   - quantize: Whether to apply quantization (default: true).
    ///   - nQuant: Number of quantization levels if quantize=true (default: 4).
    /// - Returns: CENS chromagram of shape (nChroma, nFrames).
    ///
    /// - Note: Matches `librosa.feature.chroma_cens(y, sr)`
    public func cens(
        _ signal: [Float],
        winLenSmooth: Int = 41,
        quantize: Bool = true,
        nQuant: Int = 4
    ) async -> [[Float]] {
        // Get base chroma features
        var chroma = await transform(signal)
        guard !chroma.isEmpty, !chroma[0].isEmpty else { return [] }

        let nChroma = chroma.count
        let nFrames = chroma[0].count

        // Step 1: L1 normalization per frame
        for t in 0..<nFrames {
            var sum: Float = 0
            for c in 0..<nChroma {
                sum += abs(chroma[c][t])
            }
            if sum > 1e-10 {
                for c in 0..<nChroma {
                    chroma[c][t] /= sum
                }
            }
        }

        // Step 2: Quantization (optional)
        if quantize && nQuant > 0 {
            // Quantize to [0, 1/nQuant, 2/nQuant, ..., 1]
            let step = 1.0 / Float(nQuant)
            for c in 0..<nChroma {
                for t in 0..<nFrames {
                    // Quantize: round to nearest step
                    let quantized = round(chroma[c][t] / step) * step
                    chroma[c][t] = min(1.0, max(0.0, quantized))
                }
            }
        }

        // Step 3: Temporal smoothing with Hann window
        let window = generateHannWindow(length: winLenSmooth)
        let windowSum = window.reduce(0, +)

        var smoothed = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nChroma)
        let halfWin = winLenSmooth / 2

        for c in 0..<nChroma {
            for t in 0..<nFrames {
                var sum: Float = 0
                var weightSum: Float = 0

                for w in 0..<winLenSmooth {
                    let frameIdx = t - halfWin + w
                    if frameIdx >= 0 && frameIdx < nFrames {
                        sum += chroma[c][frameIdx] * window[w]
                        weightSum += window[w]
                    }
                }

                smoothed[c][t] = weightSum > 0 ? sum / weightSum : 0
            }
        }

        // Step 4: Final L2 normalization per frame
        for t in 0..<nFrames {
            var sumSquares: Float = 0
            for c in 0..<nChroma {
                sumSquares += smoothed[c][t] * smoothed[c][t]
            }
            let norm = sqrt(sumSquares)
            if norm > 1e-10 {
                for c in 0..<nChroma {
                    smoothed[c][t] /= norm
                }
            }
        }

        return smoothed
    }

    /// Generate a Hann window for smoothing.
    private func generateHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        for i in 0..<length {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(length - 1)))
        }
        return window
    }
}

// MARK: - Utility Methods

extension Chromagram {
    /// Pitch class names.
    public static let pitchClassNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    /// Get the name of a chroma bin.
    ///
    /// - Parameter bin: Bin index (0-11).
    /// - Returns: Pitch class name.
    public static func binName(_ bin: Int) -> String {
        guard bin >= 0 && bin < 12 else { return "?" }
        return pitchClassNames[bin]
    }

    /// Find dominant pitch class in a frame.
    ///
    /// - Parameter frame: Chroma frame (12 values).
    /// - Returns: Index of dominant pitch class.
    public static func dominantPitchClass(_ frame: [Float]) -> Int {
        guard frame.count == 12 else { return 0 }

        var maxIdx = 0
        var maxVal: Float = frame[0]

        for i in 1..<12 {
            if frame[i] > maxVal {
                maxVal = frame[i]
                maxIdx = i
            }
        }

        return maxIdx
    }

    /// Compute chroma correlation between two frames.
    ///
    /// - Parameters:
    ///   - frame1: First chroma frame.
    ///   - frame2: Second chroma frame.
    /// - Returns: Correlation coefficient (-1 to 1).
    public static func correlation(_ frame1: [Float], _ frame2: [Float]) -> Float {
        guard frame1.count == 12 && frame2.count == 12 else { return 0 }

        // Compute means
        let mean1 = frame1.reduce(0, +) / 12.0
        let mean2 = frame2.reduce(0, +) / 12.0

        // Compute correlation
        var numerator: Float = 0
        var denom1: Float = 0
        var denom2: Float = 0

        for i in 0..<12 {
            let d1 = frame1[i] - mean1
            let d2 = frame2[i] - mean2
            numerator += d1 * d2
            denom1 += d1 * d1
            denom2 += d2 * d2
        }

        let denominator = sqrt(denom1 * denom2)
        return denominator > 0 ? numerator / denominator : 0
    }
}
