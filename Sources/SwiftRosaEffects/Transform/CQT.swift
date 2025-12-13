import SwiftRosaCore
import Accelerate
import Foundation

/// Errors that can occur during CQT computation.
public enum CQTError: Error, Sendable {
    case invalidParameters(String)
    case computationError(String)
}

/// Configuration for Constant-Q Transform.
public struct CQTConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Minimum frequency (Hz).
    public let fMin: Float

    /// Total number of frequency bins.
    public let nBins: Int

    /// Bins per octave (12 = semitones).
    public let binsPerOctave: Int

    /// Window type.
    public let windowType: WindowType

    /// Filter scale factor (< 1 = better time resolution).
    public let filterScale: Float

    /// Sparsity threshold for kernel pruning.
    public let sparsity: Float

    /// Create CQT configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - fMin: Minimum frequency (default: 32.70 Hz, C1).
    ///   - nBins: Number of frequency bins (default: 84, 7 octaves).
    ///   - binsPerOctave: Bins per octave (default: 12).
    ///   - windowType: Window type (default: hann).
    ///   - filterScale: Filter scale (default: 1.0).
    ///   - sparsity: Sparsity threshold (default: 0.01).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        windowType: WindowType = .hann,
        filterScale: Float = 1.0,
        sparsity: Float = 0.01
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.fMin = fMin
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.windowType = windowType
        self.filterScale = filterScale
        self.sparsity = sparsity
    }

    /// Number of octaves covered.
    public var nOctaves: Int {
        (nBins + binsPerOctave - 1) / binsPerOctave
    }

    /// Maximum frequency.
    public var fMax: Float {
        fMin * pow(2.0, Float(nBins) / Float(binsPerOctave))
    }

    /// Q factor (frequency-to-bandwidth ratio).
    /// Uses librosa's alpha formula: (2^(2/B) - 1) / (2^(2/B) + 1)
    /// This provides better frequency resolution than the standard formula.
    public var Q: Float {
        let twoOverB = 2.0 / Float(binsPerOctave)
        let ratio = pow(2.0, twoOverB)
        let alpha = (ratio - 1) / (ratio + 1)
        return filterScale / alpha
    }

    /// Get center frequency for a bin.
    public func centerFrequency(forBin bin: Int) -> Float {
        fMin * pow(2.0, Float(bin) / Float(binsPerOctave))
    }
}

/// Pre-computed CQT filter for one frequency bin.
private struct CQTFilter: Sendable {
    let centerFreq: Float
    let filterLength: Int
    let realPart: [Float]  // cos component (windowed)
    let imagPart: [Float]  // sin component (windowed)
    let l1Norm: Float      // For normalization
}

/// Parameters for processing a single octave in the multi-resolution CQT.
private struct OctaveParams: Sendable {
    let octave: Int
    let nBins: Int              // Number of bins in this octave
    let binOffset: Int          // Offset in output array
    let effectiveSR: Float      // Sample rate at this octave (after downsampling)
    let hopLength: Int          // Hop length at this octave
    let downsampleFactor: Int   // Total downsampling factor for this octave
    let filters: [CQTFilter]    // Time-domain filters for this octave
}

/// Constant-Q Transform with multi-resolution time-domain correlation.
///
/// The Constant-Q Transform (CQT) is a time-frequency representation where
/// frequency bins are geometrically spaced (constant Q-factor), making it
/// ideal for music analysis where pitch relationships are logarithmic.
///
/// This implementation uses direct time-domain correlation with multi-rate processing:
/// - Process octaves from highest to lowest frequency
/// - Use direct correlation (inner product) with windowed sinusoid filters
/// - Downsample signal by 2 between octaves
/// - L1 normalization to match librosa's default norm=1
///
/// ## Example Usage
///
/// ```swift
/// let cqt = CQT(config: CQTConfig(fMin: 32.70, nBins: 84))
///
/// // Compute CQT
/// let spectrum = await cqt.transform(audioSignal)
///
/// // Get magnitude only
/// let magnitude = await cqt.magnitude(audioSignal)
/// ```
public struct CQT: Sendable {
    /// Configuration.
    public let config: CQTConfig

    /// Pre-computed octave parameters including time-domain filters.
    private let octaveParams: [OctaveParams]

    /// Create a CQT processor.
    ///
    /// - Parameter config: CQT configuration.
    public init(config: CQTConfig = CQTConfig()) {
        self.config = config

        // Compute parameters for each octave
        var params = [OctaveParams]()

        let nOctaves = config.nOctaves
        var effectiveSR = config.sampleRate
        var hopLength = config.hopLength
        var downsampleFactor = 1  // Track cumulative downsampling

        // Process octaves from highest (last) to lowest (first)
        // This determines the effective SR for each octave
        // We iterate in reverse so that effectiveSR starts at full sample rate
        // for the highest octave and decreases for lower octaves
        for octaveIdx in (0..<nOctaves).reversed() {
            let binStart = octaveIdx * config.binsPerOctave
            let binEnd = min(binStart + config.binsPerOctave, config.nBins)
            let nBinsInOctave = binEnd - binStart

            if nBinsInOctave <= 0 { continue }

            // Create time-domain filters for this octave
            var filters = [CQTFilter]()
            for bin in binStart..<binEnd {
                let centerFreq = config.centerFrequency(forBin: bin)
                let filter = Self.createTimeDomainFilter(
                    centerFreq: centerFreq,
                    Q: config.Q,
                    effectiveSR: effectiveSR,
                    windowType: config.windowType
                )
                filters.append(filter)
            }

            params.append(OctaveParams(
                octave: octaveIdx,
                nBins: nBinsInOctave,
                binOffset: binStart,
                effectiveSR: effectiveSR,
                hopLength: hopLength,
                downsampleFactor: downsampleFactor,
                filters: filters
            ))

            // Downsample for next (lower frequency) octave
            effectiveSR /= 2
            hopLength = max(1, hopLength / 2)
            downsampleFactor *= 2
        }

        // params is now ordered [octave6, octave5, ..., octave0] which is the
        // correct processing order (highest freq first, then downsample)
        self.octaveParams = params
    }

    /// Compute CQT of signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram of shape (nBins, nFrames).
    public func transform(_ signal: [Float]) async -> ComplexMatrix {
        guard !signal.isEmpty else {
            return ComplexMatrix(real: [], imag: [])
        }

        // Compute number of frames based on original signal and hop length
        let nFrames = max(1, (signal.count + config.hopLength - 1) / config.hopLength)

        // Initialize output arrays
        var realOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)
        var imagOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)

        // Process octaves from highest to lowest frequency
        var currentSignal = signal

        for octave in octaveParams {
            // Process this octave
            processOctave(
                octave: octave,
                signal: currentSignal,
                nFrames: nFrames,
                realOutput: &realOutput,
                imagOutput: &imagOutput
            )

            // Downsample signal for next octave (antialiasing + decimation)
            currentSignal = downsample(currentSignal, factor: 2)
        }

        return ComplexMatrix(real: realOutput, imag: imagOutput)
    }

    /// Compute CQT magnitude.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nBins, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let cqt = await transform(signal)
        return cqt.magnitude
    }

    /// Compute CQT power (magnitude squared).
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

    // MARK: - Private Implementation

    /// Create time-domain filter for one CQT bin.
    ///
    /// Creates a windowed complex sinusoid at the center frequency,
    /// normalized by L1 norm to match librosa's default norm=1.
    private static func createTimeDomainFilter(
        centerFreq: Float,
        Q: Float,
        effectiveSR: Float,
        windowType: WindowType
    ) -> CQTFilter {
        // Filter length based on Q factor
        let filterLength = Int(ceil(Q * effectiveSR / centerFreq))

        // Generate window
        let window = Windows.generate(windowType, length: filterLength, periodic: false)

        // Generate windowed complex sinusoid
        var realPart = [Float](repeating: 0, count: filterLength)
        var imagPart = [Float](repeating: 0, count: filterLength)

        let phaseIncrement = 2.0 * Float.pi * centerFreq / effectiveSR
        var l1Norm: Float = 0

        for i in 0..<filterLength {
            let phase = phaseIncrement * Float(i)
            let windowVal = window[i]
            realPart[i] = windowVal * cos(phase)
            imagPart[i] = windowVal * sin(phase)
            // L1 norm of complex values: sum of |z| = sum of sqrt(re^2 + im^2)
            // For unit-amplitude sinusoid with window: sum of |window|
            l1Norm += windowVal
        }

        return CQTFilter(
            centerFreq: centerFreq,
            filterLength: filterLength,
            realPart: realPart,
            imagPart: imagPart,
            l1Norm: l1Norm
        )
    }

    /// Process one octave using direct time-domain correlation.
    private func processOctave(
        octave: OctaveParams,
        signal: [Float],
        nFrames: Int,
        realOutput: inout [[Float]],
        imagOutput: inout [[Float]]
    ) {
        guard !signal.isEmpty else { return }

        // Find the longest filter in this octave (for padding calculation)
        let maxFilterLen = octave.filters.map { $0.filterLength }.max() ?? 0
        guard maxFilterLen > 0 else { return }

        // Pad signal for centered frames
        let padLen = maxFilterLen / 2
        var paddedSignal = [Float](repeating: 0, count: padLen)
        paddedSignal.append(contentsOf: signal)
        paddedSignal.append(contentsOf: [Float](repeating: 0, count: padLen))

        // Calculate frames at this octave's sample rate
        let octaveFrames = max(1, (signal.count + octave.hopLength - 1) / octave.hopLength)

        // Frame scale factor (octave frame index -> output frame index)
        let frameScale = Float(nFrames) / Float(octaveFrames)

        // Process each frame
        for octaveFrame in 0..<octaveFrames {
            // Map to output frame
            let outFrame = min(Int(Float(octaveFrame) * frameScale), nFrames - 1)

            // Frame center position in padded signal
            let frameCenter = padLen + octaveFrame * octave.hopLength

            // Apply each filter
            for (binIdx, filter) in octave.filters.enumerated() {
                let globalBin = octave.binOffset + binIdx

                // Correlation: inner product of signal with conjugate of filter
                // CQT = sum(signal * conj(filter)) = sum(signal * (re - j*im))
                //     = sum(signal * re) - j * sum(signal * im)
                let halfLen = filter.filterLength / 2
                let startIdx = frameCenter - halfLen

                var sumReal: Float = 0
                var sumImag: Float = 0

                // Bounds check
                let actualStart = max(0, startIdx)
                let actualEnd = min(paddedSignal.count, startIdx + filter.filterLength)

                if actualEnd > actualStart {
                    for i in actualStart..<actualEnd {
                        let filterIdx = i - startIdx
                        if filterIdx >= 0 && filterIdx < filter.filterLength {
                            let sigVal = paddedSignal[i]
                            // Correlation with conjugate: (re - j*im)
                            sumReal += sigVal * filter.realPart[filterIdx]
                            sumImag -= sigVal * filter.imagPart[filterIdx]  // Conjugate
                        }
                    }
                }

                // Normalize to match librosa's norm=1 output
                // librosa formula: CQT = correlation * sqrt(filterLength) * sqrt(downsampleFactor) / L1_norm
                // The sqrt(downsampleFactor) compensates for energy spreading when downsampling
                // This gives energy-independent normalization across different filter lengths and octaves
                if filter.l1Norm > 0 {
                    let sqrtLen = sqrt(Float(filter.filterLength))
                    let sqrtDS = sqrt(Float(octave.downsampleFactor))
                    sumReal = sumReal * sqrtLen * sqrtDS / filter.l1Norm
                    sumImag = sumImag * sqrtLen * sqrtDS / filter.l1Norm
                }

                realOutput[globalBin][outFrame] = sumReal
                imagOutput[globalBin][outFrame] = sumImag
            }
        }
    }

    /// Downsample signal by factor using vDSP decimation.
    ///
    /// Uses vDSP_desamp with a proper anti-aliasing FIR filter designed to
    /// preserve energy in the passband while attenuating aliasing frequencies.
    private func downsample(_ signal: [Float], factor: Int) -> [Float] {
        guard factor > 1, signal.count > factor else { return signal }

        // Use vDSP_desamp for efficient polyphase decimation
        // Design a proper lowpass FIR filter for anti-aliasing
        // Cutoff at Nyquist/factor with reasonable transition band

        // Create a windowed-sinc lowpass filter
        // For factor=2 decimation, cutoff = 0.5 * Nyquist = 0.25 * sampleRate
        let filterTaps = 31  // Odd number for symmetric filter
        let halfTaps = filterTaps / 2
        let cutoff: Float = 0.4 / Float(factor)  // Slightly below Nyquist/factor for margin

        var filter = [Float](repeating: 0, count: filterTaps)
        var filterSum: Float = 0

        // Generate windowed-sinc filter
        for i in 0..<filterTaps {
            let n = Float(i - halfTaps)
            if n == 0 {
                filter[i] = 2 * cutoff
            } else {
                // Sinc function
                let sincVal = sin(2 * Float.pi * cutoff * n) / (Float.pi * n)
                // Hamming window for better stopband attenuation
                let window = 0.54 - 0.46 * cos(2 * Float.pi * Float(i) / Float(filterTaps - 1))
                filter[i] = sincVal * window
            }
            filterSum += filter[i]
        }

        // Normalize filter for unity gain at DC
        for i in 0..<filterTaps {
            filter[i] /= filterSum
        }

        // Use vDSP_desamp for efficient decimation
        // vDSP_desamp: C[n] = sum(A[n*DF + k] * B[k]) for k in [0, P)
        // where DF = decimation factor, P = filter length

        let outLen = (signal.count - filterTaps + factor) / factor
        guard outLen > 0 else { return signal }

        var result = [Float](repeating: 0, count: outLen)

        vDSP_desamp(
            signal,                    // Input
            vDSP_Stride(factor),       // Decimation factor
            filter,                    // Filter coefficients
            &result,                   // Output
            vDSP_Length(outLen),       // Output length
            vDSP_Length(filterTaps)    // Filter length
        )

        return result
    }

    private static func nextPowerOfTwo(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }
}

/// Internal representation of a CQT kernel for one frequency bin (legacy).
struct CQTKernel: Sendable {
    let centerFreq: Float
    let filterLength: Int
    let realPart: [Float]
    let imagPart: [Float]
}

// MARK: - Presets

extension CQT {
    /// Configuration for music analysis (7 octaves from C1).
    public static var musicAnalysis: CQT {
        CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,  // C1
            nBins: 84,    // 7 octaves
            binsPerOctave: 12,
            windowType: .hann,
            filterScale: 1.0,
            sparsity: 0.01
        ))
    }

    /// Configuration for pitch analysis with higher resolution.
    public static var pitchAnalysis: CQT {
        CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 256,
            fMin: 65.41,  // C2
            nBins: 72,    // 6 octaves
            binsPerOctave: 24,  // Quarter-tone resolution
            windowType: .hann,
            filterScale: 1.0,
            sparsity: 0.01
        ))
    }

    /// Configuration matching librosa defaults.
    public static var librosaDefault: CQT {
        CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            windowType: .hann,
            filterScale: 1.0,
            sparsity: 0.01
        ))
    }
}

// MARK: - Utility Methods

extension CQT {
    /// Get the center frequencies for all bins.
    public var centerFrequencies: [Float] {
        (0..<config.nBins).map { config.centerFrequency(forBin: $0) }
    }

    /// Convert bin index to musical note name.
    ///
    /// - Parameter bin: Bin index.
    /// - Returns: Note name (e.g., "C4", "F#5").
    public func binToNoteName(_ bin: Int) -> String {
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        // Assuming fMin is C1 (32.70 Hz)
        let noteIndex = bin % 12
        let octave = 1 + bin / 12

        return "\(noteNames[noteIndex])\(octave)"
    }

    /// Convert frequency to nearest bin index.
    ///
    /// - Parameter frequency: Frequency in Hz.
    /// - Returns: Nearest bin index, or nil if outside range.
    public func frequencyToBin(_ frequency: Float) -> Int? {
        guard frequency >= config.fMin && frequency <= config.fMax else {
            return nil
        }

        let bin = Int(round(Float(config.binsPerOctave) * log2(frequency / config.fMin)))
        return max(0, min(config.nBins - 1, bin))
    }
}
