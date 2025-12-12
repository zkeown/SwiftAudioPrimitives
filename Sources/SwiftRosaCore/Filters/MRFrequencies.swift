import Accelerate
import Foundation

/// Multi-rate frequency specification for variable bandwidth transforms.
public enum MRFrequencyScale: Sendable {
    /// Constant-Q (geometric/logarithmic spacing).
    case cq
    /// Variable-Q with ERB bandwidth matching.
    case vqt
    /// Custom Q-factor curve.
    case custom(@Sendable (Float) -> Float)
}

/// Configuration for multi-rate frequency planning.
public struct MRFrequenciesConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Minimum frequency (Hz).
    public let fMin: Float

    /// Maximum frequency (Hz, optional - computed from nBins if nil).
    public let fMax: Float?

    /// Total number of frequency bins.
    public let nBins: Int

    /// Bins per octave (default: 12).
    public let binsPerOctave: Int

    /// Frequency scale type.
    public let scale: MRFrequencyScale

    /// Alpha parameter for VQT (controls bandwidth variation).
    public let alpha: Float

    /// Gamma parameter for VQT (minimum bandwidth floor).
    public let gamma: Float

    /// Create multi-rate frequency configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - fMin: Minimum frequency (default: 32.70 Hz, C1).
    ///   - fMax: Maximum frequency (optional).
    ///   - nBins: Number of frequency bins (default: 84).
    ///   - binsPerOctave: Bins per octave (default: 12).
    ///   - scale: Frequency scale type (default: cq).
    ///   - alpha: VQT alpha parameter (default: 1.0).
    ///   - gamma: VQT gamma parameter (default: 0, disabled).
    public init(
        sampleRate: Float = 22050,
        fMin: Float = 32.70,
        fMax: Float? = nil,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        scale: MRFrequencyScale = .cq,
        alpha: Float = 1.0,
        gamma: Float = 0
    ) {
        self.sampleRate = sampleRate
        self.fMin = fMin
        self.fMax = fMax
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.scale = scale
        self.alpha = alpha
        self.gamma = gamma
    }

    /// Number of octaves covered.
    public var nOctaves: Int {
        nBins / binsPerOctave
    }

    /// Computed maximum frequency.
    public var computedFMax: Float {
        fMax ?? (fMin * pow(2.0, Float(nBins) / Float(binsPerOctave)))
    }
}

/// Frequency bin specification.
public struct FrequencyBin: Sendable {
    /// Center frequency in Hz.
    public let centerFrequency: Float

    /// Bandwidth in Hz.
    public let bandwidth: Float

    /// Q factor (center frequency / bandwidth).
    public let Q: Float

    /// Filter length in samples for this bin.
    public let filterLength: Int

    /// Octave index (0 = lowest octave).
    public let octaveIndex: Int

    /// Bin index within octave.
    public let binWithinOctave: Int
}

/// Multi-rate frequency planning utilities.
///
/// Computes frequency bins and bandwidths for variable-Q and constant-Q
/// transforms. This is the foundation for VQT, CQT, and other log-frequency
/// representations.
///
/// ## Example Usage
///
/// ```swift
/// // Standard CQT frequencies
/// let cqFreqs = MRFrequencies.compute(config: MRFrequenciesConfig(
///     fMin: 32.70, nBins: 84, binsPerOctave: 12
/// ))
///
/// // VQT with ERB-matching bandwidths
/// let vqtFreqs = MRFrequencies.compute(config: MRFrequenciesConfig(
///     fMin: 32.70, nBins: 84, scale: .vqt, gamma: 20
/// ))
///
/// // Access frequency information
/// for bin in vqtFreqs {
///     print("f=\(bin.centerFrequency) Hz, BW=\(bin.bandwidth) Hz, Q=\(bin.Q)")
/// }
/// ```
///
/// ## Formulas
///
/// **Constant-Q (CQT)**:
/// ```
/// f[k] = fMin * 2^(k/binsPerOctave)
/// Q = constant = binsPerOctave / (2^(1/binsPerOctave) - 1)
/// bandwidth[k] = f[k] / Q
/// ```
///
/// **Variable-Q (VQT)**:
/// ```
/// f[k] = fMin * 2^(k/binsPerOctave)
/// bandwidth[k] = alpha * f[k] / Q_ref + gamma
/// Q[k] = f[k] / bandwidth[k]
/// ```
///
/// The gamma parameter adds a constant floor to bandwidth, providing
/// better low-frequency resolution (matching ERB at low frequencies).
public struct MRFrequencies {
    private init() {}

    /// Compute frequency bin specifications.
    ///
    /// - Parameter config: Configuration.
    /// - Returns: Array of frequency bin specifications.
    public static func compute(config: MRFrequenciesConfig) -> [FrequencyBin] {
        var bins = [FrequencyBin]()
        bins.reserveCapacity(config.nBins)

        // Reference Q for constant-Q
        let QRef = Float(config.binsPerOctave) / (pow(2.0, 1.0 / Float(config.binsPerOctave)) - 1)

        for k in 0..<config.nBins {
            let centerFreq = config.fMin * pow(2.0, Float(k) / Float(config.binsPerOctave))

            let bandwidth: Float
            let Q: Float

            switch config.scale {
            case .cq:
                // Constant-Q: bandwidth proportional to frequency
                Q = QRef
                bandwidth = centerFreq / Q

            case .vqt:
                // Variable-Q: bandwidth = alpha * f / Q_ref + gamma
                bandwidth = config.alpha * centerFreq / QRef + config.gamma
                Q = centerFreq / bandwidth

            case .custom(let qFunction):
                Q = qFunction(centerFreq)
                bandwidth = centerFreq / Q
            }

            // Compute filter length: samples = sampleRate / bandwidth (roughly)
            // For CQT/VQT kernels, length = Q * sampleRate / centerFreq
            let filterLength = Int(ceil(Q * config.sampleRate / centerFreq))

            let octaveIndex = k / config.binsPerOctave
            let binWithinOctave = k % config.binsPerOctave

            bins.append(FrequencyBin(
                centerFrequency: centerFreq,
                bandwidth: bandwidth,
                Q: Q,
                filterLength: filterLength,
                octaveIndex: octaveIndex,
                binWithinOctave: binWithinOctave
            ))
        }

        return bins
    }

    /// Get center frequencies only.
    ///
    /// - Parameter config: Configuration.
    /// - Returns: Array of center frequencies in Hz.
    public static func centerFrequencies(config: MRFrequenciesConfig) -> [Float] {
        (0..<config.nBins).map { k in
            config.fMin * pow(2.0, Float(k) / Float(config.binsPerOctave))
        }
    }

    /// Get bandwidths only.
    ///
    /// - Parameter config: Configuration.
    /// - Returns: Array of bandwidths in Hz.
    public static func bandwidths(config: MRFrequenciesConfig) -> [Float] {
        compute(config: config).map { $0.bandwidth }
    }

    /// Get Q factors only.
    ///
    /// - Parameter config: Configuration.
    /// - Returns: Array of Q factors.
    public static func qFactors(config: MRFrequenciesConfig) -> [Float] {
        compute(config: config).map { $0.Q }
    }

    /// Compute optimal FFT sizes per octave for efficient processing.
    ///
    /// Returns FFT sizes that are powers of 2, sized to accommodate
    /// the longest filter in each octave.
    ///
    /// - Parameter bins: Frequency bin specifications.
    /// - Returns: Dictionary mapping octave index to FFT size.
    public static func optimalFFTSizes(bins: [FrequencyBin]) -> [Int: Int] {
        var sizes = [Int: Int]()

        // Group bins by octave and find max filter length per octave
        var maxLengths = [Int: Int]()
        for bin in bins {
            let current = maxLengths[bin.octaveIndex] ?? 0
            maxLengths[bin.octaveIndex] = max(current, bin.filterLength)
        }

        // Convert to power of 2
        for (octave, maxLength) in maxLengths {
            sizes[octave] = nextPowerOfTwo(maxLength)
        }

        return sizes
    }

    /// Group bins by octave for multi-rate processing.
    ///
    /// - Parameter bins: Frequency bin specifications.
    /// - Returns: Dictionary mapping octave index to bins in that octave.
    public static func groupByOctave(bins: [FrequencyBin]) -> [Int: [FrequencyBin]] {
        var groups = [Int: [FrequencyBin]]()

        for bin in bins {
            if groups[bin.octaveIndex] == nil {
                groups[bin.octaveIndex] = []
            }
            groups[bin.octaveIndex]?.append(bin)
        }

        return groups
    }

    /// Convert frequency to the nearest bin index.
    ///
    /// - Parameters:
    ///   - frequency: Frequency in Hz.
    ///   - config: Configuration.
    /// - Returns: Nearest bin index (clamped to valid range).
    public static func frequencyToBin(
        _ frequency: Float,
        config: MRFrequenciesConfig
    ) -> Int {
        let bin = Float(config.binsPerOctave) * log2(frequency / config.fMin)
        return max(0, min(config.nBins - 1, Int(round(bin))))
    }

    /// Convert bin index to center frequency.
    ///
    /// - Parameters:
    ///   - bin: Bin index.
    ///   - config: Configuration.
    /// - Returns: Center frequency in Hz.
    public static func binToFrequency(
        _ bin: Int,
        config: MRFrequenciesConfig
    ) -> Float {
        config.fMin * pow(2.0, Float(bin) / Float(config.binsPerOctave))
    }

    /// Compute ERB (Equivalent Rectangular Bandwidth) for a frequency.
    ///
    /// ERB approximates the bandwidth of human auditory filters.
    ///
    /// - Parameter frequency: Frequency in Hz.
    /// - Returns: ERB bandwidth in Hz.
    public static func erb(_ frequency: Float) -> Float {
        // Glasberg & Moore (1990) formula
        return 24.7 * (4.37 * frequency / 1000.0 + 1.0)
    }

    /// Compute gamma parameter that matches ERB at a reference frequency.
    ///
    /// - Parameters:
    ///   - refFrequency: Reference frequency to match ERB.
    ///   - binsPerOctave: Bins per octave for CQ calculation.
    /// - Returns: Gamma value for VQT configuration.
    public static func gammaForERB(
        at refFrequency: Float,
        binsPerOctave: Int = 12
    ) -> Float {
        let QRef = Float(binsPerOctave) / (pow(2.0, 1.0 / Float(binsPerOctave)) - 1)
        let cqBandwidth = refFrequency / QRef
        let erbBandwidth = erb(refFrequency)
        return max(0, erbBandwidth - cqBandwidth)
    }

    // MARK: - Private Helpers

    private static func nextPowerOfTwo(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }
}

// MARK: - Presets

extension MRFrequenciesConfig {
    /// Standard CQT configuration (7 octaves from C1).
    public static var cqt: MRFrequenciesConfig {
        MRFrequenciesConfig(
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            scale: .cq
        )
    }

    /// VQT configuration with ERB-matching gamma.
    public static var vqt: MRFrequenciesConfig {
        // Gamma computed to match ERB at ~500 Hz
        let gamma = MRFrequencies.gammaForERB(at: 500.0, binsPerOctave: 12)
        return MRFrequenciesConfig(
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            scale: .vqt,
            gamma: gamma
        )
    }

    /// High-resolution configuration (24 bins per octave).
    public static var highResolution: MRFrequenciesConfig {
        MRFrequenciesConfig(
            fMin: 32.70,
            nBins: 168,
            binsPerOctave: 24,
            scale: .cq
        )
    }

    /// Music analysis configuration (A0 to C8).
    public static var musicAnalysis: MRFrequenciesConfig {
        MRFrequenciesConfig(
            fMin: 27.50, // A0
            nBins: 88,   // 88 piano keys worth
            binsPerOctave: 12,
            scale: .cq
        )
    }
}
