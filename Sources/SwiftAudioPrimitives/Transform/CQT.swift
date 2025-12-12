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
        nBins / binsPerOctave
    }

    /// Maximum frequency.
    public var fMax: Float {
        fMin * pow(2.0, Float(nBins) / Float(binsPerOctave))
    }

    /// Q factor (frequency-to-bandwidth ratio).
    public var Q: Float {
        filterScale / (pow(2.0, 1.0 / Float(binsPerOctave)) - 1)
    }

    /// Get center frequency for a bin.
    public func centerFrequency(forBin bin: Int) -> Float {
        fMin * pow(2.0, Float(bin) / Float(binsPerOctave))
    }
}

/// Constant-Q Transform.
///
/// The Constant-Q Transform (CQT) is a time-frequency representation where
/// frequency bins are geometrically spaced (constant Q-factor), making it
/// ideal for music analysis where pitch relationships are logarithmic.
///
/// ## Key Differences from STFT
///
/// - **Frequency spacing**: Logarithmic (musical scale) vs linear
/// - **Time-frequency resolution**: Variable vs constant
/// - **Best for**: Music, pitch analysis vs general audio
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

    /// Pre-computed CQT kernels.
    private let kernels: [CQTKernel]

    /// Create a CQT processor.
    ///
    /// - Parameter config: CQT configuration.
    public init(config: CQTConfig = CQTConfig()) {
        self.config = config
        self.kernels = Self.computeKernels(config: config)
    }

    /// Compute CQT of signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram of shape (nBins, nFrames).
    public func transform(_ signal: [Float]) async -> ComplexMatrix {
        guard !signal.isEmpty else {
            return ComplexMatrix(real: [], imag: [])
        }

        // Pad signal
        let padLength = kernels.map { $0.filterLength }.max() ?? 0
        let paddedSignal = padSignal(signal, padLength: padLength / 2)

        // Compute number of frames
        let nFrames = max(1, (paddedSignal.count - padLength) / config.hopLength + 1)

        // Initialize output
        var realOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)
        var imagOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)

        // Process each bin using vectorized dot products
        paddedSignal.withUnsafeBufferPointer { signalPtr in
            for binIdx in 0..<config.nBins {
                let kernel = kernels[binIdx]

                kernel.realPart.withUnsafeBufferPointer { realKernelPtr in
                    kernel.imagPart.withUnsafeBufferPointer { imagKernelPtr in
                        for frameIdx in 0..<nFrames {
                            let center = frameIdx * config.hopLength + padLength / 2
                            let start = center - kernel.filterLength / 2
                            let end = start + kernel.filterLength

                            guard start >= 0 && end <= paddedSignal.count else { continue }

                            // Use vDSP_dotpr for vectorized convolution
                            var realSum: Float = 0
                            var imagSum: Float = 0

                            vDSP_dotpr(
                                signalPtr.baseAddress! + start, 1,
                                realKernelPtr.baseAddress!, 1,
                                &realSum,
                                vDSP_Length(kernel.filterLength)
                            )

                            vDSP_dotpr(
                                signalPtr.baseAddress! + start, 1,
                                imagKernelPtr.baseAddress!, 1,
                                &imagSum,
                                vDSP_Length(kernel.filterLength)
                            )

                            realOutput[binIdx][frameIdx] = realSum
                            imagOutput[binIdx][frameIdx] = imagSum
                        }
                    }
                }
            }
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

    private static func computeKernels(config: CQTConfig) -> [CQTKernel] {
        var kernels = [CQTKernel]()
        kernels.reserveCapacity(config.nBins)

        for bin in 0..<config.nBins {
            let centerFreq = config.centerFrequency(forBin: bin)

            // Compute filter length based on Q factor
            let filterLength = Int(ceil(config.Q * config.sampleRate / centerFreq))

            // Ensure filter length is reasonable
            let clampedLength = max(4, min(filterLength, 8192))

            // Generate window
            let window = Windows.generate(config.windowType, length: clampedLength, periodic: true)

            // Generate complex exponential kernel using vDSP
            var realPart = [Float](repeating: 0, count: clampedLength)
            var imagPart = [Float](repeating: 0, count: clampedLength)

            let normFactor = 1.0 / Float(clampedLength)
            let phaseIncrement = 2.0 * Float.pi * centerFreq / config.sampleRate

            // Generate ramp for phase values: 0, phaseIncrement, 2*phaseIncrement, ...
            var phases = [Float](repeating: 0, count: clampedLength)
            var zero: Float = 0
            var increment = phaseIncrement
            vDSP_vramp(&zero, &increment, &phases, 1, vDSP_Length(clampedLength))

            // Compute cos and sin using vForce (vectorized)
            var cosValues = [Float](repeating: 0, count: clampedLength)
            var sinValues = [Float](repeating: 0, count: clampedLength)
            var count = Int32(clampedLength)
            vvcosf(&cosValues, phases, &count)
            vvsinf(&sinValues, phases, &count)

            // Multiply by window and normalization factor
            // realPart = window * cos * normFactor
            // imagPart = window * (-sin) * normFactor
            var normFactorVar = normFactor
            var negNormFactor = -normFactor

            // realPart = cosValues * window
            vDSP_vmul(cosValues, 1, window, 1, &realPart, 1, vDSP_Length(clampedLength))
            // realPart *= normFactor
            vDSP_vsmul(realPart, 1, &normFactorVar, &realPart, 1, vDSP_Length(clampedLength))

            // imagPart = sinValues * window * (-normFactor)
            vDSP_vmul(sinValues, 1, window, 1, &imagPart, 1, vDSP_Length(clampedLength))
            vDSP_vsmul(imagPart, 1, &negNormFactor, &imagPart, 1, vDSP_Length(clampedLength))

            kernels.append(CQTKernel(
                centerFreq: centerFreq,
                filterLength: clampedLength,
                realPart: realPart,
                imagPart: imagPart
            ))
        }

        return kernels
    }

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

/// Internal representation of a CQT kernel for one frequency bin.
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
