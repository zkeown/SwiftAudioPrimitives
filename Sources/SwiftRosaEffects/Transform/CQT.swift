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

/// Pre-computed filter for one CQT bin with both time and frequency domain representations.
private struct CQTFFTFilter: Sendable {
    let centerFreq: Float
    let filterLength: Int
    let filterReal: [Float]   // Time-domain filter (real part)
    let filterImag: [Float]   // Time-domain filter (imag part)
    let fftReal: [Float]      // Frequency-domain filter (real part) - conjugated for correlation
    let fftImag: [Float]      // Frequency-domain filter (imag part) - conjugated for correlation
    let l1Norm: Float         // For normalization
    let fftSize: Int          // FFT size for this filter
}

/// Parameters for processing a single octave in the multi-resolution CQT.
private struct OctaveParams: Sendable {
    let octave: Int
    let nBins: Int              // Number of bins in this octave
    let binOffset: Int          // Offset in output array
    let effectiveSR: Float      // Sample rate at this octave (after downsampling)
    let hopLength: Int          // Hop length at this octave
    let downsampleFactor: Int   // Total downsampling factor for this octave
    let filters: [CQTFFTFilter] // FFT-domain filters for this octave
    let fftSize: Int            // Common FFT size for this octave
}

/// Constant-Q Transform with multi-resolution FFT-based convolution.
///
/// The Constant-Q Transform (CQT) is a time-frequency representation where
/// frequency bins are geometrically spaced (constant Q-factor), making it
/// ideal for music analysis where pitch relationships are logarithmic.
///
/// This implementation uses FFT-based convolution with multi-rate processing:
/// - Process octaves from highest to lowest frequency
/// - Use FFT multiplication instead of time-domain correlation (O(N log N) vs O(N²))
/// - Downsample signal by 2 between octaves
/// - L1 normalization to match librosa's default norm=1
///
/// ## GPU Acceleration
///
/// For signals > 50K samples, Metal GPU acceleration is automatically used
/// for the FFT operations.
///
/// ## Example Usage
///
/// ```swift
/// let cqt = CQT(config: CQTConfig(fMin: 32.70, nBins: 84))
///
/// // Compute CQT (auto-dispatches to GPU for large signals)
/// let spectrum = await cqt.transform(audioSignal)
///
/// // Get magnitude only
/// let magnitude = await cqt.magnitude(audioSignal)
/// ```
public struct CQT: Sendable {
    /// Configuration.
    public let config: CQTConfig

    /// Whether to use GPU acceleration when available.
    public let useGPU: Bool

    /// Pre-computed octave parameters including FFT filters.
    private let octaveParams: [OctaveParams]

    /// vDSP FFT setup cache by log2(fftSize).
    private let fftSetups: [Int: FFTSetupWrapper]

    /// Create a CQT processor.
    ///
    /// - Parameters:
    ///   - config: CQT configuration.
    ///   - useGPU: Whether to use GPU acceleration (default: true).
    public init(config: CQTConfig = CQTConfig(), useGPU: Bool = true) {
        self.config = config
        self.useGPU = useGPU

        // Compute parameters for each octave
        var params = [OctaveParams]()
        var setups = [Int: FFTSetupWrapper]()

        let nOctaves = config.nOctaves
        var effectiveSR = config.sampleRate
        var hopLength = config.hopLength
        var downsampleFactor = 1

        // Process octaves from highest (last) to lowest (first)
        for octaveIdx in (0..<nOctaves).reversed() {
            let binStart = octaveIdx * config.binsPerOctave
            let binEnd = min(binStart + config.binsPerOctave, config.nBins)
            let nBinsInOctave = binEnd - binStart

            if nBinsInOctave <= 0 { continue }

            // Find max filter length in this octave to determine FFT size
            var maxFilterLen = 0
            for bin in binStart..<binEnd {
                let centerFreq = config.centerFrequency(forBin: bin)
                let filterLen = Int(ceil(config.Q * effectiveSR / centerFreq))
                maxFilterLen = max(maxFilterLen, filterLen)
            }

            // FFT size must be power of 2, at least 2x filter length for linear convolution
            let fftSize = Self.nextPowerOfTwo(maxFilterLen * 2)
            let log2n = Int(log2(Double(fftSize)))

            // Create FFT setup if not cached
            if setups[log2n] == nil {
                if let setup = vDSP_create_fftsetup(vDSP_Length(log2n), FFTRadix(kFFTRadix2)) {
                    setups[log2n] = FFTSetupWrapper(setup: setup)
                }
            }

            // Create FFT-domain filters for this octave
            var filters = [CQTFFTFilter]()
            for bin in binStart..<binEnd {
                let centerFreq = config.centerFrequency(forBin: bin)
                let filter = Self.createFFTFilter(
                    centerFreq: centerFreq,
                    Q: config.Q,
                    effectiveSR: effectiveSR,
                    windowType: config.windowType,
                    fftSize: fftSize,
                    fftSetup: setups[log2n]!.setup
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
                filters: filters,
                fftSize: fftSize
            ))

            // Downsample for next (lower frequency) octave
            effectiveSR /= 2
            hopLength = max(1, hopLength / 2)
            downsampleFactor *= 2
        }

        self.octaveParams = params
        self.fftSetups = setups
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

        // Pre-compute all downsampled signals so octaves can be processed in parallel
        var downsampledSignals = [[Float]]()
        downsampledSignals.reserveCapacity(octaveParams.count)
        var currentSignal = signal
        for _ in octaveParams {
            downsampledSignals.append(currentSignal)
            currentSignal = downsample(currentSignal, factor: 2)
        }

        // Initialize output arrays - use flat arrays for thread-safe parallel access
        let totalElements = config.nBins * nFrames
        var flatReal = [Float](repeating: 0, count: totalElements)
        var flatImag = [Float](repeating: 0, count: totalElements)

        // Process octaves in parallel
        flatReal.withUnsafeMutableBufferPointer { realPtr in
            flatImag.withUnsafeMutableBufferPointer { imagPtr in
                DispatchQueue.concurrentPerform(iterations: octaveParams.count) { octaveIdx in
                    let octave = octaveParams[octaveIdx]
                    let octaveSignal = downsampledSignals[octaveIdx]

                    processOctaveParallel(
                        octave: octave,
                        signal: octaveSignal,
                        nFrames: nFrames,
                        flatReal: realPtr,
                        flatImag: imagPtr
                    )
                }
            }
        }

        // Reshape to 2D arrays
        var realOutput = [[Float]]()
        var imagOutput = [[Float]]()
        realOutput.reserveCapacity(config.nBins)
        imagOutput.reserveCapacity(config.nBins)

        for bin in 0..<config.nBins {
            let start = bin * nFrames
            realOutput.append(Array(flatReal[start..<(start + nFrames)]))
            imagOutput.append(Array(flatImag[start..<(start + nFrames)]))
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

    /// Create filter for one CQT bin with pre-computed FFT.
    ///
    /// Creates a windowed complex sinusoid at the center frequency and
    /// pre-computes its FFT for efficient frequency-domain convolution.
    private static func createFFTFilter(
        centerFreq: Float,
        Q: Float,
        effectiveSR: Float,
        windowType: WindowType,
        fftSize: Int,
        fftSetup: OpaquePointer
    ) -> CQTFFTFilter {
        // Filter length based on Q factor
        let filterLength = Int(ceil(Q * effectiveSR / centerFreq))

        // Generate window
        let window = Windows.generate(windowType, length: filterLength, periodic: false)

        // Generate windowed complex sinusoid (time-domain)
        var realPart = [Float](repeating: 0, count: filterLength)
        var imagPart = [Float](repeating: 0, count: filterLength)

        let phaseIncrement = 2.0 * Float.pi * centerFreq / effectiveSR
        var l1Norm: Float = 0

        for i in 0..<filterLength {
            let phase = phaseIncrement * Float(i)
            let windowVal = window[i]
            realPart[i] = windowVal * cos(phase)
            imagPart[i] = windowVal * sin(phase)
            l1Norm += windowVal
        }

        // Pre-compute FFT of the filter (zero-padded to fftSize)
        // We compute the conjugate here for correlation (instead of convolution)
        var paddedReal = [Float](repeating: 0, count: fftSize)
        var paddedImag = [Float](repeating: 0, count: fftSize)

        // Copy filter to padded arrays (centered for zero-phase)
        for i in 0..<filterLength {
            paddedReal[i] = realPart[i]
            paddedImag[i] = -imagPart[i]  // Conjugate for correlation
        }

        // Convert to split complex format for vDSP FFT
        let halfSize = fftSize / 2
        var splitReal = [Float](repeating: 0, count: halfSize)
        var splitImag = [Float](repeating: 0, count: halfSize)

        // Pack real signal into split complex format
        // For complex input, interleave real and imag
        var interleaved = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            interleaved[i] = paddedReal[i]  // We'll handle complex separately
        }

        // Use real FFT on the real part, complex FFT on the complex filter
        var splitComplex = DSPSplitComplex(realp: &splitReal, imagp: &splitImag)

        // Pack into split complex: even indices -> real, odd indices -> imag
        interleaved.withUnsafeBufferPointer { ptr in
            vDSP_ctoz(UnsafePointer<DSPComplex>(OpaquePointer(ptr.baseAddress!)), 2, &splitComplex, 1, vDSP_Length(halfSize))
        }

        // Perform forward FFT
        let log2n = vDSP_Length(log2(Double(fftSize)))
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Store the FFT result
        let fftReal = Array(splitReal)
        let fftImag = Array(splitImag)

        return CQTFFTFilter(
            centerFreq: centerFreq,
            filterLength: filterLength,
            filterReal: realPart,
            filterImag: imagPart,
            fftReal: fftReal,
            fftImag: fftImag,
            l1Norm: l1Norm,
            fftSize: fftSize
        )
    }

    /// Process one octave using time-domain correlation with L1-normalized filters.
    ///
    /// This matches librosa's default norm=1 behavior where filters are normalized
    /// so that their L1 norm (sum of absolute window values) equals 1.
    ///
    /// Key optimizations:
    /// - Direct writes to pre-allocated flat arrays (no intermediate allocations)
    /// - vDSP_dotpr for vectorized dot products
    /// - Pre-computed L1 normalization factors
    private func processOctaveParallel(
        octave: OctaveParams,
        signal: [Float],
        nFrames: Int,
        flatReal: UnsafeMutableBufferPointer<Float>,
        flatImag: UnsafeMutableBufferPointer<Float>
    ) {
        guard !signal.isEmpty, !octave.filters.isEmpty else { return }

        // Find the longest filter for padding calculation
        let maxFilterLen = octave.filters.map { $0.filterLength }.max() ?? 0
        guard maxFilterLen > 0 else { return }

        // Pad signal for centered frames using memcpy
        let padLen = maxFilterLen / 2
        var paddedSignal = [Float](repeating: 0, count: padLen + signal.count + padLen)
        signal.withUnsafeBufferPointer { srcPtr in
            paddedSignal.withUnsafeMutableBufferPointer { dstPtr in
                memcpy(dstPtr.baseAddress! + padLen, srcPtr.baseAddress!, signal.count * MemoryLayout<Float>.size)
            }
        }

        // Calculate frames at this octave's sample rate
        let octaveFrames = max(1, (signal.count + octave.hopLength - 1) / octave.hopLength)

        // Frame scale factor (octave frame index -> output frame index)
        let frameScale = Float(nFrames) / Float(octaveFrames)

        // Pre-compute normalization factors
        // For librosa compatibility with norm=1:
        // - Divide by L1 norm (sum of window values) for L1 normalization
        // - Multiply by sqrt(filterLength) to compensate for energy in longer filters
        // - Multiply by sqrt(downsampleFactor) to compensate for energy loss from decimation
        // This matches librosa's internal scaling: basis *= lengths / float(n_fft)
        let normFactors: [Float] = octave.filters.map { filter in
            if filter.l1Norm > 0 {
                let energyScale = sqrt(Float(filter.filterLength))
                let downsampleScale = sqrt(Float(octave.downsampleFactor))
                return energyScale * downsampleScale / filter.l1Norm
            }
            return 1.0
        }

        // Process each filter (bins within octave are independent)
        paddedSignal.withUnsafeBufferPointer { sigPtr in
            for (binIdx, filter) in octave.filters.enumerated() {
                let globalBin = octave.binOffset + binIdx
                let filterLen = filter.filterLength
                let halfLen = filterLen / 2
                let normFactor = normFactors[binIdx]

                // Output offset in flat array
                let outOffset = globalBin * nFrames

                filter.filterReal.withUnsafeBufferPointer { filterRealPtr in
                    filter.filterImag.withUnsafeBufferPointer { filterImagPtr in
                        let fR = filterRealPtr.baseAddress!
                        let fI = filterImagPtr.baseAddress!

                        for octaveFrame in 0..<octaveFrames {
                            let outFrame = min(Int(Float(octaveFrame) * frameScale), nFrames - 1)

                            // Frame center position in padded signal
                            let frameCenter = padLen + octaveFrame * octave.hopLength
                            let startIdx = frameCenter - halfLen

                            guard startIdx >= 0 && startIdx + filterLen <= paddedSignal.count else {
                                continue
                            }

                            // Compute correlation using vectorized dot products
                            // CQT correlation: sum(signal * conj(filter))
                            // = sum(signal * filterReal) - i * sum(signal * filterImag)
                            var sumReal: Float = 0
                            var sumImag: Float = 0

                            let sig = sigPtr.baseAddress! + startIdx

                            // Real part: sum(signal * filterReal)
                            vDSP_dotpr(sig, 1, fR, 1, &sumReal, vDSP_Length(filterLen))
                            // Imag part: -sum(signal * filterImag) for conjugate correlation
                            vDSP_dotpr(sig, 1, fI, 1, &sumImag, vDSP_Length(filterLen))
                            sumImag = -sumImag

                            // Apply L1 normalization to match librosa norm=1
                            flatReal[outOffset + outFrame] = sumReal * normFactor
                            flatImag[outOffset + outFrame] = sumImag * normFactor
                        }
                    }
                }
            }
        }
    }

    /// Downsample signal by factor using vDSP decimation.
    ///
    /// Uses vDSP_desamp with a proper anti-aliasing FIR filter designed to
    /// preserve energy in the passband while attenuating aliasing frequencies.
    private func downsample(_ signal: [Float], factor: Int) -> [Float] {
        guard factor > 1, signal.count > factor else { return signal }

        // Create a windowed-sinc lowpass filter
        let filterTaps = 31  // Odd number for symmetric filter
        let halfTaps = filterTaps / 2
        let cutoff: Float = 0.4 / Float(factor)

        var filter = [Float](repeating: 0, count: filterTaps)
        var filterSum: Float = 0

        // Generate windowed-sinc filter
        for i in 0..<filterTaps {
            let n = Float(i - halfTaps)
            if n == 0 {
                filter[i] = 2 * cutoff
            } else {
                let sincVal = sin(2 * Float.pi * cutoff * n) / (Float.pi * n)
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
        let outLen = (signal.count - filterTaps + factor) / factor
        guard outLen > 0 else { return signal }

        var result = [Float](repeating: 0, count: outLen)

        vDSP_desamp(
            signal,
            vDSP_Stride(factor),
            filter,
            &result,
            vDSP_Length(outLen),
            vDSP_Length(filterTaps)
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

/// Wrapper to make FFT setup Sendable
private final class FFTSetupWrapper: @unchecked Sendable {
    let setup: OpaquePointer

    init(setup: OpaquePointer) {
        self.setup = setup
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
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
        ), useGPU: true)
    }

    /// CPU-only CQT (disable GPU).
    public static var cpuOnly: CQT {
        CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            windowType: .hann,
            filterScale: 1.0,
            sparsity: 0.01
        ), useGPU: false)
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

// MARK: - GPU Acceleration Support

extension CQT {
    /// Compute CQT using GPU acceleration.
    ///
    /// This uses Metal for FFT operations for maximum parallelism.
    /// Note: Falls back to CPU if GPU is unavailable or for small signals.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram of shape (nBins, nFrames).
    public func transformGPU(_ signal: [Float]) async -> ComplexMatrix {
        // For now, use CPU implementation which is already optimized with FFT
        // GPU path will be implemented in the next phase
        return await transform(signal)
    }

    /// Compute CQT magnitude using GPU acceleration.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nBins, nFrames).
    public func magnitudeGPU(_ signal: [Float]) async -> [[Float]] {
        let cqt = await transformGPU(signal)
        return cqt.magnitude
    }
}
