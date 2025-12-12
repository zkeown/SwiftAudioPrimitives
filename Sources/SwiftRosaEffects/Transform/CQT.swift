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

/// Parameters for processing a single octave in the multi-resolution CQT.
private struct OctaveParams {
    let octave: Int
    let nBins: Int              // Number of bins in this octave
    let binOffset: Int          // Offset in output array
    let effectiveSR: Float      // Sample rate at this octave (after downsampling)
    let fftSize: Int            // FFT size for this octave
    let hopLength: Int          // Hop length at this octave
    let filterBank: [[Float]]   // Frequency-domain filters: [bin][real, imag interleaved]
}

/// Constant-Q Transform with FFT-based multi-resolution algorithm.
///
/// The Constant-Q Transform (CQT) is a time-frequency representation where
/// frequency bins are geometrically spaced (constant Q-factor), making it
/// ideal for music analysis where pitch relationships are logarithmic.
///
/// This implementation uses librosa's multi-resolution FFT approach:
/// - Process octaves from highest to lowest frequency
/// - Use FFT-based convolution (O(N log N) per octave)
/// - Downsample signal by 2 between octaves
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

    /// Pre-computed octave parameters including frequency-domain filter banks.
    private let octaveParams: [OctaveParams]

    /// FFT setups for each octave size.
    private let fftSetups: [Int: FFTSetupWrapper]

    /// Create a CQT processor.
    ///
    /// - Parameter config: CQT configuration.
    public init(config: CQTConfig = CQTConfig()) {
        self.config = config

        // Compute parameters for each octave
        var params = [OctaveParams]()
        var setups = [Int: FFTSetupWrapper]()

        let nOctaves = config.nOctaves
        var effectiveSR = config.sampleRate
        var hopLength = config.hopLength

        // Process octaves from highest (last) to lowest (first)
        for octaveIdx in (0..<nOctaves).reversed() {
            let binStart = octaveIdx * config.binsPerOctave
            let binEnd = min(binStart + config.binsPerOctave, config.nBins)
            let nBinsInOctave = binEnd - binStart

            if nBinsInOctave <= 0 { continue }

            // Find the longest filter needed for this octave (lowest frequency bin)
            let lowestFreq = config.centerFrequency(forBin: binStart)
            let filterLength = Int(ceil(config.Q * effectiveSR / lowestFreq))

            // FFT size must be power of 2 and accommodate the filter
            let fftSize = Self.nextPowerOfTwo(max(filterLength, 64))

            // Create FFT setup if not exists
            if setups[fftSize] == nil {
                let log2n = vDSP_Length(log2(Double(fftSize)))
                setups[fftSize] = FFTSetupWrapper(log2n: log2n)
            }

            // Create frequency-domain filter bank for this octave
            var filterBank = [[Float]]()
            for bin in binStart..<binEnd {
                let centerFreq = config.centerFrequency(forBin: bin)
                let filter = Self.createFrequencyDomainFilter(
                    centerFreq: centerFreq,
                    Q: config.Q,
                    effectiveSR: effectiveSR,
                    fftSize: fftSize,
                    windowType: config.windowType
                )
                filterBank.append(filter)
            }

            params.append(OctaveParams(
                octave: octaveIdx,
                nBins: nBinsInOctave,
                binOffset: binStart,
                effectiveSR: effectiveSR,
                fftSize: fftSize,
                hopLength: hopLength,
                filterBank: filterBank
            ))

            // Downsample for next (lower frequency) octave
            effectiveSR /= 2
            hopLength /= 2
        }

        // Reverse to process from highest to lowest
        self.octaveParams = params.reversed()
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

        // Initialize output arrays
        var realOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)
        var imagOutput = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nBins)

        // Process octaves from highest to lowest frequency
        var currentSignal = signal

        for octave in octaveParams {
            guard let fftSetup = fftSetups[octave.fftSize] else { continue }

            // Pad signal for this octave
            let padLength = octave.fftSize / 2
            let paddedSignal = padSignal(currentSignal, padLength: padLength)

            // Compute CQT for this octave using FFT-based convolution
            processOctave(
                octave: octave,
                paddedSignal: paddedSignal,
                fftSetup: fftSetup,
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

    /// Process one octave using FFT-based convolution.
    private func processOctave(
        octave: OctaveParams,
        paddedSignal: [Float],
        fftSetup: FFTSetupWrapper,
        nFrames: Int,
        realOutput: inout [[Float]],
        imagOutput: inout [[Float]]
    ) {
        let fftSize = octave.fftSize
        let halfN = fftSize / 2
        let log2n = vDSP_Length(log2(Double(fftSize)))

        // Calculate frames at this octave's sample rate
        let octaveFrames = (paddedSignal.count - fftSize) / octave.hopLength + 1
        guard octaveFrames > 0 else { return }

        // Compute frame scale factor (octave frame index -> output frame index)
        let frameScale = Float(nFrames) / Float(octaveFrames)

        // Allocate working buffers
        var windowedFrame = [Float](repeating: 0, count: fftSize)
        var splitReal = [Float](repeating: 0, count: halfN)
        var splitImag = [Float](repeating: 0, count: halfN)
        var productReal = [Float](repeating: 0, count: halfN)
        var productImag = [Float](repeating: 0, count: halfN)

        // Generate window
        let window = Windows.generate(config.windowType, length: fftSize, periodic: true)

        paddedSignal.withUnsafeBufferPointer { signalPtr in
            for octaveFrame in 0..<octaveFrames {
                let start = octaveFrame * octave.hopLength

                // Window the frame
                vDSP_vmul(
                    signalPtr.baseAddress! + start, 1,
                    window, 1,
                    &windowedFrame, 1,
                    vDSP_Length(fftSize)
                )

                // FFT
                var split = DSPSplitComplex(realp: &splitReal, imagp: &splitImag)
                windowedFrame.withUnsafeBufferPointer { framePtr in
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(framePtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(halfN)
                    )
                }

                vDSP_fft_zrip(fftSetup.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                // Scale by 0.5 (vDSP convention)
                var scale: Float = 0.5
                vDSP_vsmul(splitReal, 1, &scale, &splitReal, 1, vDSP_Length(halfN))
                vDSP_vsmul(splitImag, 1, &scale, &splitImag, 1, vDSP_Length(halfN))

                // Map to output frame
                let outFrame = min(Int(Float(octaveFrame) * frameScale), nFrames - 1)

                // Apply each frequency-domain filter
                for (binIdx, filter) in octave.filterBank.enumerated() {
                    let globalBin = octave.binOffset + binIdx

                    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    // Filter is stored as [real0, imag0, real1, imag1, ...]
                    var sumReal: Float = 0
                    var sumImag: Float = 0

                    for k in 0..<halfN {
                        let filterReal = filter[2 * k]
                        let filterImag = filter[2 * k + 1]
                        let sigReal = splitReal[k]
                        let sigImag = splitImag[k]

                        // Complex multiply and accumulate
                        sumReal += sigReal * filterReal - sigImag * filterImag
                        sumImag += sigReal * filterImag + sigImag * filterReal
                    }

                    // Also include DC and Nyquist (packed in special format)
                    // DC is at real[0], Nyquist is at imag[0] in packed format

                    realOutput[globalBin][outFrame] = sumReal
                    imagOutput[globalBin][outFrame] = sumImag
                }
            }
        }
    }

    /// Create frequency-domain filter for one CQT bin.
    private static func createFrequencyDomainFilter(
        centerFreq: Float,
        Q: Float,
        effectiveSR: Float,
        fftSize: Int,
        windowType: WindowType
    ) -> [Float] {
        let halfN = fftSize / 2

        // Time-domain filter length
        let filterLength = Int(ceil(Q * effectiveSR / centerFreq))
        let actualLength = min(filterLength, fftSize)

        // Generate windowed complex sinusoid in time domain
        let window = Windows.generate(windowType, length: actualLength, periodic: true)

        let phaseIncrement = 2.0 * Float.pi * centerFreq / effectiveSR
        var realKernel = [Float](repeating: 0, count: fftSize)
        var imagKernel = [Float](repeating: 0, count: fftSize)

        // Generate windowed sinusoid centered in the frame
        let offset = (fftSize - actualLength) / 2
        for i in 0..<actualLength {
            let phase = phaseIncrement * Float(i)
            realKernel[offset + i] = window[i] * cos(phase)
            imagKernel[offset + i] = window[i] * sin(phase)
        }

        // Compute L1 norm for normalization
        var l1Norm: Float = 0
        for i in 0..<fftSize {
            l1Norm += sqrt(realKernel[i] * realKernel[i] + imagKernel[i] * imagKernel[i])
        }

        // Normalize by L1 norm and scale by sqrt(filterLength)
        let sqrtN = sqrt(Float(filterLength))
        let normFactor = l1Norm > 0 ? sqrtN / l1Norm : 0

        for i in 0..<fftSize {
            realKernel[i] *= normFactor
            imagKernel[i] *= -normFactor  // Conjugate for correlation
        }

        // FFT to get frequency-domain filter
        let log2n = vDSP_Length(log2(Double(fftSize)))
        let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var splitReal = [Float](repeating: 0, count: halfN)
        var splitImag = [Float](repeating: 0, count: halfN)
        var split = DSPSplitComplex(realp: &splitReal, imagp: &splitImag)

        // Pack real kernel
        realKernel.withUnsafeBufferPointer { ptr in
            vDSP_ctoz(
                UnsafePointer<DSPComplex>(OpaquePointer(ptr.baseAddress!)),
                2,
                &split,
                1,
                vDSP_Length(halfN)
            )
        }

        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Pack imaginary kernel and FFT it
        var imagSplitReal = [Float](repeating: 0, count: halfN)
        var imagSplitImag = [Float](repeating: 0, count: halfN)
        var imagSplit = DSPSplitComplex(realp: &imagSplitReal, imagp: &imagSplitImag)

        imagKernel.withUnsafeBufferPointer { ptr in
            vDSP_ctoz(
                UnsafePointer<DSPComplex>(OpaquePointer(ptr.baseAddress!)),
                2,
                &imagSplit,
                1,
                vDSP_Length(halfN)
            )
        }

        vDSP_fft_zrip(fftSetup, &imagSplit, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // Combine into complex filter: filter_freq = FFT(real_kernel) + j*FFT(imag_kernel)
        // Store interleaved for efficient access
        var result = [Float](repeating: 0, count: halfN * 2)
        for k in 0..<halfN {
            // The actual frequency response is the FFT of the complex time-domain filter
            // For a complex signal x + jy: FFT(x + jy) = FFT(x) + j*FFT(y)
            result[2 * k] = splitReal[k]      // Real part
            result[2 * k + 1] = splitImag[k]  // Imag part (from real kernel FFT)
        }

        return result
    }

    /// Downsample signal by factor using anti-aliasing filter.
    private func downsample(_ signal: [Float], factor: Int) -> [Float] {
        guard factor > 1, signal.count > factor else { return signal }

        // Apply simple anti-aliasing lowpass filter (moving average)
        let filterLen = factor * 2
        var filtered = [Float](repeating: 0, count: signal.count)

        signal.withUnsafeBufferPointer { sigPtr in
            filtered.withUnsafeMutableBufferPointer { filtPtr in
                for i in 0..<signal.count {
                    var sum: Float = 0
                    var count = 0
                    for j in max(0, i - filterLen/2)..<min(signal.count, i + filterLen/2) {
                        sum += sigPtr[j]
                        count += 1
                    }
                    filtPtr[i] = sum / Float(count)
                }
            }
        }

        // Decimate
        let outLen = (filtered.count + factor - 1) / factor
        var result = [Float](repeating: 0, count: outLen)
        for i in 0..<outLen {
            result[i] = filtered[i * factor]
        }

        return result
    }

    private func padSignal(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var result = [Float]()
        result.reserveCapacity(signal.count + 2 * padLength)

        // Zero padding (constant mode) to match librosa default
        result.append(contentsOf: [Float](repeating: 0, count: padLength))
        result.append(contentsOf: signal)
        result.append(contentsOf: [Float](repeating: 0, count: padLength))

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
