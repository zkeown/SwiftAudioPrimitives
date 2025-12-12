import Accelerate
import Foundation
import SwiftRosaCore

/// Errors that can occur during VQT computation.
public enum VQTError: Error, Sendable {
    case invalidParameters(String)
    case computationError(String)
}

/// Configuration for Variable-Q Transform.
public struct VQTConfig: Sendable {
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

    /// Gamma parameter (minimum bandwidth floor in Hz).
    /// Set to 0 for pure CQT behavior.
    /// Set to a positive value for VQT with improved low-frequency resolution.
    public let gamma: Float

    /// Sparsity threshold for kernel pruning.
    public let sparsity: Float

    /// Create VQT configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - fMin: Minimum frequency (default: 32.70 Hz, C1).
    ///   - nBins: Number of frequency bins (default: 84, 7 octaves).
    ///   - binsPerOctave: Bins per octave (default: 12).
    ///   - windowType: Window type (default: hann).
    ///   - filterScale: Filter scale (default: 1.0).
    ///   - gamma: Bandwidth floor in Hz (default: 0, equivalent to CQT).
    ///   - sparsity: Sparsity threshold (default: 0.01).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        fMin: Float = 32.70,
        nBins: Int = 84,
        binsPerOctave: Int = 12,
        windowType: WindowType = .hann,
        filterScale: Float = 1.0,
        gamma: Float = 0,
        sparsity: Float = 0.01
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.fMin = fMin
        self.nBins = nBins
        self.binsPerOctave = binsPerOctave
        self.windowType = windowType
        self.filterScale = filterScale
        self.gamma = gamma
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

    /// Reference Q factor (for CQT-like behavior).
    public var QRef: Float {
        filterScale / (pow(2.0, 1.0 / Float(binsPerOctave)) - 1)
    }

    /// Get center frequency for a bin.
    public func centerFrequency(forBin bin: Int) -> Float {
        fMin * pow(2.0, Float(bin) / Float(binsPerOctave))
    }

    /// Get bandwidth for a bin (variable depending on gamma).
    public func bandwidth(forBin bin: Int) -> Float {
        let freq = centerFrequency(forBin: bin)
        return freq / QRef + gamma
    }

    /// Get Q factor for a bin.
    public func Q(forBin bin: Int) -> Float {
        let freq = centerFrequency(forBin: bin)
        let bw = bandwidth(forBin: bin)
        return freq / bw
    }
}

/// VQT kernel specification.
private struct VQTKernel: Sendable {
    let realKernel: [Float]
    let imagKernel: [Float]
    let filterLength: Int
    let centerFrequency: Float
    let bandwidth: Float
    let Q: Float
}

/// Variable-Q Transform.
///
/// The Variable-Q Transform extends the Constant-Q Transform by allowing
/// the bandwidth to vary with frequency. The gamma parameter adds a constant
/// floor to the bandwidth, providing better low-frequency resolution.
///
/// ## Key Differences from CQT
///
/// - **CQT**: `bandwidth[k] = f[k] / Q` (proportional to frequency)
/// - **VQT**: `bandwidth[k] = f[k] / Q + gamma` (floor on bandwidth)
///
/// At low frequencies, gamma dominates, giving better time resolution.
/// At high frequencies, the bandwidth approaches CQT behavior.
///
/// ## Example Usage
///
/// ```swift
/// // Standard VQT with gamma for ERB-like behavior
/// let vqt = VQT(config: VQTConfig(fMin: 32.70, nBins: 84, gamma: 20))
///
/// // Compute VQT
/// let spectrum = await vqt.transform(audioSignal)
///
/// // Get magnitude only
/// let magnitude = await vqt.magnitude(audioSignal)
/// ```
///
/// ## When to Use VQT vs CQT
///
/// - **Use VQT** when analyzing low-frequency content where you need better
///   time resolution (e.g., bass notes, low drums)
/// - **Use CQT** when you want strict logarithmic frequency resolution
///   (e.g., pitch tracking, music transcription)
public struct VQT: Sendable {
    /// Configuration.
    public let config: VQTConfig

    /// Pre-computed VQT kernels.
    private let kernels: [VQTKernel]

    /// Create a VQT processor.
    ///
    /// - Parameter config: VQT configuration.
    public init(config: VQTConfig = VQTConfig()) {
        self.config = config
        self.kernels = Self.computeKernels(config: config)
    }

    /// Compute VQT of signal.
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
                let halfLen = kernel.filterLength / 2

                kernel.realKernel.withUnsafeBufferPointer { realKernelPtr in
                    kernel.imagKernel.withUnsafeBufferPointer { imagKernelPtr in
                        for frameIdx in 0..<nFrames {
                            let center = frameIdx * config.hopLength + padLength / 2
                            let start = max(0, center - halfLen)
                            let end = min(paddedSignal.count, center + halfLen)
                            let kernelOffset = start - (center - halfLen)

                            guard end > start && kernelOffset >= 0 else { continue }

                            let length = min(end - start, kernel.filterLength - kernelOffset)
                            guard length > 0 else { continue }

                            var realDot: Float = 0
                            var imagDot: Float = 0

                            vDSP_dotpr(
                                signalPtr.baseAddress! + start, 1,
                                realKernelPtr.baseAddress! + kernelOffset, 1,
                                &realDot,
                                vDSP_Length(length)
                            )

                            vDSP_dotpr(
                                signalPtr.baseAddress! + start, 1,
                                imagKernelPtr.baseAddress! + kernelOffset, 1,
                                &imagDot,
                                vDSP_Length(length)
                            )

                            realOutput[binIdx][frameIdx] = realDot
                            imagOutput[binIdx][frameIdx] = imagDot
                        }
                    }
                }
            }
        }

        return ComplexMatrix(real: realOutput, imag: imagOutput)
    }

    /// Compute magnitude spectrogram.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nBins, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let complex = await transform(signal)
        return computeMagnitude(complex)
    }

    /// Compute power spectrogram.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram of shape (nBins, nFrames).
    public func power(_ signal: [Float]) async -> [[Float]] {
        let complex = await transform(signal)
        return computePower(complex)
    }

    /// Compute phase spectrogram.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Phase spectrogram of shape (nBins, nFrames).
    public func phase(_ signal: [Float]) async -> [[Float]] {
        let complex = await transform(signal)
        return computePhase(complex)
    }

    /// Get the center frequencies for all bins.
    ///
    /// - Returns: Array of center frequencies in Hz.
    public var frequencies: [Float] {
        (0..<config.nBins).map { config.centerFrequency(forBin: $0) }
    }

    /// Get the bandwidths for all bins.
    ///
    /// - Returns: Array of bandwidths in Hz.
    public var bandwidths: [Float] {
        (0..<config.nBins).map { config.bandwidth(forBin: $0) }
    }

    /// Get the Q factors for all bins.
    ///
    /// - Returns: Array of Q factors.
    public var qFactors: [Float] {
        (0..<config.nBins).map { config.Q(forBin: $0) }
    }

    // MARK: - Private Implementation

    /// Compute VQT kernels with variable bandwidth.
    private static func computeKernels(config: VQTConfig) -> [VQTKernel] {
        var kernels = [VQTKernel]()
        kernels.reserveCapacity(config.nBins)

        for binIdx in 0..<config.nBins {
            let centerFreq = config.centerFrequency(forBin: binIdx)
            let bandwidth = config.bandwidth(forBin: binIdx)
            let Q = config.Q(forBin: binIdx)

            // Filter length based on Q factor
            let filterLength = Int(ceil(Q * config.sampleRate / centerFreq * config.filterScale))
            let evenLength = filterLength + (filterLength % 2)  // Make even

            // Generate windowed sinusoid kernel
            let window = Windows.generate(config.windowType, length: evenLength, periodic: false)

            var realKernel = [Float](repeating: 0, count: evenLength)
            var imagKernel = [Float](repeating: 0, count: evenLength)

            let angularFreq = 2.0 * Float.pi * centerFreq / config.sampleRate

            for i in 0..<evenLength {
                let t = Float(i - evenLength / 2)
                let phase = angularFreq * t

                // Complex exponential times window
                realKernel[i] = window[i] * cos(phase)
                imagKernel[i] = window[i] * sin(phase)
            }

            // Normalize kernel
            var norm: Float = 0
            vDSP_dotpr(window, 1, window, 1, &norm, vDSP_Length(evenLength))
            norm = sqrt(norm)

            if norm > 0 {
                var scale = 1.0 / norm
                vDSP_vsmul(realKernel, 1, &scale, &realKernel, 1, vDSP_Length(evenLength))
                vDSP_vsmul(imagKernel, 1, &scale, &imagKernel, 1, vDSP_Length(evenLength))
            }

            // Apply sparsity threshold (prune small values)
            if config.sparsity > 0 {
                for i in 0..<evenLength {
                    if abs(realKernel[i]) < config.sparsity {
                        realKernel[i] = 0
                    }
                    if abs(imagKernel[i]) < config.sparsity {
                        imagKernel[i] = 0
                    }
                }
            }

            kernels.append(VQTKernel(
                realKernel: realKernel,
                imagKernel: imagKernel,
                filterLength: evenLength,
                centerFrequency: centerFreq,
                bandwidth: bandwidth,
                Q: Q
            ))
        }

        return kernels
    }

    private func padSignal(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var padded = [Float]()
        padded.reserveCapacity(signal.count + 2 * padLength)

        // Reflect padding
        for i in (1...padLength).reversed() {
            let idx = i % signal.count
            padded.append(signal[idx])
        }

        padded.append(contentsOf: signal)

        for i in 1...padLength {
            let idx = (signal.count - 1) - (i % signal.count)
            padded.append(signal[max(0, idx)])
        }

        return padded
    }

    private func computeMagnitude(_ complex: ComplexMatrix) -> [[Float]] {
        guard !complex.real.isEmpty else { return [] }

        var magnitude = [[Float]]()
        magnitude.reserveCapacity(complex.real.count)

        for binIdx in 0..<complex.real.count {
            let nFrames = complex.real[binIdx].count
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                let real = complex.real[binIdx][frameIdx]
                let imag = complex.imag[binIdx][frameIdx]
                row[frameIdx] = sqrt(real * real + imag * imag)
            }

            magnitude.append(row)
        }

        return magnitude
    }

    private func computePower(_ complex: ComplexMatrix) -> [[Float]] {
        guard !complex.real.isEmpty else { return [] }

        var power = [[Float]]()
        power.reserveCapacity(complex.real.count)

        for binIdx in 0..<complex.real.count {
            let nFrames = complex.real[binIdx].count
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                let real = complex.real[binIdx][frameIdx]
                let imag = complex.imag[binIdx][frameIdx]
                row[frameIdx] = real * real + imag * imag
            }

            power.append(row)
        }

        return power
    }

    private func computePhase(_ complex: ComplexMatrix) -> [[Float]] {
        guard !complex.real.isEmpty else { return [] }

        var phase = [[Float]]()
        phase.reserveCapacity(complex.real.count)

        for binIdx in 0..<complex.real.count {
            let nFrames = complex.real[binIdx].count
            var row = [Float](repeating: 0, count: nFrames)

            for frameIdx in 0..<nFrames {
                let real = complex.real[binIdx][frameIdx]
                let imag = complex.imag[binIdx][frameIdx]
                row[frameIdx] = atan2(imag, real)
            }

            phase.append(row)
        }

        return phase
    }
}

// MARK: - Presets

extension VQT {
    /// Standard VQT (equivalent to CQT, gamma=0).
    public static var standard: VQT {
        VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            gamma: 0
        ))
    }

    /// VQT with ERB-like bandwidth (gamma matches ERB at ~500 Hz).
    public static var erb: VQT {
        // Compute gamma to match ERB at 500 Hz
        let gamma = MRFrequencies.gammaForERB(at: 500.0, binsPerOctave: 12)
        return VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 32.70,
            nBins: 84,
            binsPerOctave: 12,
            gamma: gamma
        ))
    }

    /// VQT for bass analysis (lower fMin, higher gamma).
    public static var bass: VQT {
        VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 27.50,  // A0
            nBins: 60,    // 5 octaves
            binsPerOctave: 12,
            gamma: 30     // Higher gamma for better time resolution at low frequencies
        ))
    }

    /// VQT for music analysis (piano range).
    public static var music: VQT {
        VQT(config: VQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            fMin: 27.50,  // A0
            nBins: 88,    // Full piano range
            binsPerOctave: 12,
            gamma: 20
        ))
    }
}
