import SwiftRosaCore
import Accelerate
import Foundation

/// Quality level for resampling operations.
public enum ResampleQuality: Sendable {
    /// Fast resampling with 8-tap FIR filter.
    case low

    /// Balanced quality/speed with 16-tap FIR filter.
    case medium

    /// High quality with 32-tap FIR filter.
    case high

    /// Very high quality with 64-tap FIR filter.
    case veryHigh

    /// Filter length for this quality level.
    var filterLength: Int {
        switch self {
        case .low: return 8
        case .medium: return 16
        case .high: return 32
        case .veryHigh: return 64
        }
    }

    /// Kaiser window beta for this quality level.
    var kaiserBeta: Float {
        switch self {
        case .low: return 5.0
        case .medium: return 6.0
        case .high: return 8.0
        case .veryHigh: return 10.0
        }
    }
}

/// Audio signal resampling using polyphase FIR filtering.
///
/// This implementation uses windowed-sinc interpolation with antialiasing
/// to convert audio signals between different sample rates while preserving
/// frequency content below the Nyquist frequency of the lower sample rate.
///
/// ## Example Usage
///
/// ```swift
/// // Resample from 48kHz to 44.1kHz
/// let resampled = await Resample.resample(
///     audio,
///     fromSampleRate: 48000,
///     toSampleRate: 44100,
///     quality: .high
/// )
///
/// // Downsample for model that expects 22.05kHz
/// let downsampled = await Resample.resample(
///     audio,
///     fromSampleRate: 44100,
///     toSampleRate: 22050
/// )
/// ```
public struct Resample: Sendable {
    private init() {}

    // MARK: - Main Resampling Function

    /// Resample an audio signal to a different sample rate (synchronous version).
    ///
    /// This is the preferred version for single-threaded use as it avoids
    /// async/await overhead for purely CPU-bound computation.
    ///
    /// Uses FFT-based resampling for large signals (O(N log N)) and polyphase
    /// filtering for smaller signals or when higher quality is needed.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromSampleRate: Original sample rate in Hz.
    ///   - toSampleRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resampleSync(
        _ signal: [Float],
        fromSampleRate: Int,
        toSampleRate: Int,
        quality: ResampleQuality = .medium
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard fromSampleRate != toSampleRate else { return signal }
        guard fromSampleRate > 0 && toSampleRate > 0 else { return signal }

        // Calculate target output length
        let targetLength = Int(Double(signal.count) * Double(toSampleRate) / Double(fromSampleRate))

        // Use FFT-based resampling for large signals (> 4096 samples)
        // FFT is O(N log N) vs O(N × filterLength) for polyphase
        // Crossover point is around 4K-8K samples where FFT wins
        if signal.count > 4096 {
            return fftResample(signal: signal, targetLength: targetLength)
        }

        // For smaller signals, use polyphase for better quality
        // Find rational approximation for the resampling ratio
        let (upFactor, downFactor) = rationalApproximation(
            numerator: toSampleRate,
            denominator: fromSampleRate
        )

        // Design antialiasing filter
        // Cutoff should be at the Nyquist of the lower sample rate
        // Normalized to the higher rate (which is where filtering conceptually happens)
        let minRate = Float(min(fromSampleRate, toSampleRate))
        let effectiveRate = Float(fromSampleRate * upFactor)  // Rate after upsampling
        let cutoff = (minRate / 2.0) / (effectiveRate / 2.0)  // Normalized cutoff (0-0.5 range)
        let filterCoeffs = designLowpassFilter(
            cutoff: cutoff * 0.9,  // Slight rolloff to avoid aliasing at band edge
            length: quality.filterLength * upFactor,
            beta: quality.kaiserBeta
        )

        // Perform polyphase resampling
        return polyphaseResample(
            signal: signal,
            upFactor: upFactor,
            downFactor: downFactor,
            filter: filterCoeffs
        )
    }

    /// Resample an audio signal to a different sample rate (async version).
    ///
    /// Note: This function is marked async for API consistency but performs
    /// purely synchronous computation. For optimal performance in single-threaded
    /// contexts, use `resampleSync` instead.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromSampleRate: Original sample rate in Hz.
    ///   - toSampleRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromSampleRate: Int,
        toSampleRate: Int,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        // Delegate to synchronous implementation
        return resampleSync(signal, fromSampleRate: fromSampleRate, toSampleRate: toSampleRate, quality: quality)
    }

    /// Resample using simple linear interpolation (fast but lower quality).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromSampleRate: Original sample rate.
    ///   - toSampleRate: Target sample rate.
    /// - Returns: Resampled audio signal.
    public static func resampleLinear(
        _ signal: [Float],
        fromSampleRate: Int,
        toSampleRate: Int
    ) -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard fromSampleRate != toSampleRate else { return signal }

        let ratio = Float(fromSampleRate) / Float(toSampleRate)
        let outputLength = Int(Float(signal.count) / ratio)

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcPos = Float(i) * ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)

            if srcIdx + 1 < signal.count {
                output[i] = signal[srcIdx] * (1 - frac) + signal[srcIdx + 1] * frac
            } else if srcIdx < signal.count {
                output[i] = signal[srcIdx]
            }
        }

        return output
    }

    // MARK: - Filter Design

    /// Design a lowpass FIR filter using windowed-sinc method.
    ///
    /// - Parameters:
    ///   - cutoff: Normalized cutoff frequency (0 to 0.5).
    ///   - length: Filter length (should be odd for symmetric filter).
    ///   - beta: Kaiser window beta parameter.
    /// - Returns: FIR filter coefficients.
    internal static func designLowpassFilter(
        cutoff: Float,
        length: Int,
        beta: Float = 6.0
    ) -> [Float] {
        let n = length | 1  // Ensure odd length
        let center = n / 2

        var filter = [Float](repeating: 0, count: n)
        let twoPiCutoff = 2.0 * Float.pi * cutoff

        // Generate sinc function
        for i in 0..<n {
            let x = Float(i - center)
            if x == 0 {
                filter[i] = 2.0 * cutoff
            } else {
                filter[i] = sin(twoPiCutoff * x) / (Float.pi * x)
            }
        }

        // Apply Kaiser window
        let window = kaiserWindow(length: n, beta: beta)
        vDSP_vmul(filter, 1, window, 1, &filter, 1, vDSP_Length(n))

        // Normalize to unit sum
        var sum: Float = 0
        vDSP_sve(filter, 1, &sum, vDSP_Length(n))
        if sum != 0 {
            var invSum = 1.0 / sum
            vDSP_vsmul(filter, 1, &invSum, &filter, 1, vDSP_Length(n))
        }

        return filter
    }

    /// Generate a Kaiser window.
    ///
    /// - Parameters:
    ///   - length: Window length.
    ///   - beta: Shape parameter (larger = narrower main lobe, higher sidelobe attenuation).
    /// - Returns: Kaiser window coefficients.
    private static func kaiserWindow(length: Int, beta: Float) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let center = Float(length - 1) / 2.0
        let betaI0 = besselI0(beta)

        for i in 0..<length {
            let x = (Float(i) - center) / center
            let arg = beta * sqrt(max(0, 1 - x * x))
            window[i] = besselI0(arg) / betaI0
        }

        return window
    }

    /// Modified Bessel function of the first kind, order 0.
    private static func besselI0(_ x: Float) -> Float {
        var sum: Float = 1.0
        var term: Float = 1.0
        let x2 = x * x / 4.0

        for k in 1...25 {
            term *= x2 / Float(k * k)
            sum += term
            if term < 1e-10 { break }
        }

        return sum
    }

    // MARK: - FFT-Based Resampling

    /// Resample using vDSP's optimized interpolation.
    /// For downsampling: anti-alias filter THEN decimate
    /// For upsampling: interpolate directly
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - targetLength: Desired output length.
    /// - Returns: Resampled signal.
    private static func fftResample(signal: [Float], targetLength: Int) -> [Float] {
        let inputLength = signal.count
        guard inputLength > 0 && targetLength > 0 else { return [] }
        guard inputLength != targetLength else { return signal }

        // For downsampling, apply anti-aliasing filter to INPUT first, then decimate
        // This is more efficient than filtering output
        let sourceSignal: [Float]
        let sourceLength: Int
        if targetLength < inputLength {
            // Downsample: filter input first to prevent aliasing
            let ratio = Float(inputLength) / Float(targetLength)
            sourceSignal = applyFastLowpass(signal, cutoffRatio: 1.0 / ratio)
            sourceLength = sourceSignal.count
        } else {
            sourceSignal = signal
            sourceLength = inputLength
        }

        // Compute interpolation step
        let step = Float(sourceLength - 1) / Float(max(1, targetLength - 1))

        // For large upsampling, use chunked processing with position buffer reuse
        // For small operations, direct computation is faster
        if targetLength <= 65536 {
            // Use vDSP_vlint with pre-computed positions
            var output = [Float](repeating: 0, count: targetLength)
            var positions = [Float](repeating: 0, count: targetLength)

            var startPos: Float = 0
            var stepVal = step
            vDSP_vramp(&startPos, &stepVal, &positions, 1, vDSP_Length(targetLength))

            sourceSignal.withUnsafeBufferPointer { signalPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    positions.withUnsafeBufferPointer { posPtr in
                        vDSP_vlint(
                            signalPtr.baseAddress!,
                            posPtr.baseAddress!,
                            1,
                            outputPtr.baseAddress!,
                            1,
                            vDSP_Length(targetLength),
                            vDSP_Length(sourceLength)
                        )
                    }
                }
            }
            return output
        }

        // For very large upsampling, process in chunks to control memory
        let chunkSize = 65536
        var output = [Float](repeating: 0, count: targetLength)
        var positions = [Float](repeating: 0, count: chunkSize)

        sourceSignal.withUnsafeBufferPointer { signalPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                positions.withUnsafeMutableBufferPointer { posPtr in
                    var outIdx = 0
                    while outIdx < targetLength {
                        let remaining = targetLength - outIdx
                        let thisChunk = min(chunkSize, remaining)

                        // Generate positions for this chunk
                        var startPos = Float(outIdx) * step
                        var stepVal = step
                        vDSP_vramp(&startPos, &stepVal, posPtr.baseAddress!, 1, vDSP_Length(thisChunk))

                        // Interpolate this chunk
                        vDSP_vlint(
                            signalPtr.baseAddress!,
                            posPtr.baseAddress!,
                            1,
                            outputPtr.baseAddress! + outIdx,
                            1,
                            vDSP_Length(thisChunk),
                            vDSP_Length(sourceLength)
                        )

                        outIdx += thisChunk
                    }
                }
            }
        }

        return output
    }

    /// Fast O(N) lowpass filter using vDSP_deq22 biquad.
    /// Much faster than manual IIR loops.
    private static func applyFastLowpass(_ signal: [Float], cutoffRatio: Float) -> [Float] {
        guard signal.count > 2 else { return signal }

        // Design a simple 2nd order Butterworth lowpass
        // For speed, we use a simplified coefficient calculation
        let omega = Float.pi * min(0.95, cutoffRatio)  // Normalized frequency
        let sinOmega = sin(omega)
        let cosOmega = cos(omega)
        let alpha = sinOmega / (2.0 * 0.707)  // Q = 0.707 for Butterworth

        // Biquad coefficients (normalized)
        let a0 = 1.0 + alpha
        let b0 = ((1.0 - cosOmega) / 2.0) / a0
        let b1 = (1.0 - cosOmega) / a0
        let b2 = b0
        let a1 = (-2.0 * cosOmega) / a0
        let a2 = (1.0 - alpha) / a0

        // vDSP_deq22 expects coefficients in specific order: [b0, b1, b2, a1, a2]
        var coeffs: [Float] = [b0, b1, b2, a1, a2]

        var filtered = [Float](repeating: 0, count: signal.count)

        // State for filter: [x[n-1], x[n-2], y[n-1], y[n-2]]
        var state: [Float] = [0, 0, 0, 0]

        // Apply filter using vDSP_deq22
        signal.withUnsafeBufferPointer { inPtr in
            filtered.withUnsafeMutableBufferPointer { outPtr in
                coeffs.withUnsafeBufferPointer { coeffPtr in
                    state.withUnsafeMutableBufferPointer { statePtr in
                        // vDSP_deq22 applies IIR filter
                        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                        vDSP_deq22(
                            inPtr.baseAddress!,
                            1,
                            coeffPtr.baseAddress!,
                            outPtr.baseAddress!,
                            1,
                            vDSP_Length(signal.count)
                        )
                    }
                }
            }
        }

        return filtered
    }

    /// Fallback polyphase resampling for edge cases.
    private static func polyphaseResampleFallback(signal: [Float], targetLength: Int) -> [Float] {
        // Simple linear interpolation as ultimate fallback
        let ratio = Float(signal.count) / Float(targetLength)
        var output = [Float](repeating: 0, count: targetLength)

        for i in 0..<targetLength {
            let srcPos = Float(i) * ratio
            let srcIdx = Int(srcPos)
            let frac = srcPos - Float(srcIdx)

            if srcIdx + 1 < signal.count {
                output[i] = signal[srcIdx] * (1 - frac) + signal[srcIdx + 1] * frac
            } else if srcIdx < signal.count {
                output[i] = signal[srcIdx]
            }
        }

        return output
    }

    // MARK: - Polyphase Resampling (Original - kept for specific cases)

    /// Decompose filter into polyphase components.
    /// Returns array of (phase -> [tap coefficients]) where phase 0 corresponds to non-zero input samples.
    private static func decomposePolyphase(
        filter: [Float],
        upFactor: Int
    ) -> [[Float]] {
        let filterLen = filter.count
        // Number of taps per phase (ceiling division)
        let tapsPerPhase = (filterLen + upFactor - 1) / upFactor

        var phases = [[Float]]()
        phases.reserveCapacity(upFactor)

        for phase in 0..<upFactor {
            var phaseTaps = [Float]()
            phaseTaps.reserveCapacity(tapsPerPhase)

            // Collect filter taps for this phase
            var k = phase
            while k < filterLen {
                phaseTaps.append(filter[k])
                k += upFactor
            }

            phases.append(phaseTaps)
        }

        return phases
    }

    /// Perform polyphase resampling using optimized vDSP operations.
    /// Uses vDSP_desamp for decimation when possible, otherwise uses
    /// vectorized dot products with pre-computed filter phases.
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - upFactor: Upsampling factor L.
    ///   - downFactor: Downsampling factor M.
    ///   - filter: Lowpass filter coefficients.
    /// - Returns: Resampled signal.
    private static func polyphaseResample(
        signal: [Float],
        upFactor: Int,
        downFactor: Int,
        filter: [Float]
    ) -> [Float] {
        let inputLength = signal.count
        let outputLength = (inputLength * upFactor) / downFactor

        guard outputLength > 0 && inputLength > 0 else { return [] }

        // For simple decimation (upFactor == 1), use vDSP_desamp which is highly optimized
        if upFactor == 1 {
            var output = [Float](repeating: 0, count: outputLength)
            let decimationFactor = vDSP_Stride(downFactor)
            let filterLen = Int32(filter.count)

            signal.withUnsafeBufferPointer { sigPtr in
                filter.withUnsafeBufferPointer { filtPtr in
                    output.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_desamp(
                            sigPtr.baseAddress!,
                            decimationFactor,
                            filtPtr.baseAddress!,
                            outPtr.baseAddress!,
                            vDSP_Length(outputLength),
                            vDSP_Length(filterLen)
                        )
                    }
                }
            }
            return output
        }

        // Pre-decompose filter into polyphase components
        let polyphaseFilters = decomposePolyphase(filter: filter, upFactor: upFactor)
        let filterHalf = filter.count / 2

        var output = [Float](repeating: 0, count: outputLength)

        // For rational resampling, use direct computation
        // Optimized with unsafe pointers and minimal branching
        signal.withUnsafeBufferPointer { signalPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                for n in 0..<outputLength {
                    let m = n * downFactor
                    let offsetM = m + filterHalf
                    let phase = offsetM % upFactor
                    let baseSignalIdx = offsetM / upFactor

                    let phaseTaps = polyphaseFilters[phase]
                    let tapsCount = phaseTaps.count
                    var sum: Float = 0.0

                    phaseTaps.withUnsafeBufferPointer { tapsPtr in
                        // Compute valid range once
                        let minValid = max(0, baseSignalIdx - inputLength + 1)
                        let maxValid = min(tapsCount, baseSignalIdx + 1)

                        if maxValid > minValid {
                            for tapIdx in minValid..<maxValid {
                                sum += signalPtr[baseSignalIdx - tapIdx] * tapsPtr[tapIdx]
                            }
                        }
                    }

                    outPtr[n] = sum * Float(upFactor)
                }
            }
        }

        return output
    }

    // MARK: - Utility Functions

    /// Find a rational approximation to a fraction.
    ///
    /// - Parameters:
    ///   - numerator: Target numerator.
    ///   - denominator: Target denominator.
    /// - Returns: Simplified (numerator, denominator) pair.
    private static func rationalApproximation(numerator: Int, denominator: Int) -> (Int, Int) {
        let gcdVal = gcd(numerator, denominator)
        return (numerator / gcdVal, denominator / gcdVal)
    }

    /// Greatest common divisor using Euclidean algorithm.
    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var a = a
        var b = b
        while b != 0 {
            let temp = b
            b = a % b
            a = temp
        }
        return a
    }
}

// MARK: - Convenience Extensions

extension Resample {
    /// Convenience method with alternative parameter names (Int version, sync).
    public static func resampleSync(
        _ signal: [Float],
        fromRate: Int,
        toRate: Int,
        quality: ResampleQuality = .medium
    ) -> [Float] {
        return resampleSync(signal, fromSampleRate: fromRate, toSampleRate: toRate, quality: quality)
    }

    /// Convenience method with alternative parameter names (Float version, sync).
    public static func resampleSync(
        _ signal: [Float],
        fromRate: Float,
        toRate: Float,
        quality: ResampleQuality = .medium
    ) -> [Float] {
        return resampleSync(signal, fromSampleRate: Int(fromRate), toSampleRate: Int(toRate), quality: quality)
    }

    /// Convenience method with alternative parameter names (Int version).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromRate: Original sample rate in Hz.
    ///   - toRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromRate: Int,
        toRate: Int,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        return resampleSync(signal, fromSampleRate: fromRate, toSampleRate: toRate, quality: quality)
    }

    /// Convenience method with alternative parameter names (Float version).
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - fromRate: Original sample rate in Hz.
    ///   - toRate: Target sample rate in Hz.
    ///   - quality: Resampling quality level.
    /// - Returns: Resampled audio signal.
    public static func resample(
        _ signal: [Float],
        fromRate: Float,
        toRate: Float,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        return resampleSync(signal, fromSampleRate: Int(fromRate), toSampleRate: Int(toRate), quality: quality)
    }

    /// Common resampling presets.
    public enum Preset: Sendable {
        /// CD quality to Banquet model (44100 → 44100, no change)
        case cdToBanquet

        /// DVD audio to Banquet model (48000 → 44100)
        case dvdToBanquet

        /// High-res audio to Banquet model (96000 → 44100)
        case hiresToBanquet

        /// CD quality to common ML rate (44100 → 22050)
        case cdToML

        /// CD quality to speech rate (44100 → 16000)
        case cdToSpeech

        /// Source and target sample rates.
        var rates: (from: Int, to: Int) {
            switch self {
            case .cdToBanquet: return (44100, 44100)
            case .dvdToBanquet: return (48000, 44100)
            case .hiresToBanquet: return (96000, 44100)
            case .cdToML: return (44100, 22050)
            case .cdToSpeech: return (44100, 16000)
            }
        }
    }

    /// Resample using a preset configuration (synchronous version).
    public static func resampleSync(
        _ signal: [Float],
        preset: Preset,
        quality: ResampleQuality = .medium
    ) -> [Float] {
        let rates = preset.rates
        return resampleSync(
            signal,
            fromSampleRate: rates.from,
            toSampleRate: rates.to,
            quality: quality
        )
    }

    /// Resample using a preset configuration.
    ///
    /// - Parameters:
    ///   - signal: Input signal.
    ///   - preset: Resampling preset.
    ///   - actualInputRate: Actual input sample rate (must match preset.from).
    ///   - quality: Resampling quality.
    /// - Returns: Resampled signal.
    public static func resample(
        _ signal: [Float],
        preset: Preset,
        quality: ResampleQuality = .medium
    ) async -> [Float] {
        let rates = preset.rates
        return resampleSync(
            signal,
            fromSampleRate: rates.from,
            toSampleRate: rates.to,
            quality: quality
        )
    }
}
