import Accelerate
import Foundation

/// Normalization mode for mel filterbank.
public enum MelNorm: String, Sendable {
    /// Slaney-style normalization (area = 1).
    case slaney
}

/// Sparse representation of a mel filterbank for efficient multiplication.
///
/// Since mel filters are triangular and ~95% of entries are zero,
/// storing only the non-zero range per mel band dramatically reduces computation.
public struct SparseMelFilterbank: Sendable {
    /// Start index (inclusive) of non-zero region for each mel band.
    public let startIndices: [Int]

    /// End index (exclusive) of non-zero region for each mel band.
    public let endIndices: [Int]

    /// Packed non-zero filter values for all mel bands.
    /// Access with: values[offsets[m]..<offsets[m+1]] for mel band m.
    public let values: [Float]

    /// Offset into values array for each mel band. Length = nMels + 1.
    public let offsets: [Int]

    /// Number of mel bands.
    public let nMels: Int

    /// Number of frequency bins.
    public let nFreqs: Int
}

/// Mel filterbank generator.
public struct MelFilterbank: Sendable {
    /// Number of mel bands.
    public let nMels: Int

    /// FFT size.
    public let nFFT: Int

    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Minimum frequency in Hz.
    public let fMin: Float

    /// Maximum frequency in Hz.
    public let fMax: Float

    /// Use HTK formula (default: false = Slaney formula).
    public let htk: Bool

    /// Normalization mode.
    public let norm: MelNorm?

    /// Create mel filterbank configuration.
    ///
    /// - Parameters:
    ///   - nMels: Number of mel bands (default: 128).
    ///   - nFFT: FFT size (default: 2048).
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - fMin: Minimum frequency (default: 0).
    ///   - fMax: Maximum frequency (default: sampleRate/2).
    ///   - htk: Use HTK mel formula (default: false).
    ///   - norm: Normalization mode (default: .slaney).
    public init(
        nMels: Int = 128,
        nFFT: Int = 2048,
        sampleRate: Float = 22050,
        fMin: Float = 0,
        fMax: Float? = nil,
        htk: Bool = false,
        norm: MelNorm? = .slaney
    ) {
        self.nMels = nMels
        self.nFFT = nFFT
        self.sampleRate = sampleRate
        self.fMin = fMin
        self.fMax = fMax ?? (sampleRate / 2)
        self.htk = htk
        self.norm = norm
    }

    /// Number of frequency bins.
    public var nFreqs: Int { nFFT / 2 + 1 }

    /// Generate the mel filterbank matrix.
    ///
    /// - Returns: Filterbank matrix of shape (nMels, nFreqs).
    public func filters() -> [[Float]] {
        // Convert frequency limits to mel scale
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // Create mel points (nMels + 2 for the endpoints)
        let melPoints = linspace(melMin, melMax, count: nMels + 2)

        // Convert mel points back to Hz
        let hzPoints = melPoints.map { melToHz($0) }

        // Convert Hz to FFT bin indices (unused but kept for reference)
        _ = hzPoints.map { hz -> Int in
            Int(floor((Float(nFFT) + 1) * hz / sampleRate))
        }

        // FFT frequencies
        let fftFreqs = (0..<nFreqs).map { Float($0) * sampleRate / Float(nFFT) }

        // Build filterbank
        var filterbank = [[Float]]()
        filterbank.reserveCapacity(nMels)

        for m in 0..<nMels {
            var filter = [Float](repeating: 0, count: nFreqs)

            let fLeft = hzPoints[m]
            let fCenter = hzPoints[m + 1]
            let fRight = hzPoints[m + 2]

            for k in 0..<nFreqs {
                let freq = fftFreqs[k]

                if freq >= fLeft && freq < fCenter {
                    // Rising slope
                    filter[k] = (freq - fLeft) / (fCenter - fLeft)
                } else if freq >= fCenter && freq <= fRight {
                    // Falling slope
                    filter[k] = (fRight - freq) / (fRight - fCenter)
                }
            }

            // Apply normalization
            if norm == .slaney {
                let enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m])
                for k in 0..<nFreqs {
                    filter[k] *= enorm
                }
            }

            filterbank.append(filter)
        }

        return filterbank
    }

    // MARK: - Mel Scale Conversion

    private func hzToMel(_ hz: Float) -> Float {
        if htk {
            // HTK formula
            return 2595.0 * log10(1.0 + hz / 700.0)
        } else {
            // Slaney formula (linear below 1000 Hz, log above)
            let fSp: Float = 200.0 / 3.0
            let minLogHz: Float = 1000.0
            let minLogMel = minLogHz / fSp
            let logStep: Float = logf(6.4) / 27.0

            if hz < minLogHz {
                return hz / fSp
            } else {
                return minLogMel + logf(hz / minLogHz) / logStep
            }
        }
    }

    private func melToHz(_ mel: Float) -> Float {
        if htk {
            // HTK formula
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        } else {
            // Slaney formula
            let fSp: Float = 200.0 / 3.0
            let minLogHz: Float = 1000.0
            let minLogMel = minLogHz / fSp
            let logStep: Float = logf(6.4) / 27.0

            if mel < minLogMel {
                return mel * fSp
            } else {
                return minLogHz * expf(logStep * (mel - minLogMel))
            }
        }
    }

    private func linspace(_ start: Float, _ end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    /// Generate a sparse representation of the mel filterbank for efficient computation.
    ///
    /// Since mel filters are triangular with ~95% zero entries, sparse representation
    /// reduces multiply-add operations dramatically during spectrogram computation.
    ///
    /// - Returns: Sparse filterbank with only non-zero entries stored.
    public func sparseFilters() -> SparseMelFilterbank {
        // Convert frequency limits to mel scale
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // Create mel points (nMels + 2 for the endpoints)
        let melPoints = linspace(melMin, melMax, count: nMels + 2)

        // Convert mel points back to Hz
        let hzPoints = melPoints.map { melToHz($0) }

        // FFT frequencies
        let fftFreqs = (0..<nFreqs).map { Float($0) * sampleRate / Float(nFFT) }

        // Build sparse filterbank
        var startIndices = [Int]()
        var endIndices = [Int]()
        var values = [Float]()
        var offsets = [Int]()

        startIndices.reserveCapacity(nMels)
        endIndices.reserveCapacity(nMels)
        offsets.reserveCapacity(nMels + 1)

        for m in 0..<nMels {
            offsets.append(values.count)

            let fLeft = hzPoints[m]
            let fCenter = hzPoints[m + 1]
            let fRight = hzPoints[m + 2]

            // Find the range of non-zero bins
            var startIdx = -1
            var endIdx = -1

            for k in 0..<nFreqs {
                let freq = fftFreqs[k]
                if freq >= fLeft && freq <= fRight {
                    if startIdx == -1 { startIdx = k }
                    endIdx = k + 1
                }
            }

            // Handle case where no bins fall within the filter range
            if startIdx == -1 {
                startIdx = 0
                endIdx = 0
            }

            startIndices.append(startIdx)
            endIndices.append(endIdx)

            // Store only non-zero values
            if startIdx < endIdx {
                for k in startIdx..<endIdx {
                    let freq = fftFreqs[k]
                    var value: Float = 0

                    if freq >= fLeft && freq < fCenter {
                        value = (freq - fLeft) / (fCenter - fLeft)
                    } else if freq >= fCenter && freq <= fRight {
                        value = (fRight - freq) / (fRight - fCenter)
                    }

                    // Apply normalization
                    if norm == .slaney {
                        let enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m])
                        value *= enorm
                    }

                    values.append(value)
                }
            }
        }

        // Final offset for end-of-array
        offsets.append(values.count)

        return SparseMelFilterbank(
            startIndices: startIndices,
            endIndices: endIndices,
            values: values,
            offsets: offsets,
            nMels: nMels,
            nFreqs: nFreqs
        )
    }
}
