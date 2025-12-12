import SwiftRosaCore
import Accelerate
import Foundation

/// Mel-frequency cepstral coefficients (MFCC) extractor.
///
/// MFCCs are a compact representation of the spectral envelope commonly used
/// in speech recognition, speaker identification, and music information retrieval.
///
/// The computation pipeline is:
/// 1. Compute mel spectrogram
/// 2. Apply log compression
/// 3. Apply Discrete Cosine Transform (DCT-II)
/// 4. Keep first `nMFCC` coefficients
/// 5. Optionally apply liftering
///
/// ## Example Usage
///
/// ```swift
/// let mfcc = MFCC(sampleRate: 22050, nMFCC: 13)
/// let coefficients = await mfcc.transform(audioSignal)
/// // coefficients has shape (nMFCC, nFrames)
///
/// // From pre-computed mel spectrogram
/// let melSpec = await melSpectrogram.transform(audio)
/// let coefficients = mfcc.fromMelSpectrogram(melSpec)
/// ```
public struct MFCC: Sendable {
    /// Mel spectrogram generator.
    public let melSpectrogram: MelSpectrogram

    /// Number of MFCC coefficients to return.
    public let nMFCC: Int

    /// DCT type (2 = standard MFCC).
    public let dctType: Int

    /// Liftering coefficient (0 = no liftering).
    public let lifter: Int

    /// Cached liftering weights.
    private let lifterWeights: [Float]?

    /// Cached DCT matrix in flat row-major format for vDSP_mmul.
    /// Shape: (nMFCC, nMels) stored as flat array of size nMFCC * nMels.
    private let cachedDCTMatrix: [Float]

    /// Number of mel bands for this instance.
    private let nMels: Int

    /// Create an MFCC extractor.
    ///
    /// - Parameters:
    ///   - sampleRate: Audio sample rate in Hz (default: 22050).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: 512).
    ///   - nMels: Number of mel bands (default: 128).
    ///   - nMFCC: Number of MFCC coefficients (default: 20).
    ///   - fMin: Minimum frequency (default: 0).
    ///   - fMax: Maximum frequency (default: sampleRate/2).
    ///   - windowType: Window function (default: hann).
    ///   - center: Center frames (default: true).
    ///   - dctType: DCT type, 2 = standard (default: 2).
    ///   - lifter: Liftering coefficient (default: 0 = disabled).
    public init(
        sampleRate: Float = 22050,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        nMels: Int = 128,
        nMFCC: Int = 20,
        fMin: Float = 0,
        fMax: Float? = nil,
        windowType: WindowType = .hann,
        center: Bool = true,
        dctType: Int = 2,
        lifter: Int = 0
    ) {
        self.melSpectrogram = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: nFFT,
            windowType: windowType,
            center: center,
            padMode: .constant(0),  // librosa default: zeros
            nMels: nMels,
            fMin: fMin,
            fMax: fMax,
            htk: false,
            norm: .slaney
        )

        self.nMFCC = min(nMFCC, nMels)
        self.dctType = dctType
        self.lifter = lifter
        self.nMels = nMels

        // Pre-compute and cache DCT matrix in flat format for vDSP_mmul
        self.cachedDCTMatrix = Self.computeFlatDCTMatrix(nMels: nMels, nMFCC: self.nMFCC)

        // Pre-compute liftering weights if needed
        if lifter > 0 {
            self.lifterWeights = Self.computeLifterWeights(nMFCC: self.nMFCC, lifter: lifter)
        } else {
            self.lifterWeights = nil
        }
    }

    // MARK: - Transform Methods

    /// Compute MFCCs from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: MFCC matrix of shape (nMFCC, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        let melSpec = await melSpectrogram.transform(signal)
        return fromMelSpectrogram(melSpec)
    }

    /// Compute MFCCs from pre-computed mel spectrogram.
    ///
    /// - Parameter melSpec: Mel spectrogram of shape (nMels, nFrames).
    /// - Returns: MFCC matrix of shape (nMFCC, nFrames).
    public func fromMelSpectrogram(_ melSpec: [[Float]]) -> [[Float]] {
        guard !melSpec.isEmpty && !melSpec[0].isEmpty else { return [] }

        // 1. Apply log compression
        let logMel = applyLogCompression(melSpec)

        // 2. Apply DCT to each frame
        let mfcc = applyDCT(logMel)

        // 3. Apply liftering if configured
        if let weights = lifterWeights {
            return applyLiftering(mfcc, weights: weights)
        }

        return mfcc
    }

    /// Compute MFCCs from power spectrogram.
    ///
    /// - Parameter powerSpec: Power spectrogram of shape (nFreqs, nFrames).
    /// - Returns: MFCC matrix of shape (nMFCC, nFrames).
    public func fromPowerSpectrogram(_ powerSpec: [[Float]]) async -> [[Float]] {
        let melSpec = await melSpectrogram.fromPower(powerSpec)
        return fromMelSpectrogram(melSpec)
    }

    // MARK: - Private Implementation

    /// Apply log compression using power_to_db (10 * log10) to match librosa.
    ///
    /// librosa uses `power_to_db` which computes `10 * log10(S/ref)` with:
    /// - ref=1.0 (default) for absolute dB
    /// - amin=1e-10 as minimum amplitude floor
    /// - top_db=80.0 for dynamic range limiting
    private func applyLogCompression(_ melSpec: [[Float]]) -> [[Float]] {
        let amin: Float = 1e-10
        let topDb: Float = 80.0  // librosa default
        let ref: Float = 1.0     // librosa default ref value
        let rows = melSpec.count
        guard rows > 0 else { return [] }
        let cols = melSpec[0].count

        var result = [[Float]]()
        result.reserveCapacity(rows)

        // Precompute log10 scale factor: log10(x) = ln(x) / ln(10)
        let log10Scale: Float = 1.0 / log(10.0)
        let logRef: Float = 10.0 * log10(ref)

        // First pass: compute all dB values and find max for top_db thresholding
        var maxDb: Float = -Float.greatestFiniteMagnitude

        for row in melSpec {
            var dbRow = [Float](repeating: 0, count: cols)

            // Clamp to minimum value
            var clampedRow = [Float](repeating: 0, count: cols)
            var aminVar = amin
            var maxVal: Float = .greatestFiniteMagnitude
            vDSP_vclip(row, 1, &aminVar, &maxVal, &clampedRow, 1, vDSP_Length(cols))

            // Apply vectorized natural log using vForce
            var logRow = [Float](repeating: 0, count: cols)
            var n = Int32(cols)
            vvlogf(&logRow, clampedRow, &n)

            // Convert to dB: 10 * log10(S/ref) = 10 * log10(S) - 10 * log10(ref)
            var scale: Float = 10.0 * log10Scale
            vDSP_vsmul(logRow, 1, &scale, &dbRow, 1, vDSP_Length(cols))

            // Subtract log(ref) (for ref=1.0 this is 0, but keep for correctness)
            if logRef != 0 {
                var negLogRef = -logRef
                vDSP_vsadd(dbRow, 1, &negLogRef, &dbRow, 1, vDSP_Length(cols))
            }

            // Track max for top_db thresholding
            var rowMax: Float = 0
            vDSP_maxv(dbRow, 1, &rowMax, vDSP_Length(cols))
            maxDb = max(maxDb, rowMax)

            result.append(dbRow)
        }

        // Second pass: apply top_db threshold relative to max
        let threshold = maxDb - topDb
        for i in 0..<rows {
            var thresholdVar = threshold
            vDSP_vthr(result[i], 1, &thresholdVar, &result[i], 1, vDSP_Length(cols))
        }

        return result
    }

    /// Apply DCT-II to transform mel spectrum to cepstral coefficients.
    ///
    /// Uses vDSP_mmul for optimized matrix multiplication:
    /// mfcc = dctMatrix (nMFCC x nMels) * logMel (nMels x nFrames) = result (nMFCC x nFrames)
    private func applyDCT(_ logMel: [[Float]]) -> [[Float]] {
        let inputMels = logMel.count
        let nFrames = logMel[0].count

        // Flatten logMel to row-major format (nMels x nFrames)
        var flatLogMel = [Float](repeating: 0, count: inputMels * nFrames)
        for m in 0..<inputMels {
            for t in 0..<nFrames {
                flatLogMel[m * nFrames + t] = logMel[m][t]
            }
        }

        // Allocate output (nMFCC x nFrames)
        var flatMFCC = [Float](repeating: 0, count: nMFCC * nFrames)

        // Matrix multiply using vDSP_mmul
        // C = A * B where:
        // A = cachedDCTMatrix (nMFCC x nMels), row-major
        // B = flatLogMel (nMels x nFrames), row-major
        // C = flatMFCC (nMFCC x nFrames), row-major
        vDSP_mmul(
            cachedDCTMatrix, 1,         // A matrix
            flatLogMel, 1,              // B matrix
            &flatMFCC, 1,               // C result
            vDSP_Length(nMFCC),         // M: rows of A and C
            vDSP_Length(nFrames),       // N: columns of B and C
            vDSP_Length(nMels)          // K: columns of A, rows of B
        )

        // Reshape to [[Float]]
        var mfcc = [[Float]]()
        mfcc.reserveCapacity(nMFCC)

        for k in 0..<nMFCC {
            let start = k * nFrames
            mfcc.append(Array(flatMFCC[start..<(start + nFrames)]))
        }

        return mfcc
    }

    /// Compute DCT-II matrix for mel to cepstral conversion matching scipy.fft.dct(type=2, norm='ortho').
    private func computeDCTMatrix(nMels: Int, nMFCC: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(nMFCC)
        let twoN = 2.0 * Float(nMels)

        for k in 0..<nMFCC {
            var row = [Float](repeating: 0, count: nMels)

            // scipy norm='ortho' scaling factors
            let scale: Float = (k == 0)
                ? sqrt(1.0 / (4.0 * Float(nMels)))
                : sqrt(1.0 / (2.0 * Float(nMels)))

            for m in 0..<nMels {
                // DCT-II: 2 * cos(pi * k * (2m + 1) / (2N))
                let angle = Float.pi * Float(k) * (2.0 * Float(m) + 1.0) / twoN
                row[m] = 2.0 * scale * cos(angle)
            }

            matrix.append(row)
        }

        return matrix
    }

    /// Apply sinusoidal liftering to emphasize higher-order coefficients.
    private func applyLiftering(_ mfcc: [[Float]], weights: [Float]) -> [[Float]] {
        var result = [[Float]]()
        result.reserveCapacity(mfcc.count)

        for (k, row) in mfcc.enumerated() {
            if k < weights.count {
                var liftedRow = [Float](repeating: 0, count: row.count)
                var weight = weights[k]
                vDSP_vsmul(row, 1, &weight, &liftedRow, 1, vDSP_Length(row.count))
                result.append(liftedRow)
            } else {
                result.append(row)
            }
        }

        return result
    }

    /// Compute liftering weights.
    private static func computeLifterWeights(nMFCC: Int, lifter: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: nMFCC)

        for n in 0..<nMFCC {
            // Liftering formula: 1 + (L/2) * sin(pi * n / L)
            weights[n] = 1.0 + Float(lifter) / 2.0 * sin(Float.pi * Float(n) / Float(lifter))
        }

        return weights
    }

    /// Compute flat DCT-II matrix for vDSP_mmul matching scipy.fft.dct(type=2, norm='ortho').
    ///
    /// scipy norm='ortho' formula:
    ///   y[k] = 2 * sum_{m=0}^{N-1} x[m] * cos(pi * k * (2m+1) / (2N)) * scale
    ///   where scale = sqrt(1/(4N)) for k=0, sqrt(1/(2N)) for k>0
    ///
    /// Returns matrix in row-major format (nMFCC x nMels) as flat array.
    private static func computeFlatDCTMatrix(nMels: Int, nMFCC: Int) -> [Float] {
        var matrix = [Float](repeating: 0, count: nMFCC * nMels)
        let twoN = 2.0 * Float(nMels)

        for k in 0..<nMFCC {
            // scipy norm='ortho' scaling factors
            let scale: Float = (k == 0)
                ? sqrt(1.0 / (4.0 * Float(nMels)))
                : sqrt(1.0 / (2.0 * Float(nMels)))

            for m in 0..<nMels {
                // DCT-II: 2 * cos(pi * k * (2m + 1) / (2N))
                let angle = Float.pi * Float(k) * (2.0 * Float(m) + 1.0) / twoN
                matrix[k * nMels + m] = 2.0 * scale * cos(angle)
            }
        }

        return matrix
    }
}

// MARK: - Delta Features

extension MFCC {
    /// Compute delta (first derivative) features.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width for delta computation (default: 9).
    /// - Returns: Delta features of same shape.
    public static func delta(_ features: [[Float]], width: Int = 9) -> [[Float]] {
        guard !features.isEmpty && !features[0].isEmpty else { return [] }

        let nFeatures = features.count
        let nFrames = features[0].count
        let halfWidth = width / 2

        var deltaFeatures = [[Float]]()
        deltaFeatures.reserveCapacity(nFeatures)

        // Compute normalization factor
        var norm: Float = 0
        for n in 1...halfWidth {
            norm += Float(n * n)
        }
        norm *= 2

        for f in 0..<nFeatures {
            var deltaRow = [Float](repeating: 0, count: nFrames)

            for t in 0..<nFrames {
                var sum: Float = 0

                for n in 1...halfWidth {
                    let leftIdx = max(0, t - n)
                    let rightIdx = min(nFrames - 1, t + n)
                    sum += Float(n) * (features[f][rightIdx] - features[f][leftIdx])
                }

                deltaRow[t] = sum / norm
            }

            deltaFeatures.append(deltaRow)
        }

        return deltaFeatures
    }

    /// Compute delta-delta (second derivative) features.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - width: Context width for delta computation.
    /// - Returns: Delta-delta features of same shape.
    public static func deltaDelta(_ features: [[Float]], width: Int = 9) -> [[Float]] {
        let d = delta(features, width: width)
        return delta(d, width: width)
    }

    /// Compute MFCCs with delta and delta-delta features.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - includeDelta: Include delta features.
    ///   - includeDeltaDelta: Include delta-delta features.
    /// - Returns: Stacked features of shape (nMFCC * (1 + delta + deltaDelta), nFrames).
    public func transformWithDeltas(
        _ signal: [Float],
        includeDelta: Bool = true,
        includeDeltaDelta: Bool = true
    ) async -> [[Float]] {
        let mfcc = await transform(signal)

        var result = mfcc

        if includeDelta {
            let d = Self.delta(mfcc)
            result.append(contentsOf: d)
        }

        if includeDeltaDelta {
            let dd = Self.deltaDelta(mfcc)
            result.append(contentsOf: dd)
        }

        return result
    }
}

// MARK: - Convenience Presets

extension MFCC {
    /// Standard 13-coefficient MFCC for speech recognition.
    public static var speechRecognition: MFCC {
        MFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,  // 10ms at 16kHz
            nMels: 40,
            nMFCC: 13,
            fMin: 20,
            fMax: 8000,
            lifter: 22
        )
    }

    /// MFCC configuration for music analysis.
    public static var musicAnalysis: MFCC {
        MFCC(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 20,
            fMin: 0,
            fMax: nil,
            lifter: 0
        )
    }

    /// MFCC configuration matching librosa defaults.
    public static var librosaDefault: MFCC {
        MFCC(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 20,
            fMin: 0,
            fMax: nil,
            windowType: .hann,
            center: true,
            dctType: 2,
            lifter: 0
        )
    }
}
