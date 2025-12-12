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
            padMode: .reflect,
            nMels: nMels,
            fMin: fMin,
            fMax: fMax,
            htk: false,
            norm: .slaney
        )

        self.nMFCC = min(nMFCC, nMels)
        self.dctType = dctType
        self.lifter = lifter

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
    public func fromPowerSpectrogram(_ powerSpec: [[Float]]) -> [[Float]] {
        let melSpec = melSpectrogram.fromPower(powerSpec)
        return fromMelSpectrogram(melSpec)
    }

    // MARK: - Private Implementation

    /// Apply log compression with floor value.
    private func applyLogCompression(_ melSpec: [[Float]]) -> [[Float]] {
        let amin: Float = 1e-10
        let rows = melSpec.count
        let cols = melSpec[0].count

        var result = [[Float]]()
        result.reserveCapacity(rows)

        for row in melSpec {
            var logRow = [Float](repeating: 0, count: cols)

            for j in 0..<cols {
                logRow[j] = log(max(row[j], amin))
            }

            result.append(logRow)
        }

        return result
    }

    /// Apply DCT-II to transform mel spectrum to cepstral coefficients.
    private func applyDCT(_ logMel: [[Float]]) -> [[Float]] {
        let nMels = logMel.count
        let nFrames = logMel[0].count

        // Pre-compute DCT-II matrix
        let dctMatrix = computeDCTMatrix(nMels: nMels, nMFCC: nMFCC)

        // Apply DCT to each frame: mfcc = dctMatrix * logMel
        var mfcc = [[Float]]()
        mfcc.reserveCapacity(nMFCC)

        for k in 0..<nMFCC {
            var mfccRow = [Float](repeating: 0, count: nFrames)

            for t in 0..<nFrames {
                var sum: Float = 0
                for m in 0..<nMels {
                    sum += dctMatrix[k][m] * logMel[m][t]
                }
                mfccRow[t] = sum
            }

            mfcc.append(mfccRow)
        }

        return mfcc
    }

    /// Compute DCT-II matrix for mel to cepstral conversion.
    private func computeDCTMatrix(nMels: Int, nMFCC: Int) -> [[Float]] {
        var matrix = [[Float]]()
        matrix.reserveCapacity(nMFCC)

        let scale = sqrt(2.0 / Float(nMels))

        for k in 0..<nMFCC {
            var row = [Float](repeating: 0, count: nMels)

            for m in 0..<nMels {
                // DCT-II formula: cos(pi * k * (m + 0.5) / N)
                let angle = Float.pi * Float(k) * (Float(m) + 0.5) / Float(nMels)
                row[m] = scale * cos(angle)
            }

            // First coefficient gets different scaling
            if k == 0 {
                var firstScale = Float(1.0 / sqrt(2.0))
                vDSP_vsmul(row, 1, &firstScale, &row, 1, vDSP_Length(nMels))
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
