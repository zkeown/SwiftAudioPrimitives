import Foundation

/// Centralized tolerance definitions for librosa validation tests.
///
/// These tolerances are justified based on:
/// - Float32 precision (~7 significant digits)
/// - Error accumulation in different operation types
/// - Algorithm-specific characteristics
///
/// **Tolerance Tiers:**
/// 1. Exact (1e-6): Pure mathematical formulas with no error accumulation
/// 2. FFT (1e-5): Single FFT operation, minimal rounding
/// 3. Linear (1e-4): Matrix operations, error grows with dimensions
/// 4. Log-domain (1e-3): log/exp amplify small differences
/// 5. Iterative (5%): Random initialization, convergence variance
/// 6. Detection (frame-based): Approximate by nature
public struct ValidationTolerances {

    // MARK: - Tier 1: Exact Operations

    /// Window functions: pure mathematical formula
    public static let window: Double = 1e-6

    /// Mel scale conversion: pure mathematical formula
    public static let melScale: Double = 1e-6

    /// Hz to mel and mel to hz conversions
    public static let frequencyConversion: Double = 1e-6

    // MARK: - Tier 2: FFT Operations

    /// STFT magnitude: single FFT, Float32 precision
    public static let stftMagnitude: Double = 1e-5

    /// STFT phase: single FFT, Float32 precision
    public static let stftPhase: Double = 1e-5

    /// ISTFT reconstruction: forward + inverse accumulation
    public static let istftReconstruction: Double = 1e-4

    /// CQT magnitude: multi-resolution FFT
    public static let cqtMagnitude: Double = 1e-4

    /// VQT magnitude: variable-Q multi-resolution
    public static let vqtMagnitude: Double = 1e-4

    /// Pseudo-CQT magnitude
    public static let pseudoCqtMagnitude: Double = 1e-4

    // MARK: - Tier 3: Linear Operations

    /// Mel filterbank: matrix multiplication
    public static let melFilterbank: Double = 1e-4

    /// Mel spectrogram: FFT + matrix multiply
    public static let melSpectrogram: Double = 1e-4

    /// Spectral centroid: weighted mean
    public static let spectralCentroid: Double = 1e-4

    /// Spectral bandwidth: weighted std
    public static let spectralBandwidth: Double = 1e-4

    /// Spectral rolloff: threshold search
    public static let spectralRolloff: Double = 1e-4

    /// Spectral flatness: geometric/arithmetic mean ratio
    public static let spectralFlatness: Double = 1e-4

    /// Spectral flux: frame-to-frame difference
    public static let spectralFlux: Double = 1e-4

    /// Spectral contrast: band-wise peak/valley ratio
    public static let spectralContrast: Double = 1e-3

    /// Zero crossing rate: simple counting
    public static let zeroCrossingRate: Double = 1e-4

    /// RMS energy: simple power calculation
    public static let rmsEnergy: Double = 1e-4

    /// Delta features: FIR filtering
    public static let delta: Double = 1e-4

    /// Chromagram: pitch class projection
    public static let chromagram: Double = 1e-3

    /// Tonnetz: tonal centroid projection
    public static let tonnetz: Double = 1e-3

    /// Polynomial features
    public static let polyFeatures: Double = 1e-4

    // MARK: - Tier 4: Log-domain Operations

    /// MFCC: mel + log + DCT chain
    public static let mfcc: Double = 1e-3

    /// MFCC correlation coefficient (pattern similarity)
    public static let mfccCorrelation: Double = 0.95

    /// PCEN: adaptive gain with log compression
    public static let pcen: Double = 1e-3

    /// dB conversions: 10*log10 or 20*log10
    public static let dbConversion: Double = 1e-3

    /// Tempogram: rhythm periodicities
    public static let tempogram: Double = 1e-3

    // MARK: - Tier 5: Iterative/Stochastic Operations

    /// Griffin-Lim: iterative phase reconstruction
    public static let griffinLimSpectralError: Double = 0.05  // 5%

    /// NMF: iterative factorization
    public static let nmfReconstructionError: Double = 0.10  // 10%

    /// HPSS: median filtering decomposition
    public static let hpssEnergyRatio: Double = 0.10  // 10%

    // MARK: - Tier 6: Detection Operations

    /// Onset detection: frame tolerance
    public static let onsetFrameTolerance: Int = 3

    /// Beat tracking: frame tolerance
    public static let beatFrameTolerance: Int = 5

    /// Tempo estimation: percentage tolerance
    public static let tempoPercentTolerance: Double = 0.05  // 5%

    /// Pitch detection: cents tolerance
    public static let pitchCentsTolerance: Double = 50.0  // +/- 50 cents

    // MARK: - Resampling

    /// Resampling: output length tolerance (samples)
    public static let resampleLengthTolerance: Int = 5

    /// Resampling: energy preservation ratio
    public static let resampleEnergyRatio: Double = 0.10  // 10%

    // MARK: - Phase Vocoder

    /// Phase vocoder: output length tolerance ratio
    public static let phaseVocoderLengthRatio: Double = 0.01  // 1%

    /// Phase vocoder: energy preservation ratio
    public static let phaseVocoderEnergyRatio: Double = 0.15  // 15%
}

// MARK: - Validation Helpers

/// Helper functions for validation assertions
public enum ValidationHelpers {

    /// Compute maximum absolute error between two arrays
    public static func maxAbsoluteError(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// Compute maximum absolute error between two 2D arrays
    public static func maxAbsoluteError(_ a: [[Float]], _ b: [[Float]]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        var maxError: Float = 0
        for (rowA, rowB) in zip(a, b) {
            let rowError = maxAbsoluteError(rowA, rowB)
            maxError = max(maxError, rowError)
        }
        return maxError
    }

    /// Compute maximum relative error between two arrays (for non-zero values)
    public static func maxRelativeError(_ a: [Float], _ b: [Float], threshold: Float = 1e-6) -> Float {
        guard a.count == b.count else { return Float.infinity }
        var maxError: Float = 0
        for (va, vb) in zip(a, b) {
            if abs(vb) > threshold {
                let relError = abs(va - vb) / abs(vb)
                maxError = max(maxError, relError)
            }
        }
        return maxError
    }

    /// Compute Pearson correlation coefficient between two arrays
    public static func correlation(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, a.count > 1 else { return 0 }

        let n = Float(a.count)
        let meanA = a.reduce(0, +) / n
        let meanB = b.reduce(0, +) / n

        var varA: Float = 0
        var varB: Float = 0
        var cov: Float = 0

        for (va, vb) in zip(a, b) {
            let diffA = va - meanA
            let diffB = vb - meanB
            varA += diffA * diffA
            varB += diffB * diffB
            cov += diffA * diffB
        }

        let stdA = sqrt(varA / n)
        let stdB = sqrt(varB / n)

        guard stdA > 1e-10 && stdB > 1e-10 else { return 0 }

        return cov / (n * stdA * stdB)
    }

    /// Compute normalized RMSE between two arrays
    public static func normalizedRMSE(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }

        var sumSquaredError: Float = 0
        var sumSquaredRef: Float = 0

        for (va, vb) in zip(a, b) {
            sumSquaredError += (va - vb) * (va - vb)
            sumSquaredRef += vb * vb
        }

        guard sumSquaredRef > 1e-10 else { return 0 }

        return sqrt(sumSquaredError / sumSquaredRef)
    }

    /// Check if two frame indices match within tolerance
    public static func framesMatch(_ detected: [Int], _ expected: [Int], tolerance: Int) -> Bool {
        // For each expected frame, check if there's a detected frame within tolerance
        for expectedFrame in expected {
            let matchFound = detected.contains { abs($0 - expectedFrame) <= tolerance }
            if !matchFound {
                return false
            }
        }
        return true
    }

    /// Compute frame detection metrics (precision, recall, F1)
    public static func frameDetectionMetrics(
        detected: [Int],
        expected: [Int],
        tolerance: Int
    ) -> (precision: Float, recall: Float, f1: Float) {
        guard !expected.isEmpty else {
            return (detected.isEmpty ? 1.0 : 0.0, 1.0, detected.isEmpty ? 1.0 : 0.0)
        }
        guard !detected.isEmpty else {
            return (0.0, 0.0, 0.0)
        }

        var truePositives = 0
        var usedExpected = Set<Int>()

        for det in detected {
            for (idx, exp) in expected.enumerated() {
                if !usedExpected.contains(idx) && abs(det - exp) <= tolerance {
                    truePositives += 1
                    usedExpected.insert(idx)
                    break
                }
            }
        }

        let precision = Float(truePositives) / Float(detected.count)
        let recall = Float(truePositives) / Float(expected.count)
        let f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0

        return (precision, recall, f1)
    }
}

// MARK: - Test Signal Generation

/// Generate standard test signals for validation
public enum TestSignalGenerator {

    /// Generate a pure sine wave
    public static func sine(frequency: Float, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return (0..<nSamples).map { i in
            sin(2 * Float.pi * frequency * Float(i) / Float(sampleRate))
        }
    }

    /// Generate a multi-tone signal (harmonic series)
    public static func multiTone(fundamentalFrequency: Float = 440, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return (0..<nSamples).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2 * Float.pi * fundamentalFrequency * t) +
                   0.3 * sin(2 * Float.pi * fundamentalFrequency * 2 * t) +
                   0.2 * sin(2 * Float.pi * fundamentalFrequency * 4 * t)
        }
    }

    /// Generate white noise (seeded for reproducibility)
    public static func whiteNoise(sampleRate: Int = 22050, duration: Float = 1.0, amplitude: Float = 0.1, seed: UInt64 = 42) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        srand48(Int(seed))
        return (0..<nSamples).map { _ in
            Float(drand48() * 2 - 1) * amplitude
        }
    }

    /// Generate a linear chirp (frequency sweep)
    public static func chirp(fmin: Float = 100, fmax: Float = 8000, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        let rate = (fmax - fmin) / duration
        return (0..<nSamples).map { i in
            let t = Float(i) / Float(sampleRate)
            let instantFreq = fmin + rate * t / 2
            return sin(2 * Float.pi * instantFreq * t)
        }
    }

    /// Generate a click train (impulses at regular intervals)
    public static func clickTrain(clicksPerSecond: Int = 4, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        var signal = [Float](repeating: 0, count: nSamples)
        let clickInterval = sampleRate / clicksPerSecond
        for i in stride(from: 0, to: nSamples, by: clickInterval) {
            signal[i] = 1.0
        }
        return signal
    }

    /// Generate an AM-modulated signal
    public static func amModulated(carrierFreq: Float = 1000, modulatorFreq: Float = 5, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return (0..<nSamples).map { i in
            let t = Float(i) / Float(sampleRate)
            let carrier = sin(2 * Float.pi * carrierFreq * t)
            let modulator = 0.5 * (1 + sin(2 * Float.pi * modulatorFreq * t))
            return carrier * modulator
        }
    }

    /// Generate silence
    public static func silence(sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return [Float](repeating: 0, count: nSamples)
    }

    /// Generate DC offset
    public static func dcOffset(value: Float = 0.5, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return [Float](repeating: value, count: nSamples)
    }

    /// Generate single impulse at center
    public static func impulse(sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        var signal = [Float](repeating: 0, count: nSamples)
        signal[nSamples / 2] = 1.0
        return signal
    }
}
