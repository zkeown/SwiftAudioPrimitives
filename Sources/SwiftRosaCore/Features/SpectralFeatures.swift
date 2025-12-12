import Accelerate
import Foundation

/// Configuration for spectral feature extraction.
public struct SpectralFeaturesConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// FFT size.
    public let nFFT: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Window type.
    public let windowType: WindowType

    /// Whether to center frames.
    public let center: Bool

    /// Create spectral features configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop length (default: 512).
    ///   - windowType: Window type (default: hann).
    ///   - center: Center frames (default: true).
    public init(
        sampleRate: Float = 22050,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        windowType: WindowType = .hann,
        center: Bool = true
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.windowType = windowType
        self.center = center
    }

    /// Number of frequency bins.
    public var nFreqs: Int { nFFT / 2 + 1 }

    /// Frequency array for spectral bins.
    public var frequencies: [Float] {
        var freqs = [Float](repeating: 0, count: nFreqs)
        let binWidth = sampleRate / Float(nFFT)
        for i in 0..<nFreqs {
            freqs[i] = Float(i) * binWidth
        }
        return freqs
    }
}

/// Result containing all spectral features.
public struct SpectralFeatureResult: Sendable {
    /// Spectral centroid (Hz).
    public let centroid: [Float]

    /// Spectral bandwidth.
    public let bandwidth: [Float]

    /// Spectral rolloff frequency (Hz).
    public let rolloff: [Float]

    /// Spectral flatness (0-1).
    public let flatness: [Float]

    /// Spectral flux.
    public let flux: [Float]

    /// Spectral contrast per subband.
    public let contrast: [[Float]]

    /// Number of frames.
    public let nFrames: Int
}

/// Spectral feature extractor.
///
/// Extracts various spectral features from audio signals, commonly used
/// for audio classification, music information retrieval, and speech analysis.
///
/// ## Example Usage
///
/// ```swift
/// let features = SpectralFeatures(config: SpectralFeaturesConfig(sampleRate: 22050))
///
/// // Extract individual features
/// let centroid = await features.centroid(audioSignal)
///
/// // Extract all features at once (more efficient)
/// let result = await features.extractAll(audioSignal)
/// ```
public struct SpectralFeatures: Sendable {
    /// Configuration.
    public let config: SpectralFeaturesConfig

    private let stft: STFT

    /// Cached frequency array.
    private let frequencies: [Float]

    /// Create spectral feature extractor.
    ///
    /// - Parameter config: Configuration (default: librosa-compatible).
    public init(config: SpectralFeaturesConfig = SpectralFeaturesConfig()) {
        self.config = config
        self.stft = STFT(config: STFTConfig(
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.nFFT,
            windowType: config.windowType,
            center: config.center,
            padMode: .reflect
        ))
        self.frequencies = config.frequencies
    }

    // MARK: - Individual Features

    /// Compute spectral centroid.
    ///
    /// The spectral centroid is the weighted mean of frequencies,
    /// indicating the "center of mass" of the spectrum.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Spectral centroid per frame (Hz).
    public func centroid(_ signal: [Float]) async -> [Float] {
        let mag = await stft.magnitude(signal)
        return centroid(magnitudeSpec: mag)
    }

    /// Compute spectral centroid from pre-computed magnitude spectrogram.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram (nFreqs, nFrames).
    /// - Returns: Spectral centroid per frame (Hz).
    public func centroid(magnitudeSpec: [[Float]]) -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        var result = [Float](repeating: 0, count: nFrames)

        // Process each frame using vectorized operations
        frequencies.withUnsafeBufferPointer { freqPtr in
            for t in 0..<nFrames {
                // Extract magnitude column for this frame
                var magCol = [Float](repeating: 0, count: nFreqs)
                for f in 0..<nFreqs {
                    magCol[f] = magnitudeSpec[f][t]
                }

                // Compute weighted sum: dot product of frequencies and magnitudes
                var weightedSum: Float = 0
                vDSP_dotpr(freqPtr.baseAddress!, 1, magCol, 1, &weightedSum, vDSP_Length(nFreqs))

                // Compute total weight: sum of magnitudes
                var totalWeight: Float = 0
                vDSP_sve(magCol, 1, &totalWeight, vDSP_Length(nFreqs))

                result[t] = totalWeight > 0 ? weightedSum / totalWeight : 0
            }
        }

        return result
    }

    /// Compute spectral bandwidth.
    ///
    /// The spectral bandwidth is the weighted standard deviation of frequencies
    /// around the centroid.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - p: Order of the bandwidth (default: 2 for standard deviation).
    /// - Returns: Spectral bandwidth per frame.
    public func bandwidth(_ signal: [Float], p: Float = 2.0) async -> [Float] {
        let mag = await stft.magnitude(signal)
        return bandwidth(magnitudeSpec: mag, p: p)
    }

    /// Compute spectral bandwidth from pre-computed magnitude spectrogram.
    public func bandwidth(magnitudeSpec: [[Float]], p: Float = 2.0) -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let centroids = centroid(magnitudeSpec: magnitudeSpec)
        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        var result = [Float](repeating: 0, count: nFrames)
        let invP = 1.0 / p

        frequencies.withUnsafeBufferPointer { freqPtr in
            for t in 0..<nFrames {
                let cent = centroids[t]

                // Extract magnitude column for this frame
                var magCol = [Float](repeating: 0, count: nFreqs)
                for f in 0..<nFreqs {
                    magCol[f] = magnitudeSpec[f][t]
                }

                // Compute deviations: |freq - centroid|
                var deviations = [Float](repeating: 0, count: nFreqs)
                var negCent = -cent
                vDSP_vsadd(freqPtr.baseAddress!, 1, &negCent, &deviations, 1, vDSP_Length(nFreqs))
                vDSP_vabs(deviations, 1, &deviations, 1, vDSP_Length(nFreqs))

                // Compute deviation^p
                if p == 2.0 {
                    // Special case: use squared (faster)
                    vDSP_vsq(deviations, 1, &deviations, 1, vDSP_Length(nFreqs))
                } else {
                    // General case: element-wise power
                    var pVar = p
                    var n = Int32(nFreqs)
                    vvpowsf(&deviations, &pVar, deviations, &n)
                }

                // Compute weighted sum: dot product of deviation^p and magnitudes
                var weightedSum: Float = 0
                vDSP_dotpr(deviations, 1, magCol, 1, &weightedSum, vDSP_Length(nFreqs))

                // Compute total weight: sum of magnitudes
                var totalWeight: Float = 0
                vDSP_sve(magCol, 1, &totalWeight, vDSP_Length(nFreqs))

                if totalWeight > 0 {
                    result[t] = pow(weightedSum / totalWeight, invP)
                }
            }
        }

        return result
    }

    /// Compute spectral rolloff.
    ///
    /// The rolloff frequency is the frequency below which a specified
    /// percentage of total spectral energy is contained.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - rollPercent: Energy percentage threshold (default: 0.85).
    /// - Returns: Rolloff frequency per frame (Hz).
    public func rolloff(_ signal: [Float], rollPercent: Float = 0.85) async -> [Float] {
        let mag = await stft.magnitude(signal)
        return rolloff(magnitudeSpec: mag, rollPercent: rollPercent)
    }

    /// Compute spectral rolloff from pre-computed magnitude spectrogram.
    public func rolloff(magnitudeSpec: [[Float]], rollPercent: Float = 0.85) -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        var result = [Float](repeating: 0, count: nFrames)

        for t in 0..<nFrames {
            // Extract magnitude column for this frame
            var magCol = [Float](repeating: 0, count: nFreqs)
            for f in 0..<nFreqs {
                magCol[f] = magnitudeSpec[f][t]
            }

            // Compute total energy using vDSP_sve
            var totalEnergy: Float = 0
            vDSP_sve(magCol, 1, &totalEnergy, vDSP_Length(nFreqs))

            let threshold = totalEnergy * rollPercent

            // Compute cumulative sum
            var cumSum = [Float](repeating: 0, count: nFreqs)
            // vDSP doesn't have prefix sum, but we can use a running sum
            // For small nFreqs this is fine; for large arrays we could parallelize
            var runningSum: Float = 0
            for f in 0..<nFreqs {
                runningSum += magCol[f]
                cumSum[f] = runningSum
                if runningSum >= threshold {
                    result[t] = frequencies[f]
                    break
                }
            }
        }

        return result
    }

    /// Compute spectral flatness.
    ///
    /// Spectral flatness measures how noise-like vs tonal a signal is.
    /// Values close to 1 indicate noise, values close to 0 indicate tonal content.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Spectral flatness per frame (0-1).
    public func flatness(_ signal: [Float]) async -> [Float] {
        let mag = await stft.magnitude(signal)
        return flatness(magnitudeSpec: mag)
    }

    /// Compute spectral flatness from pre-computed magnitude spectrogram.
    public func flatness(magnitudeSpec: [[Float]]) -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count
        let amin: Float = 1e-10

        var result = [Float](repeating: 0, count: nFrames)
        let invNFreqs = 1.0 / Float(nFreqs)

        for t in 0..<nFrames {
            // Extract magnitude column for this frame
            var magCol = [Float](repeating: 0, count: nFreqs)
            for f in 0..<nFreqs {
                magCol[f] = max(magnitudeSpec[f][t], amin)
            }

            // Compute log of magnitudes using vForce
            var logMag = [Float](repeating: 0, count: nFreqs)
            var n = Int32(nFreqs)
            vvlogf(&logMag, magCol, &n)

            // Compute sum of logs (for geometric mean)
            var logSum: Float = 0
            vDSP_sve(logMag, 1, &logSum, vDSP_Length(nFreqs))

            // Compute arithmetic sum
            var arithmeticSum: Float = 0
            vDSP_sve(magCol, 1, &arithmeticSum, vDSP_Length(nFreqs))

            // Geometric mean = exp(mean(log(x)))
            let geometricMean = exp(logSum * invNFreqs)
            let arithmeticMean = arithmeticSum * invNFreqs

            result[t] = arithmeticMean > amin ? geometricMean / arithmeticMean : 0
        }

        return result
    }

    /// Compute spectral flux.
    ///
    /// Spectral flux measures how quickly the spectrum changes between frames,
    /// useful for onset detection.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Spectral flux per frame.
    public func flux(_ signal: [Float]) async -> [Float] {
        let mag = await stft.magnitude(signal)
        return flux(magnitudeSpec: mag)
    }

    /// Compute spectral flux from pre-computed magnitude spectrogram.
    public func flux(magnitudeSpec: [[Float]]) -> [Float] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        var result = [Float](repeating: 0, count: nFrames)

        // Pre-allocate buffers for vectorized operations
        var currMag = [Float](repeating: 0, count: nFreqs)
        var prevMag = [Float](repeating: 0, count: nFreqs)
        var diff = [Float](repeating: 0, count: nFreqs)

        // Extract first frame as previous
        for f in 0..<nFreqs {
            prevMag[f] = magnitudeSpec[f][0]
        }

        for t in 1..<nFrames {
            // Extract current magnitude column
            for f in 0..<nFreqs {
                currMag[f] = magnitudeSpec[f][t]
            }

            // Compute difference: curr - prev
            vDSP_vsub(prevMag, 1, currMag, 1, &diff, 1, vDSP_Length(nFreqs))

            // Half-wave rectification: max(0, diff) using vDSP_vthres (threshold at 0)
            var zero: Float = 0
            vDSP_vthres(diff, 1, &zero, &diff, 1, vDSP_Length(nFreqs))

            // Sum the positive differences
            var fluxSum: Float = 0
            vDSP_sve(diff, 1, &fluxSum, vDSP_Length(nFreqs))

            result[t] = fluxSum

            // Swap buffers
            swap(&currMag, &prevMag)
        }

        return result
    }

    /// Compute spectral contrast.
    ///
    /// Spectral contrast measures the difference between peaks and valleys
    /// in each frequency subband.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - nBands: Number of frequency subbands (default: 6).
    ///   - fMin: Minimum frequency (default: 200 Hz).
    ///   - quantile: Quantile for peak/valley detection (default: 0.02).
    /// - Returns: Spectral contrast per subband (nBands, nFrames).
    public func contrast(
        _ signal: [Float],
        nBands: Int = 6,
        fMin: Float = 200,
        quantile: Float = 0.02
    ) async -> [[Float]] {
        let mag = await stft.magnitude(signal)
        return contrast(magnitudeSpec: mag, nBands: nBands, fMin: fMin, quantile: quantile)
    }

    /// Compute spectral contrast from pre-computed magnitude spectrogram.
    public func contrast(
        magnitudeSpec: [[Float]],
        nBands: Int = 6,
        fMin: Float = 200,
        quantile: Float = 0.02
    ) -> [[Float]] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        // Compute octave band edges
        let fMax = config.sampleRate / 2
        var bandEdges = [Int]()

        // Find frequency bin for fMin
        let fMinBin = Int(fMin / (config.sampleRate / Float(config.nFFT)))
        bandEdges.append(max(0, fMinBin))

        // Octave-spaced bands
        var freq = fMin
        for _ in 0..<nBands {
            freq *= 2  // Octave spacing
            if freq >= fMax { break }
            let bin = Int(freq / (config.sampleRate / Float(config.nFFT)))
            bandEdges.append(min(nFreqs - 1, bin))
        }
        bandEdges.append(nFreqs - 1)

        // Compute contrast for each band
        var result = [[Float]]()
        result.reserveCapacity(nBands)

        for band in 0..<min(nBands, bandEdges.count - 1) {
            let startBin = bandEdges[band]
            let endBin = bandEdges[band + 1]

            var bandContrast = [Float](repeating: 0, count: nFrames)

            for t in 0..<nFrames {
                // Extract band magnitudes
                var bandMags = [Float]()
                for f in startBin..<endBin {
                    bandMags.append(magnitudeSpec[f][t])
                }

                guard !bandMags.isEmpty else { continue }

                // Sort to find peaks and valleys
                bandMags.sort()

                let nSamples = bandMags.count
                let nQuantile = max(1, Int(Float(nSamples) * quantile))

                // Valley: mean of bottom quantile
                var valleySum: Float = 0
                for i in 0..<nQuantile {
                    valleySum += bandMags[i]
                }
                let valley = valleySum / Float(nQuantile)

                // Peak: mean of top quantile
                var peakSum: Float = 0
                for i in (nSamples - nQuantile)..<nSamples {
                    peakSum += bandMags[i]
                }
                let peak = peakSum / Float(nQuantile)

                // Contrast in dB
                let amin: Float = 1e-10
                bandContrast[t] = log10(max(peak, amin)) - log10(max(valley, amin))
            }

            result.append(bandContrast)
        }

        return result
    }

    // MARK: - Batch Extraction

    /// Extract all spectral features at once.
    ///
    /// More efficient than calling individual methods when multiple
    /// features are needed.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: All spectral features.
    public func extractAll(_ signal: [Float]) async -> SpectralFeatureResult {
        let mag = await stft.magnitude(signal)

        return SpectralFeatureResult(
            centroid: centroid(magnitudeSpec: mag),
            bandwidth: bandwidth(magnitudeSpec: mag),
            rolloff: rolloff(magnitudeSpec: mag),
            flatness: flatness(magnitudeSpec: mag),
            flux: flux(magnitudeSpec: mag),
            contrast: contrast(magnitudeSpec: mag),
            nFrames: mag.isEmpty ? 0 : mag[0].count
        )
    }
}

// MARK: - Presets

extension SpectralFeatures {
    /// Configuration for speech analysis.
    public static var speechAnalysis: SpectralFeatures {
        SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            windowType: .hann,
            center: true
        ))
    }

    /// Configuration for music analysis.
    public static var musicAnalysis: SpectralFeatures {
        SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true
        ))
    }

    /// Configuration matching librosa defaults.
    public static var librosaDefault: SpectralFeatures {
        SpectralFeatures(config: SpectralFeaturesConfig(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            center: true
        ))
    }
}
