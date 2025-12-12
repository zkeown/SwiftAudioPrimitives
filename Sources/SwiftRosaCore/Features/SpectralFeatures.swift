import Accelerate
import Foundation

// MARK: - Workspace for Buffer Reuse

/// Reusable workspace for spectral feature computation to avoid per-frame allocations.
///
/// This class holds pre-allocated buffers that are reused across frames,
/// significantly reducing memory allocations in hot paths.
public final class SpectralFeaturesWorkspace: @unchecked Sendable {
    /// Magnitude column buffer (nFreqs).
    var magCol: [Float]
    /// Secondary magnitude column buffer for flux computation.
    var prevMag: [Float]
    /// Deviations buffer for bandwidth computation (nFreqs).
    var deviations: [Float]
    /// Log magnitude buffer for flatness computation (nFreqs).
    var logMag: [Float]
    /// Difference buffer for flux computation (nFreqs).
    var diff: [Float]
    /// Band magnitudes buffer for contrast computation (max band size).
    var bandMags: [Float]

    /// Create a workspace for the given FFT size.
    ///
    /// - Parameter nFreqs: Number of frequency bins (typically nFFT/2 + 1).
    public init(nFreqs: Int) {
        self.magCol = [Float](repeating: 0, count: nFreqs)
        self.prevMag = [Float](repeating: 0, count: nFreqs)
        self.deviations = [Float](repeating: 0, count: nFreqs)
        self.logMag = [Float](repeating: 0, count: nFreqs)
        self.diff = [Float](repeating: 0, count: nFreqs)
        // Max band size is nFreqs (conservative)
        self.bandMags = [Float](repeating: 0, count: nFreqs)
    }
}

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

    /// Reusable workspace for buffer reuse across feature computations.
    private let workspace: SpectralFeaturesWorkspace

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
        self.workspace = SpectralFeaturesWorkspace(nFreqs: config.nFreqs)
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
                // Extract magnitude column for this frame (reusing workspace buffer)
                for f in 0..<nFreqs {
                    workspace.magCol[f] = magnitudeSpec[f][t]
                }

                // Compute weighted sum: dot product of frequencies and magnitudes
                var weightedSum: Float = 0
                vDSP_dotpr(freqPtr.baseAddress!, 1, workspace.magCol, 1, &weightedSum, vDSP_Length(nFreqs))

                // Compute total weight: sum of magnitudes
                var totalWeight: Float = 0
                vDSP_sve(workspace.magCol, 1, &totalWeight, vDSP_Length(nFreqs))

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

                // Extract magnitude column for this frame (reusing workspace buffer)
                for f in 0..<nFreqs {
                    workspace.magCol[f] = magnitudeSpec[f][t]
                }

                // Compute deviations: |freq - centroid| (reusing workspace buffer)
                var negCent = -cent
                vDSP_vsadd(freqPtr.baseAddress!, 1, &negCent, &workspace.deviations, 1, vDSP_Length(nFreqs))
                vDSP_vabs(workspace.deviations, 1, &workspace.deviations, 1, vDSP_Length(nFreqs))

                // Compute deviation^p
                if p == 2.0 {
                    // Special case: use squared (faster)
                    vDSP_vsq(workspace.deviations, 1, &workspace.deviations, 1, vDSP_Length(nFreqs))
                } else {
                    // General case: element-wise power
                    var pVar = p
                    var n = Int32(nFreqs)
                    vvpowsf(&workspace.deviations, &pVar, workspace.deviations, &n)
                }

                // Compute weighted sum: dot product of deviation^p and magnitudes
                var weightedSum: Float = 0
                vDSP_dotpr(workspace.deviations, 1, workspace.magCol, 1, &weightedSum, vDSP_Length(nFreqs))

                // Compute total weight: sum of magnitudes
                var totalWeight: Float = 0
                vDSP_sve(workspace.magCol, 1, &totalWeight, vDSP_Length(nFreqs))

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
            // Extract magnitude column for this frame (reusing workspace buffer)
            for f in 0..<nFreqs {
                workspace.magCol[f] = magnitudeSpec[f][t]
            }

            // Compute total energy using vDSP_sve
            var totalEnergy: Float = 0
            vDSP_sve(workspace.magCol, 1, &totalEnergy, vDSP_Length(nFreqs))

            let threshold = totalEnergy * rollPercent

            // Compute cumulative sum inline (no need to store full array)
            // Early exit when threshold is reached
            var runningSum: Float = 0
            for f in 0..<nFreqs {
                runningSum += workspace.magCol[f]
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
            // Extract magnitude column for this frame (reusing workspace buffer)
            for f in 0..<nFreqs {
                workspace.magCol[f] = max(magnitudeSpec[f][t], amin)
            }

            // Compute log of magnitudes using vForce (reusing workspace buffer)
            var n = Int32(nFreqs)
            vvlogf(&workspace.logMag, workspace.magCol, &n)

            // Compute sum of logs (for geometric mean)
            var logSum: Float = 0
            vDSP_sve(workspace.logMag, 1, &logSum, vDSP_Length(nFreqs))

            // Compute arithmetic sum
            var arithmeticSum: Float = 0
            vDSP_sve(workspace.magCol, 1, &arithmeticSum, vDSP_Length(nFreqs))

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

        // Extract first frame as previous
        for f in 0..<nFreqs {
            workspace.prevMag[f] = magnitudeSpec[f][0]
        }

        for t in 1..<nFrames {
            // Extract current magnitude column (reusing workspace buffer)
            for f in 0..<nFreqs {
                workspace.magCol[f] = magnitudeSpec[f][t]
            }

            // Compute difference: curr - prev
            vDSP_vsub(workspace.prevMag, 1, workspace.magCol, 1, &workspace.diff, 1, vDSP_Length(nFreqs))

            // Half-wave rectification: max(0, diff) using vDSP_vthres (threshold at 0)
            var zero: Float = 0
            vDSP_vthres(workspace.diff, 1, &zero, &workspace.diff, 1, vDSP_Length(nFreqs))

            // Sum the positive differences
            var fluxSum: Float = 0
            vDSP_sve(workspace.diff, 1, &fluxSum, vDSP_Length(nFreqs))

            result[t] = fluxSum

            // Copy current to previous for next iteration
            memcpy(&workspace.prevMag, workspace.magCol, nFreqs * MemoryLayout<Float>.size)
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
    ///   - nBands: Number of frequency subbands (default: 6, matching librosa).
    ///   - fMin: Minimum frequency (default: 200 Hz).
    ///   - quantile: Quantile for peak/valley detection (default: 0.02).
    /// - Returns: Spectral contrast per subband (nBands + 1, nFrames). Output has nBands + 1 rows
    ///           because librosa includes an extra band for the remaining high frequencies.
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
    /// Output shape is (nBands + 1, nFrames) to match librosa convention.
    public func contrast(
        magnitudeSpec: [[Float]],
        nBands: Int = 6,
        fMin: Float = 200,
        quantile: Float = 0.02
    ) -> [[Float]] {
        guard !magnitudeSpec.isEmpty && !magnitudeSpec[0].isEmpty else { return [] }

        let nFreqs = magnitudeSpec.count
        let nFrames = magnitudeSpec[0].count

        // Compute octave band edges to match librosa's approach
        // librosa creates n_bands + 2 octave edges: [0, fMin, 2*fMin, 4*fMin, ..., 2^nBands * fMin]
        // Then iterates through n_bands + 1 pairs
        let freqResolution = config.sampleRate / Float(config.nFFT)
        var bandEdges = [Int]()

        // First edge at bin 0 (DC)
        bandEdges.append(0)

        // Octave-spaced bands: fMin, 2*fMin, 4*fMin, ..., up to nBands + 1 total frequencies
        for i in 0...nBands {
            let freq = fMin * pow(2.0, Float(i))
            let bin = min(nFreqs - 1, Int(freq / freqResolution))
            bandEdges.append(bin)
        }

        // Total edges: 1 (DC) + (nBands + 1) frequencies = nBands + 2 edges
        // This gives nBands + 1 bands (pairs of edges)
        let nOutputBands = bandEdges.count - 1
        var result = [[Float]]()
        result.reserveCapacity(nOutputBands)

        let amin: Float = 1e-10

        for band in 0..<nOutputBands {
            let startBin = bandEdges[band]
            let endBin = bandEdges[band + 1]
            let bandSize = endBin - startBin

            guard bandSize > 0 else {
                result.append([Float](repeating: 0, count: nFrames))
                continue
            }

            var bandContrast = [Float](repeating: 0, count: nFrames)
            let nQuantile = max(1, Int(Float(bandSize) * quantile))

            for t in 0..<nFrames {
                // Extract band magnitudes into workspace buffer (reusing allocation)
                for i in 0..<bandSize {
                    workspace.bandMags[i] = magnitudeSpec[startBin + i][t]
                }

                // Use partial sort approach: find min/max quantiles without full sort
                // For small quantiles (typical 2%), this is much faster than full sort
                // Use workspace.bandMags directly (not a stale copy)
                let (valleyMean, peakMean) = computeQuantileMeans(
                    workspace.bandMags, count: bandSize, nQuantile: nQuantile
                )

                // Contrast in dB
                bandContrast[t] = log10(max(peakMean, amin)) - log10(max(valleyMean, amin))
            }

            result.append(bandContrast)
        }

        return result
    }

    /// Compute mean of bottom and top quantiles efficiently.
    /// Uses partial selection instead of full sort for better performance.
    private func computeQuantileMeans(
        _ data: [Float],
        count: Int,
        nQuantile: Int
    ) -> (valley: Float, peak: Float) {
        // For very small arrays or large quantiles, fall back to sort
        if count <= 16 || nQuantile > count / 4 {
            // Small array: sort is fast enough
            var sorted = Array(data[0..<count])
            sorted.sort()

            var valleySum: Float = 0
            for i in 0..<nQuantile {
                valleySum += sorted[i]
            }

            var peakSum: Float = 0
            for i in (count - nQuantile)..<count {
                peakSum += sorted[i]
            }

            return (valleySum / Float(nQuantile), peakSum / Float(nQuantile))
        }

        // For larger arrays: use partial sort / selection approach
        // Find nQuantile smallest and nQuantile largest values
        var copy = Array(data[0..<count])

        // Partial sort for bottom quantile (partition around nQuantile-th element)
        let bottomK = nQuantile
        partialSort(&copy, k: bottomK)

        var valleySum: Float = 0
        for i in 0..<bottomK {
            valleySum += copy[i]
        }

        // Partial sort for top quantile (partition around (count - nQuantile)-th element)
        let topStart = count - nQuantile
        partialSort(&copy, k: topStart)

        var peakSum: Float = 0
        for i in topStart..<count {
            peakSum += copy[i]
        }

        return (valleySum / Float(nQuantile), peakSum / Float(nQuantile))
    }

    /// Partially sort array so that elements [0..<k] are the k smallest.
    /// Uses QuickSelect algorithm (O(n) average case).
    private func partialSort(_ array: inout [Float], k: Int) {
        guard k > 0 && k < array.count else { return }

        var left = 0
        var right = array.count - 1

        while left < right {
            // Partition
            let pivot = array[(left + right) / 2]
            var i = left
            var j = right

            while i <= j {
                while array[i] < pivot { i += 1 }
                while array[j] > pivot { j -= 1 }
                if i <= j {
                    array.swapAt(i, j)
                    i += 1
                    j -= 1
                }
            }

            // Narrow search to the partition containing k
            if j < k { left = i }
            if k < i { right = j }
        }
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

    /// Extract all spectral features using contiguous STFT output.
    ///
    /// This is the most efficient path: uses contiguous memory throughout
    /// the entire pipeline, avoiding all [[Float]] intermediate allocations.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: All spectral features.
    public func extractAllContiguous(_ signal: [Float]) async -> SpectralFeatureResult {
        let mag = await stft.magnitudeContiguous(signal)

        return SpectralFeatureResult(
            centroid: centroidContiguous(magnitudeSpec: mag),
            bandwidth: bandwidthContiguous(magnitudeSpec: mag),
            rolloff: rolloffContiguous(magnitudeSpec: mag),
            flatness: flatnessContiguous(magnitudeSpec: mag),
            flux: fluxContiguous(magnitudeSpec: mag),
            contrast: contrastContiguous(magnitudeSpec: mag),
            nFrames: mag.cols
        )
    }

    // MARK: - Contiguous Input Methods

    /// Compute spectral centroid from contiguous magnitude spectrogram.
    ///
    /// This method operates directly on row-major contiguous data,
    /// providing optimal cache performance for column (frame) access.
    ///
    /// - Parameter magnitudeSpec: Contiguous magnitude spectrogram (nFreqs × nFrames).
    /// - Returns: Spectral centroid per frame (Hz).
    public func centroidContiguous(magnitudeSpec: ContiguousMatrix) -> [Float] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols

        var result = [Float](repeating: 0, count: nFrames)

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            frequencies.withUnsafeBufferPointer { freqPtr in
                for t in 0..<nFrames {
                    // Extract column into workspace buffer to match regular method's computation
                    for f in 0..<nFreqs {
                        workspace.magCol[f] = magPtr[f * nFrames + t]
                    }

                    // Compute weighted sum: dot product of frequencies and magnitudes
                    var weightedSum: Float = 0
                    vDSP_dotpr(freqPtr.baseAddress!, 1, workspace.magCol, 1, &weightedSum, vDSP_Length(nFreqs))

                    // Compute total weight: sum of magnitudes
                    var totalWeight: Float = 0
                    vDSP_sve(workspace.magCol, 1, &totalWeight, vDSP_Length(nFreqs))

                    result[t] = totalWeight > 0 ? weightedSum / totalWeight : 0
                }
            }
        }

        return result
    }

    /// Compute spectral bandwidth from contiguous magnitude spectrogram.
    public func bandwidthContiguous(magnitudeSpec: ContiguousMatrix, p: Float = 2.0) -> [Float] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let centroids = centroidContiguous(magnitudeSpec: magnitudeSpec)
        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols
        let invP = 1.0 / p

        var result = [Float](repeating: 0, count: nFrames)

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            frequencies.withUnsafeBufferPointer { freqPtr in
                for t in 0..<nFrames {
                    let cent = centroids[t]

                    // Extract column into workspace (required for subsequent operations)
                    for f in 0..<nFreqs {
                        workspace.magCol[f] = magPtr[f * nFrames + t]
                    }

                    // Compute deviations: |freq - centroid|
                    var negCent = -cent
                    vDSP_vsadd(freqPtr.baseAddress!, 1, &negCent, &workspace.deviations, 1, vDSP_Length(nFreqs))
                    vDSP_vabs(workspace.deviations, 1, &workspace.deviations, 1, vDSP_Length(nFreqs))

                    // Compute deviation^p
                    if p == 2.0 {
                        vDSP_vsq(workspace.deviations, 1, &workspace.deviations, 1, vDSP_Length(nFreqs))
                    } else {
                        var pVar = p
                        var n = Int32(nFreqs)
                        vvpowsf(&workspace.deviations, &pVar, workspace.deviations, &n)
                    }

                    // Weighted sum and total weight - use workspace arrays directly (not copies)
                    var weightedSum: Float = 0
                    var totalWeight: Float = 0
                    vDSP_dotpr(workspace.deviations, 1, workspace.magCol, 1, &weightedSum, vDSP_Length(nFreqs))
                    vDSP_sve(workspace.magCol, 1, &totalWeight, vDSP_Length(nFreqs))

                    if totalWeight > 0 {
                        result[t] = pow(weightedSum / totalWeight, invP)
                    }
                }
            }
        }

        return result
    }

    /// Compute spectral rolloff from contiguous magnitude spectrogram.
    public func rolloffContiguous(magnitudeSpec: ContiguousMatrix, rollPercent: Float = 0.85) -> [Float] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols

        var result = [Float](repeating: 0, count: nFrames)

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            for t in 0..<nFrames {
                // Compute total energy using strided sum
                var totalEnergy: Float = 0
                vDSP_sve(
                    magPtr.baseAddress! + t, vDSP_Stride(nFrames),
                    &totalEnergy,
                    vDSP_Length(nFreqs)
                )

                let threshold = totalEnergy * rollPercent

                // Cumulative sum with early exit
                var runningSum: Float = 0
                for f in 0..<nFreqs {
                    runningSum += magPtr[f * nFrames + t]
                    if runningSum >= threshold {
                        result[t] = frequencies[f]
                        break
                    }
                }
            }
        }

        return result
    }

    /// Compute spectral flatness from contiguous magnitude spectrogram.
    public func flatnessContiguous(magnitudeSpec: ContiguousMatrix) -> [Float] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols
        let amin: Float = 1e-10
        let invNFreqs = 1.0 / Float(nFreqs)

        var result = [Float](repeating: 0, count: nFrames)

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            for t in 0..<nFrames {
                // Extract column with floor clamping
                for f in 0..<nFreqs {
                    workspace.magCol[f] = max(magPtr[f * nFrames + t], amin)
                }

                // Compute log of magnitudes
                var n = Int32(nFreqs)
                vvlogf(&workspace.logMag, workspace.magCol, &n)

                // Sum of logs (for geometric mean)
                var logSum: Float = 0
                vDSP_sve(workspace.logMag, 1, &logSum, vDSP_Length(nFreqs))

                // Arithmetic sum
                var arithmeticSum: Float = 0
                vDSP_sve(workspace.magCol, 1, &arithmeticSum, vDSP_Length(nFreqs))

                let geometricMean = exp(logSum * invNFreqs)
                let arithmeticMean = arithmeticSum * invNFreqs

                result[t] = arithmeticMean > amin ? geometricMean / arithmeticMean : 0
            }
        }

        return result
    }

    /// Compute spectral flux from contiguous magnitude spectrogram.
    public func fluxContiguous(magnitudeSpec: ContiguousMatrix) -> [Float] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols

        var result = [Float](repeating: 0, count: nFrames)

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            // Extract first frame as previous
            for f in 0..<nFreqs {
                workspace.prevMag[f] = magPtr[f * nFrames]
            }

            for t in 1..<nFrames {
                // Extract current column
                for f in 0..<nFreqs {
                    workspace.magCol[f] = magPtr[f * nFrames + t]
                }

                // Compute difference: curr - prev
                vDSP_vsub(workspace.prevMag, 1, workspace.magCol, 1, &workspace.diff, 1, vDSP_Length(nFreqs))

                // Half-wave rectification
                var zero: Float = 0
                vDSP_vthres(workspace.diff, 1, &zero, &workspace.diff, 1, vDSP_Length(nFreqs))

                // Sum positive differences
                var fluxSum: Float = 0
                vDSP_sve(workspace.diff, 1, &fluxSum, vDSP_Length(nFreqs))

                result[t] = fluxSum

                // Copy current to previous
                memcpy(&workspace.prevMag, workspace.magCol, nFreqs * MemoryLayout<Float>.size)
            }
        }

        return result
    }

    /// Compute spectral contrast from contiguous magnitude spectrogram.
    /// Output shape is (nBands + 1, nFrames) to match librosa convention.
    public func contrastContiguous(
        magnitudeSpec: ContiguousMatrix,
        nBands: Int = 6,
        fMin: Float = 200,
        quantile: Float = 0.02
    ) -> [[Float]] {
        guard magnitudeSpec.rows > 0 && magnitudeSpec.cols > 0 else { return [] }

        let nFreqs = magnitudeSpec.rows
        let nFrames = magnitudeSpec.cols

        // Compute octave band edges to match librosa's approach
        let freqResolution = config.sampleRate / Float(config.nFFT)
        var bandEdges = [Int]()

        // First edge at bin 0 (DC)
        bandEdges.append(0)

        // Octave-spaced bands: fMin, 2*fMin, 4*fMin, ..., up to nBands + 1 total frequencies
        for i in 0...nBands {
            let freq = fMin * pow(2.0, Float(i))
            let bin = min(nFreqs - 1, Int(freq / freqResolution))
            bandEdges.append(bin)
        }

        // Total edges: 1 (DC) + (nBands + 1) frequencies = nBands + 2 edges
        // This gives nBands + 1 bands (pairs of edges)
        let nOutputBands = bandEdges.count - 1
        var result = [[Float]]()
        result.reserveCapacity(nOutputBands)

        let amin: Float = 1e-10

        magnitudeSpec.data.withUnsafeBufferPointer { magPtr in
            for band in 0..<nOutputBands {
                let startBin = bandEdges[band]
                let endBin = bandEdges[band + 1]
                let bandSize = endBin - startBin

                guard bandSize > 0 else {
                    result.append([Float](repeating: 0, count: nFrames))
                    continue
                }

                var bandContrast = [Float](repeating: 0, count: nFrames)
                let nQuantile = max(1, Int(Float(bandSize) * quantile))

                for t in 0..<nFrames {
                    // Extract band magnitudes from contiguous data
                    for i in 0..<bandSize {
                        workspace.bandMags[i] = magPtr[(startBin + i) * nFrames + t]
                    }

                    // Use workspace.bandMags directly (not a copy)
                    let (valleyMean, peakMean) = computeQuantileMeans(
                        workspace.bandMags, count: bandSize, nQuantile: nQuantile
                    )

                    bandContrast[t] = log10(max(peakMean, amin)) - log10(max(valleyMean, amin))
                }

                result.append(bandContrast)
            }
        }

        return result
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
