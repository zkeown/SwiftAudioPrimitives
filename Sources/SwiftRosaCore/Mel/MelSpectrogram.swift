import Accelerate
import Foundation

/// Mel spectrogram generator with automatic GPU/CPU dispatch.
///
/// Uses Metal GPU acceleration for large spectrograms (>50K elements) and
/// optimized vDSP matrix multiplication for CPU fallback.
public struct MelSpectrogram: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// STFT processor.
    public let stft: STFT

    /// Mel filterbank.
    public let melFilterbank: MelFilterbank

    /// Cached filterbank matrix (2D for API compatibility).
    private let filters: [[Float]]

    /// Pre-flattened filterbank for vDSP_mmul: (nMels × nFreqs) row-major.
    private let flatFilters: [Float]

    /// Sparse filterbank for efficient CPU computation.
    private let sparseFilters: SparseMelFilterbank

    /// Number of frequency bins.
    private let nFreqs: Int

    /// Optional Metal engine for GPU acceleration.
    private let metalEngine: MetalEngine?

    /// Whether to prefer GPU acceleration when available.
    public let useGPU: Bool

    /// Create mel spectrogram generator.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: 512).
    ///   - winLength: Window length (default: nFFT).
    ///   - windowType: Window function (default: hann).
    ///   - center: Pad signal for centered frames (default: true).
    ///   - padMode: Padding mode (default: constant/zero, matching librosa).
    ///   - nMels: Number of mel bands (default: 128).
    ///   - fMin: Minimum frequency (default: 0).
    ///   - fMax: Maximum frequency (default: sampleRate/2).
    ///   - htk: Use HTK mel formula (default: false).
    ///   - norm: Mel normalization (default: .slaney).
    ///   - useGPU: Prefer GPU acceleration when available (default: true).
    public init(
        sampleRate: Float = 22050,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        winLength: Int? = nil,
        windowType: WindowType = .hann,
        center: Bool = true,
        padMode: PadMode = .constant(0),
        nMels: Int = 128,
        fMin: Float = 0,
        fMax: Float? = nil,
        htk: Bool = false,
        norm: MelNorm? = .slaney,
        useGPU: Bool = true
    ) {
        self.sampleRate = sampleRate
        self.useGPU = useGPU

        let stftConfig = STFTConfig(
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            windowType: windowType,
            center: center,
            padMode: padMode
        )
        self.stft = STFT(config: stftConfig)

        self.melFilterbank = MelFilterbank(
            nMels: nMels,
            nFFT: nFFT,
            sampleRate: sampleRate,
            fMin: fMin,
            fMax: fMax,
            htk: htk,
            norm: norm
        )

        self.filters = melFilterbank.filters()
        self.nFreqs = nFFT / 2 + 1

        // Pre-flatten filterbank for efficient vDSP_mmul: (nMels × nFreqs) row-major
        var flat = [Float](repeating: 0, count: nMels * nFreqs)
        for m in 0..<nMels {
            for f in 0..<min(nFreqs, filters[m].count) {
                flat[m * nFreqs + f] = filters[m][f]
            }
        }
        self.flatFilters = flat

        // Create sparse filterbank for efficient CPU computation
        self.sparseFilters = melFilterbank.sparseFilters()

        // Initialize Metal engine if GPU is preferred and available
        if useGPU && MetalEngine.isAvailable {
            self.metalEngine = try? MetalEngine()
        } else {
            self.metalEngine = nil
        }
    }

    /// Compute mel spectrogram from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        // Compute power spectrogram
        let powerSpec = await stft.power(signal)

        // Apply mel filterbank with GPU/CPU auto-dispatch
        return await applyFilterbank(powerSpec)
    }

    /// Compute mel spectrogram from magnitude spectrogram.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func fromMagnitude(_ magnitudeSpec: [[Float]]) async -> [[Float]] {
        // Square magnitude to get power
        let powerSpec = magnitudeSpec.map { row in
            var squared = [Float](repeating: 0, count: row.count)
            vDSP_vsq(row, 1, &squared, 1, vDSP_Length(row.count))
            return squared
        }

        return await applyFilterbank(powerSpec)
    }

    /// Compute mel spectrogram from power spectrogram.
    ///
    /// - Parameter powerSpec: Power spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func fromPower(_ powerSpec: [[Float]]) async -> [[Float]] {
        await applyFilterbank(powerSpec)
    }

    // MARK: - Private Implementation

    /// Apply mel filterbank with automatic GPU/CPU dispatch.
    private func applyFilterbank(_ powerSpec: [[Float]]) async -> [[Float]] {
        guard !powerSpec.isEmpty, let firstRow = powerSpec.first, !firstRow.isEmpty else {
            return []
        }

        let nFrames = firstRow.count
        let specNFreqs = powerSpec.count
        let nMels = filters.count
        let elementCount = nMels * nFrames * specNFreqs

        // Try GPU path for large data
        if let engine = metalEngine, MetalEngine.shouldUseGPU(elementCount: elementCount) {
            do {
                return try await engine.applyMelFilterbank(
                    spectrogram: powerSpec,
                    filterbank: filters
                )
            } catch {
                // Fall through to CPU on GPU error
            }
        }

        // Optimized CPU path using vDSP_mmul
        return applyFilterbankCPU(powerSpec)
    }

    /// Apply mel filterbank using sparse representation for efficient computation.
    ///
    /// Since mel filters are triangular with ~95% zeros, we only compute
    /// over the non-zero range of each filter, reducing operations significantly.
    private func applyFilterbankCPU(_ powerSpec: [[Float]]) -> [[Float]] {
        guard !powerSpec.isEmpty, let firstRow = powerSpec.first, !firstRow.isEmpty else {
            return []
        }

        let nFrames = firstRow.count
        let specNFreqs = powerSpec.count
        let nMels = sparseFilters.nMels

        // Output: (nMels × nFrames)
        var melSpec = [[Float]]()
        melSpec.reserveCapacity(nMels)

        // Process each mel band using sparse filter
        for m in 0..<nMels {
            var output = [Float](repeating: 0, count: nFrames)

            let freqStart = sparseFilters.startIndices[m]
            let freqEnd = sparseFilters.endIndices[m]
            let filterOffset = sparseFilters.offsets[m]

            // Clamp to actual spectrogram size
            let clampedStart = min(freqStart, specNFreqs)
            let clampedEnd = min(freqEnd, specNFreqs)

            guard clampedStart < clampedEnd && clampedStart < specNFreqs else {
                melSpec.append(output)
                continue
            }

            // Compute weighted sum over non-zero filter range
            sparseFilters.values.withUnsafeBufferPointer { filterValuesPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    // For each frequency bin in the sparse filter range
                    for k in clampedStart..<clampedEnd {
                        let filterIdx = filterOffset + (k - freqStart)

                        // Safety bounds check
                        guard filterIdx >= 0 && filterIdx < sparseFilters.values.count else { continue }
                        guard k < specNFreqs else { continue }

                        let filterVal = filterValuesPtr[filterIdx]

                        // Add weighted contribution: output += filterVal * powerSpec[k]
                        powerSpec[k].withUnsafeBufferPointer { specPtr in
                            // vDSP_vsma: D = A * B + C (scalar multiply-add)
                            var scale = filterVal
                            vDSP_vsma(
                                specPtr.baseAddress!, 1,
                                &scale,
                                outputPtr.baseAddress!, 1,
                                outputPtr.baseAddress!, 1,
                                vDSP_Length(min(nFrames, specPtr.count))
                            )
                        }
                    }
                }
            }

            melSpec.append(output)
        }

        return melSpec
    }
}
