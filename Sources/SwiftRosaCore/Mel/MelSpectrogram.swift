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
    ///   - padMode: Padding mode (default: reflect).
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
        padMode: PadMode = .reflect,
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

    /// Apply mel filterbank using optimized vDSP matrix multiplication.
    ///
    /// Matrix multiply: melSpec = filters × powerSpec
    /// - filters: (nMels × nFreqs) row-major
    /// - powerSpec: (nFreqs × nFrames) row-major
    /// - result: (nMels × nFrames) row-major
    private func applyFilterbankCPU(_ powerSpec: [[Float]]) -> [[Float]] {
        guard !powerSpec.isEmpty, let firstRow = powerSpec.first, !firstRow.isEmpty else {
            return []
        }

        let nFrames = firstRow.count
        let specNFreqs = powerSpec.count
        let nMels = filters.count

        // Flatten power spectrogram to row-major: (nFreqs × nFrames)
        // Use memcpy for each row instead of element-by-element copy
        var flatPower = [Float](repeating: 0, count: specNFreqs * nFrames)
        flatPower.withUnsafeMutableBufferPointer { flatPtr in
            for f in 0..<specNFreqs {
                let row = powerSpec[f]
                let copyLen = min(nFrames, row.count)
                row.withUnsafeBufferPointer { rowPtr in
                    memcpy(flatPtr.baseAddress! + f * nFrames, rowPtr.baseAddress!, copyLen * MemoryLayout<Float>.size)
                }
            }
        }

        // Output: (nMels × nFrames) row-major
        var flatResult = [Float](repeating: 0, count: nMels * nFrames)

        // Matrix multiply: C = A × B
        // A: (nMels × nFreqs), B: (nFreqs × nFrames), C: (nMels × nFrames)
        // Note: vDSP_mmul expects row-major storage
        flatFilters.withUnsafeBufferPointer { filterPtr in
            flatPower.withUnsafeBufferPointer { powerPtr in
                flatResult.withUnsafeMutableBufferPointer { resultPtr in
                    vDSP_mmul(
                        filterPtr.baseAddress!, 1,  // A
                        powerPtr.baseAddress!, 1,   // B
                        resultPtr.baseAddress!, 1,  // C
                        vDSP_Length(nMels),         // M (rows of A, rows of C)
                        vDSP_Length(nFrames),       // N (cols of B, cols of C)
                        vDSP_Length(nFreqs)         // K (cols of A, rows of B)
                    )
                }
            }
        }

        // Reshape to [[Float]] using Array slicing (faster than element copy)
        var melSpec = [[Float]]()
        melSpec.reserveCapacity(nMels)
        for m in 0..<nMels {
            let start = m * nFrames
            melSpec.append(Array(flatResult[start..<(start + nFrames)]))
        }

        return melSpec
    }
}
