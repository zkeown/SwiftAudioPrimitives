import Accelerate
import Foundation

/// Mel spectrogram generator.
public struct MelSpectrogram: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// STFT processor.
    public let stft: STFT

    /// Mel filterbank.
    public let melFilterbank: MelFilterbank

    /// Cached filterbank matrix.
    private let filters: [[Float]]

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
        norm: MelNorm? = .slaney
    ) {
        self.sampleRate = sampleRate

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
    }

    /// Compute mel spectrogram from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        // Compute power spectrogram
        let powerSpec = await stft.power(signal)

        // Apply mel filterbank
        return applyFilterbank(powerSpec)
    }

    /// Compute mel spectrogram from magnitude spectrogram.
    ///
    /// - Parameter magnitudeSpec: Magnitude spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func fromMagnitude(_ magnitudeSpec: [[Float]]) -> [[Float]] {
        // Square magnitude to get power
        let powerSpec = magnitudeSpec.map { row in
            var squared = [Float](repeating: 0, count: row.count)
            vDSP_vsq(row, 1, &squared, 1, vDSP_Length(row.count))
            return squared
        }

        return applyFilterbank(powerSpec)
    }

    /// Compute mel spectrogram from power spectrogram.
    ///
    /// - Parameter powerSpec: Power spectrogram of shape (nFreqs, nFrames).
    /// - Returns: Mel spectrogram of shape (nMels, nFrames).
    public func fromPower(_ powerSpec: [[Float]]) -> [[Float]] {
        applyFilterbank(powerSpec)
    }

    // MARK: - Private Implementation

    private func applyFilterbank(_ powerSpec: [[Float]]) -> [[Float]] {
        guard !powerSpec.isEmpty else { return [] }

        let nFrames = powerSpec[0].count
        let nFreqs = powerSpec.count
        let nMels = filters.count

        var melSpec = [[Float]]()
        melSpec.reserveCapacity(nMels)

        for m in 0..<nMels {
            var melRow = [Float](repeating: 0, count: nFrames)

            // Matrix-vector multiply: mel[m, t] = sum_f(filter[m, f] * power[f, t])
            for t in 0..<nFrames {
                var sum: Float = 0
                for f in 0..<min(nFreqs, filters[m].count) {
                    sum += filters[m][f] * powerSpec[f][t]
                }
                melRow[t] = sum
            }

            melSpec.append(melRow)
        }

        return melSpec
    }
}
