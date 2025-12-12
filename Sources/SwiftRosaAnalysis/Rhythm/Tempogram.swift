import Accelerate
import Foundation
import SwiftRosaCore

/// Type of tempogram computation.
public enum TempogramType: Sendable {
    /// Autocorrelation-based tempogram (default).
    case autocorrelation
    /// Fourier-based tempogram.
    case fourier
}

/// Configuration for Tempogram computation.
public struct TempogramConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Window length for local autocorrelation.
    public let winLength: Int

    /// Whether to center frames.
    public let center: Bool

    /// Tempogram type.
    public let type: TempogramType

    /// BPM range to consider.
    public let bpmRange: (min: Float, max: Float)

    /// Create Tempogram configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - winLength: Window length for autocorrelation (default: 384).
    ///   - center: Center frames (default: true).
    ///   - type: Tempogram type (default: autocorrelation).
    ///   - bpmRange: BPM range (default: 30-300).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        winLength: Int = 384,
        center: Bool = true,
        type: TempogramType = .autocorrelation,
        bpmRange: (min: Float, max: Float) = (30, 300)
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.winLength = winLength
        self.center = center
        self.type = type
        self.bpmRange = bpmRange
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }
}

/// Tempogram computation.
///
/// Computes a tempo-based representation showing local tempo strength
/// over time, useful for rhythm analysis and beat tracking.
///
/// ## Example Usage
///
/// ```swift
/// let tempogram = Tempogram(config: TempogramConfig(bpmRange: (60, 180)))
///
/// // Compute tempogram from audio
/// let tg = await tempogram.transform(audioSignal)
///
/// // Or from pre-computed onset envelope
/// let tg = tempogram.transform(onsetEnvelope: envelope)
/// ```
public struct Tempogram: Sendable {
    /// Configuration.
    public let config: TempogramConfig

    /// Onset detector for computing onset envelope.
    private let onsetDetector: OnsetDetector

    /// Create Tempogram processor.
    ///
    /// - Parameter config: Configuration.
    public init(config: TempogramConfig = TempogramConfig()) {
        self.config = config
        self.onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: config.sampleRate,
            hopLength: config.hopLength
        ))
    }

    /// Compute tempogram from audio signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tempogram of shape (nTempoBins, nFrames).
    public func transform(_ signal: [Float]) async -> [[Float]] {
        let envelope = await onsetDetector.onsetStrength(signal)
        return transform(onsetEnvelope: envelope)
    }

    /// Compute tempogram from onset envelope.
    ///
    /// - Parameter onsetEnvelope: Pre-computed onset strength envelope.
    /// - Returns: Tempogram of shape (nTempoBins, nFrames).
    public func transform(onsetEnvelope: [Float]) -> [[Float]] {
        switch config.type {
        case .autocorrelation:
            return computeAutocorrelationTempogram(onsetEnvelope)
        case .fourier:
            return computeFourierTempogram(onsetEnvelope)
        }
    }

    /// Get the BPM values corresponding to tempogram rows.
    ///
    /// - Returns: Array of BPM values.
    public var bpmAxis: [Float] {
        let fps = config.framesPerSecond
        let lagMin = Int(60.0 * fps / config.bpmRange.max)
        let lagMax = Int(60.0 * fps / config.bpmRange.min)
        let nLags = lagMax - lagMin + 1

        var bpms = [Float](repeating: 0, count: nLags)
        for i in 0..<nLags {
            let lag = lagMin + i
            if lag > 0 {
                bpms[i] = 60.0 * fps / Float(lag)
            }
        }
        return bpms
    }

    // MARK: - Private Implementation

    /// Compute autocorrelation-based tempogram.
    private func computeAutocorrelationTempogram(_ envelope: [Float]) -> [[Float]] {
        guard !envelope.isEmpty else { return [] }

        let fps = config.framesPerSecond
        let lagMin = max(1, Int(60.0 * fps / config.bpmRange.max))
        let lagMax = min(config.winLength - 1, Int(60.0 * fps / config.bpmRange.min))
        let nLags = lagMax - lagMin + 1

        guard nLags > 0 else { return [] }

        // Pad envelope if centering
        let paddedEnvelope: [Float]
        if config.center {
            let padLength = config.winLength / 2
            paddedEnvelope = padEnvelope(envelope, padLength: padLength)
        } else {
            paddedEnvelope = envelope
        }

        // Compute number of output frames
        let nFrames = max(0, 1 + (paddedEnvelope.count - config.winLength) / config.hopLength)
        guard nFrames > 0 else { return [] }

        // Initialize output
        var tempogram = [[Float]]()
        tempogram.reserveCapacity(nLags)
        for _ in 0..<nLags {
            tempogram.append([Float](repeating: 0, count: nFrames))
        }

        // Hann window for smoothing
        let window = Windows.generate(.hann, length: config.winLength, periodic: true)

        // Process each frame
        for frameIdx in 0..<nFrames {
            let start = frameIdx * config.hopLength
            let end = min(start + config.winLength, paddedEnvelope.count)
            let frameLength = end - start

            guard frameLength >= lagMax else { continue }

            // Extract and window the frame
            var frame = [Float](repeating: 0, count: frameLength)
            for i in 0..<frameLength {
                frame[i] = paddedEnvelope[start + i] * window[i]
            }

            // Compute autocorrelation for each lag
            for lagIdx in 0..<nLags {
                let lag = lagMin + lagIdx
                var acf: Float = 0

                // Autocorrelation: sum(x[t] * x[t + lag])
                for t in 0..<(frameLength - lag) {
                    acf += frame[t] * frame[t + lag]
                }

                // Normalize
                let norm = Float(frameLength - lag)
                if norm > 0 {
                    acf /= norm
                }

                tempogram[lagIdx][frameIdx] = acf
            }
        }

        return tempogram
    }

    /// Compute Fourier-based tempogram.
    private func computeFourierTempogram(_ envelope: [Float]) -> [[Float]] {
        guard !envelope.isEmpty else { return [] }

        let fps = config.framesPerSecond

        // Pad envelope if centering
        let paddedEnvelope: [Float]
        if config.center {
            let padLength = config.winLength / 2
            paddedEnvelope = padEnvelope(envelope, padLength: padLength)
        } else {
            paddedEnvelope = envelope
        }

        // Compute number of output frames
        let nFrames = max(0, 1 + (paddedEnvelope.count - config.winLength) / config.hopLength)
        guard nFrames > 0 else { return [] }

        // Compute frequency bins for BPM range
        let bpmMin = config.bpmRange.min
        let bpmMax = config.bpmRange.max
        let freqMin = bpmMin / 60.0  // Hz
        let freqMax = bpmMax / 60.0  // Hz

        // FFT setup
        let fftSize = nextPowerOfTwo(config.winLength)
        let log2n = vDSP_Length(log2(Double(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Compute frequency resolution
        let freqResolution = fps / Float(fftSize)

        // Find bin range
        let binMin = max(1, Int(freqMin / freqResolution))
        let binMax = min(fftSize / 2, Int(freqMax / freqResolution) + 1)
        let nBins = binMax - binMin

        guard nBins > 0 else { return [] }

        // Initialize output
        var tempogram = [[Float]]()
        tempogram.reserveCapacity(nBins)
        for _ in 0..<nBins {
            tempogram.append([Float](repeating: 0, count: nFrames))
        }

        // Hann window
        let window = Windows.generate(.hann, length: config.winLength, periodic: true)

        // FFT buffers
        var realInput = [Float](repeating: 0, count: fftSize / 2)
        var imagInput = [Float](repeating: 0, count: fftSize / 2)
        var windowedFrame = [Float](repeating: 0, count: fftSize)

        // Process each frame
        for frameIdx in 0..<nFrames {
            let start = frameIdx * config.hopLength
            let end = min(start + config.winLength, paddedEnvelope.count)
            let frameLength = end - start

            // Zero the buffer
            windowedFrame = [Float](repeating: 0, count: fftSize)

            // Window and copy
            for i in 0..<frameLength {
                windowedFrame[i] = paddedEnvelope[start + i] * window[i]
            }

            // FFT
            realInput = [Float](repeating: 0, count: fftSize / 2)
            imagInput = [Float](repeating: 0, count: fftSize / 2)

            realInput.withUnsafeMutableBufferPointer { realPtr in
                imagInput.withUnsafeMutableBufferPointer { imagPtr in
                    var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                    windowedFrame.withUnsafeBufferPointer { inputPtr in
                        vDSP_ctoz(
                            UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                            2, &split, 1, vDSP_Length(fftSize / 2)
                        )
                    }

                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }

            // Extract magnitude in BPM range
            for binIdx in 0..<nBins {
                let fftBin = binMin + binIdx
                if fftBin < fftSize / 2 {
                    let real = realInput[fftBin]
                    let imag = imagInput[fftBin]
                    tempogram[binIdx][frameIdx] = sqrt(real * real + imag * imag)
                }
            }
        }

        return tempogram
    }

    private func padEnvelope(_ envelope: [Float], padLength: Int) -> [Float] {
        guard !envelope.isEmpty else { return envelope }

        var padded = [Float]()
        padded.reserveCapacity(envelope.count + 2 * padLength)

        // Reflect padding
        for i in (1...padLength).reversed() {
            let idx = i % envelope.count
            padded.append(envelope[idx])
        }

        padded.append(contentsOf: envelope)

        for i in 1...padLength {
            let idx = (envelope.count - 1) - (i % envelope.count)
            padded.append(envelope[max(0, idx)])
        }

        return padded
    }

    private func nextPowerOfTwo(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }
}

// MARK: - Presets

extension Tempogram {
    /// Standard tempogram for general use.
    public static var standard: Tempogram {
        Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 384,
            type: .autocorrelation,
            bpmRange: (30, 300)
        ))
    }

    /// Tempogram for music with typical tempos (60-180 BPM).
    public static var music: Tempogram {
        Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 512,
            type: .autocorrelation,
            bpmRange: (60, 180)
        ))
    }
}
