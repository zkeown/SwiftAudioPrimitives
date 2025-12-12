import Accelerate
import Foundation

/// Padding mode for signal centering.
public enum PadMode: Sendable {
    /// Reflect signal at boundaries (default, matches librosa).
    case reflect
    /// Pad with constant value.
    case constant(Float)
    /// Pad with edge values.
    case edge
}

/// Configuration for STFT computation.
public struct STFTConfig: Sendable {
    /// FFT size. Default: 2048.
    public let nFFT: Int

    /// Hop length between frames. Default: nFFT/4.
    public let hopLength: Int

    /// Window length. Default: nFFT.
    public let winLength: Int

    /// Window function type. Default: hann.
    public let windowType: WindowType

    /// Whether to center the signal by padding. Default: true.
    public let center: Bool

    /// Padding mode when center is true. Default: reflect.
    public let padMode: PadMode

    /// Create STFT configuration.
    ///
    /// - Parameters:
    ///   - nFFT: FFT size (default: 2048).
    ///   - hopLength: Hop between frames (default: nFFT/4).
    ///   - winLength: Window length (default: nFFT).
    ///   - windowType: Window function (default: hann).
    ///   - center: Pad signal for centered frames (default: true).
    ///   - padMode: Padding mode (default: reflect).
    public init(
        nFFT: Int = 2048,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        windowType: WindowType = .hann,
        center: Bool = true,
        padMode: PadMode = .reflect
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength ?? (nFFT / 4)
        self.winLength = winLength ?? nFFT
        self.windowType = windowType
        self.center = center
        self.padMode = padMode
    }

    /// Number of frequency bins in output (nFFT/2 + 1 for real input).
    public var nFreqs: Int { nFFT / 2 + 1 }
}

/// Wrapper for FFTSetup to make it Sendable
/// Note: FFTSetup is thread-safe for read operations after creation
final class FFTSetupWrapper: @unchecked Sendable {
    let setup: FFTSetup

    init(log2n: vDSP_Length) {
        self.setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }
}

/// Short-Time Fourier Transform.
public struct STFT: Sendable {
    public let config: STFTConfig

    private let window: [Float]
    private let fftSetupWrapper: FFTSetupWrapper

    /// Create STFT processor.
    ///
    /// - Parameter config: STFT configuration.
    public init(config: STFTConfig = STFTConfig()) {
        self.config = config
        self.window = Windows.generate(config.windowType, length: config.winLength, periodic: true)

        // Create FFT setup
        let log2n = vDSP_Length(log2(Double(config.nFFT)))
        self.fftSetupWrapper = FFTSetupWrapper(log2n: log2n)
    }

    /// Compute STFT of a signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Complex spectrogram of shape (nFreqs, nFrames).
    public func transform(_ signal: [Float]) async -> ComplexMatrix {
        let paddedSignal = config.center ? padSignal(signal) : signal

        let frames = frameSignal(paddedSignal)
        let nFrames = frames.count

        // Apply window to each frame and compute FFT
        var realResult = [[Float]]()
        var imagResult = [[Float]]()
        realResult.reserveCapacity(config.nFreqs)
        imagResult.reserveCapacity(config.nFreqs)

        // Initialize output arrays
        for _ in 0..<config.nFreqs {
            realResult.append([Float](repeating: 0, count: nFrames))
            imagResult.append([Float](repeating: 0, count: nFrames))
        }

        // Process each frame
        for (frameIdx, frame) in frames.enumerated() {
            // Apply window
            var windowedFrame = [Float](repeating: 0, count: config.nFFT)

            // Copy and window the frame (zero-pad if winLength < nFFT)
            for i in 0..<min(frame.count, config.winLength) {
                windowedFrame[i] = frame[i] * window[i]
            }

            // Compute FFT
            let (fftReal, fftImag) = computeFFT(windowedFrame)

            // Store results (only positive frequencies)
            for freqIdx in 0..<config.nFreqs {
                realResult[freqIdx][frameIdx] = fftReal[freqIdx]
                imagResult[freqIdx][frameIdx] = fftImag[freqIdx]
            }
        }

        return ComplexMatrix(real: realResult, imag: imagResult)
    }

    /// Compute magnitude spectrogram.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Magnitude spectrogram of shape (nFreqs, nFrames).
    public func magnitude(_ signal: [Float]) async -> [[Float]] {
        let spectrogram = await transform(signal)
        return spectrogram.magnitude
    }

    /// Compute power spectrogram (magnitude squared).
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Power spectrogram of shape (nFreqs, nFrames).
    public func power(_ signal: [Float]) async -> [[Float]] {
        let spectrogram = await transform(signal)
        let mag = spectrogram.magnitude

        // Square the magnitude
        return mag.map { row in
            var squared = [Float](repeating: 0, count: row.count)
            vDSP_vsq(row, 1, &squared, 1, vDSP_Length(row.count))
            return squared
        }
    }

    // MARK: - Private Implementation

    private func padSignal(_ signal: [Float]) -> [Float] {
        let padLength = config.nFFT / 2

        switch config.padMode {
        case .reflect:
            return reflectPad(signal, padLength: padLength)
        case .constant(let value):
            let padding = [Float](repeating: value, count: padLength)
            return padding + signal + padding
        case .edge:
            let leftPad = [Float](repeating: signal.first ?? 0, count: padLength)
            let rightPad = [Float](repeating: signal.last ?? 0, count: padLength)
            return leftPad + signal + rightPad
        }
    }

    private func reflectPad(_ signal: [Float], padLength: Int) -> [Float] {
        guard !signal.isEmpty else { return signal }

        var result = [Float]()
        result.reserveCapacity(signal.count + 2 * padLength)

        // Left padding (reflect)
        for i in (1...padLength).reversed() {
            let idx = i % signal.count
            result.append(signal[idx])
        }

        // Original signal
        result.append(contentsOf: signal)

        // Right padding (reflect)
        for i in 1...padLength {
            let idx = (signal.count - 1) - (i % signal.count)
            result.append(signal[max(0, idx)])
        }

        return result
    }

    private func frameSignal(_ signal: [Float]) -> [[Float]] {
        let nFrames = 1 + (signal.count - config.nFFT) / config.hopLength
        var frames = [[Float]]()
        frames.reserveCapacity(nFrames)

        for i in 0..<nFrames {
            let start = i * config.hopLength
            let end = min(start + config.nFFT, signal.count)
            var frame = Array(signal[start..<end])

            // Zero-pad if necessary
            if frame.count < config.nFFT {
                frame.append(contentsOf: [Float](repeating: 0, count: config.nFFT - frame.count))
            }

            frames.append(frame)
        }

        return frames
    }

    private func computeFFT(_ input: [Float]) -> (real: [Float], imag: [Float]) {
        let n = config.nFFT
        let log2n = vDSP_Length(log2(Double(n)))

        // Prepare input in split complex format
        var realInput = [Float](repeating: 0, count: n / 2)
        var imagInput = [Float](repeating: 0, count: n / 2)

        // Convert to split complex format for FFT
        realInput.withUnsafeMutableBufferPointer { realPtr in
            imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                // Pack real data into split complex format
                input.withUnsafeBufferPointer { inputPtr in
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(n / 2)
                    )
                }

                // Perform FFT
                vDSP_fft_zrip(fftSetupWrapper.setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                // Scale by 1/2 (vDSP FFT convention)
                var scale: Float = 0.5
                vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, vDSP_Length(n / 2))
                vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, vDSP_Length(n / 2))
            }
        }

        // Unpack to full complex spectrum (positive frequencies only)
        var fftReal = [Float](repeating: 0, count: n / 2 + 1)
        var fftImag = [Float](repeating: 0, count: n / 2 + 1)

        // DC component
        fftReal[0] = realInput[0]
        fftImag[0] = 0

        // Positive frequencies
        for i in 1..<(n / 2) {
            fftReal[i] = realInput[i]
            fftImag[i] = imagInput[i]
        }

        // Nyquist
        fftReal[n / 2] = imagInput[0]
        fftImag[n / 2] = 0

        return (fftReal, fftImag)
    }
}
