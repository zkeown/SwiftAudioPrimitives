import Accelerate
import Foundation

/// Zero-crossing rate computation.
///
/// The zero-crossing rate (ZCR) is the rate at which the signal changes sign.
/// It's useful for distinguishing between voiced and unvoiced speech, and
/// for simple pitch estimation.
///
/// ## Example Usage
///
/// ```swift
/// let zcr = ZeroCrossingRate(frameLength: 2048, hopLength: 512)
///
/// // Frame-wise ZCR
/// let frameZCR = zcr.transform(audioSignal)
///
/// // Global ZCR for entire signal
/// let globalZCR = ZeroCrossingRate.compute(audioSignal)
/// ```
public struct ZeroCrossingRate: Sendable {
    /// Frame length in samples.
    public let frameLength: Int

    /// Hop length between frames.
    public let hopLength: Int

    /// Whether to center frames by padding.
    public let center: Bool

    /// Create a zero-crossing rate extractor.
    ///
    /// - Parameters:
    ///   - frameLength: Frame length in samples (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - center: Center frames by padding (default: true).
    public init(
        frameLength: Int = 2048,
        hopLength: Int = 512,
        center: Bool = true
    ) {
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.center = center
    }

    /// Compute frame-wise zero-crossing rate.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Zero-crossing rate per frame (0 to 1).
    public func transform(_ signal: [Float]) -> [Float] {
        guard !signal.isEmpty else { return [] }

        let paddedSignal: [Float]
        if center {
            let padLength = frameLength / 2
            paddedSignal = padSignal(signal, padLength: padLength)
        } else {
            paddedSignal = signal
        }

        let nFrames = max(0, 1 + (paddedSignal.count - frameLength) / hopLength)
        var result = [Float](repeating: 0, count: nFrames)

        for frameIdx in 0..<nFrames {
            let start = frameIdx * hopLength
            let end = min(start + frameLength, paddedSignal.count)

            // Count zero crossings in this frame
            var crossings = 0
            for i in (start + 1)..<end {
                let sign1 = paddedSignal[i - 1] >= 0
                let sign2 = paddedSignal[i] >= 0
                if sign1 != sign2 {
                    crossings += 1
                }
            }

            // Normalize by frame length
            let frameLen = end - start
            result[frameIdx] = frameLen > 1 ? Float(crossings) / Float(frameLen - 1) : 0
        }

        return result
    }

    /// Compute zero-crossing rate of entire signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Zero-crossing rate (0 to 1).
    public static func compute(_ signal: [Float]) -> Float {
        guard signal.count > 1 else { return 0 }

        var crossings = 0
        for i in 1..<signal.count {
            let sign1 = signal[i - 1] >= 0
            let sign2 = signal[i] >= 0
            if sign1 != sign2 {
                crossings += 1
            }
        }

        return Float(crossings) / Float(signal.count - 1)
    }

    /// Compute zero-crossing rate with threshold.
    ///
    /// Only counts crossings that exceed a threshold distance from zero,
    /// which can help reduce noise sensitivity.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - threshold: Minimum distance from zero to count as crossing.
    /// - Returns: Zero-crossing rate (0 to 1).
    public static func computeWithThreshold(_ signal: [Float], threshold: Float) -> Float {
        guard signal.count > 1 else { return 0 }

        var crossings = 0
        for i in 1..<signal.count {
            // Check if both samples exceed threshold on opposite sides
            let above1 = signal[i - 1] > threshold
            let below1 = signal[i - 1] < -threshold
            let above2 = signal[i] > threshold
            let below2 = signal[i] < -threshold

            if (above1 && below2) || (below1 && above2) {
                crossings += 1
            }
        }

        return Float(crossings) / Float(signal.count - 1)
    }

    /// Count zero crossings in a signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Total number of zero crossings.
    public static func count(_ signal: [Float]) -> Int {
        guard signal.count > 1 else { return 0 }

        var crossings = 0
        for i in 1..<signal.count {
            let sign1 = signal[i - 1] >= 0
            let sign2 = signal[i] >= 0
            if sign1 != sign2 {
                crossings += 1
            }
        }

        return crossings
    }

    /// Find zero-crossing positions in a signal.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of sample indices where zero crossings occur.
    public static func positions(_ signal: [Float]) -> [Int] {
        guard signal.count > 1 else { return [] }

        var positions = [Int]()

        for i in 1..<signal.count {
            let sign1 = signal[i - 1] >= 0
            let sign2 = signal[i] >= 0
            if sign1 != sign2 {
                positions.append(i)
            }
        }

        return positions
    }

    // MARK: - Private Implementation

    private func padSignal(_ signal: [Float], padLength: Int) -> [Float] {
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
}

// MARK: - Presets

extension ZeroCrossingRate {
    /// Configuration for speech analysis.
    public static var speechAnalysis: ZeroCrossingRate {
        ZeroCrossingRate(
            frameLength: 512,
            hopLength: 160,
            center: true
        )
    }

    /// Configuration for music analysis.
    public static var musicAnalysis: ZeroCrossingRate {
        ZeroCrossingRate(
            frameLength: 2048,
            hopLength: 512,
            center: true
        )
    }

    /// Configuration matching librosa defaults.
    public static var librosaDefault: ZeroCrossingRate {
        ZeroCrossingRate(
            frameLength: 2048,
            hopLength: 512,
            center: true
        )
    }
}
