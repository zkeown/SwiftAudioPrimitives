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

    /// Threshold for treating values as zero (matches librosa default of 1e-10).
    /// Values with absolute value <= threshold are treated as zero.
    public let threshold: Float

    /// Create a zero-crossing rate extractor.
    ///
    /// - Parameters:
    ///   - frameLength: Frame length in samples (default: 2048).
    ///   - hopLength: Hop length between frames (default: 512).
    ///   - center: Center frames by padding (default: true).
    ///   - threshold: Values within this threshold are treated as zero (default: 1e-10).
    public init(
        frameLength: Int = 2048,
        hopLength: Int = 512,
        center: Bool = true,
        threshold: Float = 1e-10
    ) {
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.center = center
        self.threshold = threshold
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
        guard nFrames > 0 else { return [] }

        // OPTIMIZED: Precompute ALL crossings once, then use cumulative sum
        // This is O(N) + O(nFrames) instead of O(nFrames × frameLength)

        let n = paddedSignal.count - 1
        guard n > 0 else { return [Float](repeating: 0, count: nFrames) }

        // Step 1: Compute crossings using librosa's signbit method with zero_pos=True
        // librosa clips values within threshold to 0, then treats 0 as positive (signbit=false)
        // A crossing occurs when signbit(x[i]) != signbit(x[i-1])
        // With zero_pos=True: signbit is true only for negative values (not zero)
        var crossings = [Float](repeating: 0, count: n)
        let thresh = threshold
        paddedSignal.withUnsafeBufferPointer { sigPtr in
            crossings.withUnsafeMutableBufferPointer { crossPtr in
                for i in 0..<n {
                    // Clip values within threshold to zero
                    let x0 = abs(sigPtr[i]) <= thresh ? Float(0) : sigPtr[i]
                    let x1 = abs(sigPtr[i + 1]) <= thresh ? Float(0) : sigPtr[i + 1]

                    // signbit: true for negative, false for zero/positive (zero_pos=True)
                    let sign0 = x0 < 0
                    let sign1 = x1 < 0

                    // Crossing when signs differ
                    crossPtr[i] = sign0 != sign1 ? 1.0 : 0.0
                }
            }
        }

        // Step 3: Compute cumulative sum
        // Note: vDSP_vrsum has incorrect indexing for our use case, use manual cumsum
        var cumsum = [Float](repeating: 0, count: n + 1)
        cumsum[0] = 0
        for i in 0..<n {
            cumsum[i + 1] = cumsum[i] + crossings[i]
        }

        // Step 4: Compute per-frame ZCR using cumsum differences
        var result = [Float](repeating: 0, count: nFrames)
        // Normalize by frameLength (not frameLength-1) to match librosa's behavior
        // librosa uses np.mean(crossings) where crossings array includes first sample as 0
        let invFrameLength = 1.0 / Float(frameLength)

        result.withUnsafeMutableBufferPointer { resultPtr in
            cumsum.withUnsafeBufferPointer { csPtr in
                for frameIdx in 0..<nFrames {
                    let start = frameIdx * hopLength
                    // The cumsum array has n+1 elements (indices 0..n)
                    // Frame of length L covers samples start..<start+L, which has L-1 crossing opportunities
                    // Cumsum[i] = total crossings in samples 0..i (exclusive of i)
                    // So crossings in frame = cumsum[start + frameLength - 1] - cumsum[start]
                    let end = min(start + frameLength - 1, n)

                    if end > start {
                        // Number of crossings in frame = cumsum[end] - cumsum[start]
                        let frameCrossings = csPtr[end] - csPtr[start]
                        resultPtr[frameIdx] = frameCrossings * invFrameLength
                    }
                }
            }
        }

        return result
    }

    /// Compute zero-crossing rate of entire signal.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - threshold: Values within this threshold are treated as zero (default: 1e-10).
    /// - Returns: Zero-crossing rate (0 to 1).
    public static func compute(_ signal: [Float], threshold: Float = 1e-10) -> Float {
        guard signal.count > 1 else { return 0 }

        let n = signal.count - 1

        // Count zero crossings using librosa's signbit method with zero_pos=True
        var crossings: Int = 0
        for i in 0..<n {
            // Clip values within threshold to zero
            let x0 = abs(signal[i]) <= threshold ? Float(0) : signal[i]
            let x1 = abs(signal[i + 1]) <= threshold ? Float(0) : signal[i + 1]

            // signbit: true for negative, false for zero/positive (zero_pos=True)
            let sign0 = x0 < 0
            let sign1 = x1 < 0

            // Crossing when signs differ
            if sign0 != sign1 {
                crossings += 1
            }
        }
        return Float(crossings) / Float(n)
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
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - threshold: Values within this threshold are treated as zero (default: 1e-10).
    /// - Returns: Total number of zero crossings.
    public static func count(_ signal: [Float], threshold: Float = 1e-10) -> Int {
        guard signal.count > 1 else { return 0 }

        let n = signal.count - 1
        var crossings: Int = 0

        for i in 0..<n {
            // Clip values within threshold to zero
            let x0 = abs(signal[i]) <= threshold ? Float(0) : signal[i]
            let x1 = abs(signal[i + 1]) <= threshold ? Float(0) : signal[i + 1]

            // signbit: true for negative, false for zero/positive (zero_pos=True)
            let sign0 = x0 < 0
            let sign1 = x1 < 0

            // Crossing when signs differ
            if sign0 != sign1 {
                crossings += 1
            }
        }
        return crossings
    }

    /// Find zero-crossing positions in a signal.
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - threshold: Values within this threshold are treated as zero (default: 1e-10).
    /// - Returns: Array of sample indices where zero crossings occur.
    public static func positions(_ signal: [Float], threshold: Float = 1e-10) -> [Int] {
        guard signal.count > 1 else { return [] }

        var positions = [Int]()

        for i in 1..<signal.count {
            // Clip values within threshold to zero
            let x0 = abs(signal[i - 1]) <= threshold ? Float(0) : signal[i - 1]
            let x1 = abs(signal[i]) <= threshold ? Float(0) : signal[i]

            // signbit: true for negative, false for zero/positive (zero_pos=True)
            // A crossing occurs when one is negative and the other is not
            let sign0 = x0 < 0
            let sign1 = x1 < 0
            if sign0 != sign1 {
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

        // Left padding (edge) - librosa uses edge padding for ZCR
        let leftValue = signal[0]
        for _ in 0..<padLength {
            result.append(leftValue)
        }

        // Original signal
        result.append(contentsOf: signal)

        // Right padding (edge)
        let rightValue = signal[signal.count - 1]
        for _ in 0..<padLength {
            result.append(rightValue)
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
