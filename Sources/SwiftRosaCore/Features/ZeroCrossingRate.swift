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
        guard nFrames > 0 else { return [] }

        // OPTIMIZED: Precompute ALL crossings once, then use cumulative sum
        // This is O(N) + O(nFrames) instead of O(nFrames × frameLength)

        let n = paddedSignal.count - 1
        guard n > 0 else { return [Float](repeating: 0, count: nFrames) }

        // Step 1: Compute sign products for entire signal (vectorized)
        var signProduct = [Float](repeating: 0, count: n)
        paddedSignal.withUnsafeBufferPointer { sigPtr in
            signProduct.withUnsafeMutableBufferPointer { prodPtr in
                vDSP_vmul(
                    sigPtr.baseAddress!, 1,
                    sigPtr.baseAddress! + 1, 1,
                    prodPtr.baseAddress!, 1,
                    vDSP_Length(n)
                )
            }
        }

        // Step 2: Create binary crossing indicators
        // signProduct < 0 means crossing occurred
        var crossings = [Float](repeating: 0, count: n)
        for i in 0..<n {
            crossings[i] = signProduct[i] < 0 ? 1.0 : 0.0
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
    /// - Parameter signal: Input audio signal.
    /// - Returns: Zero-crossing rate (0 to 1).
    public static func compute(_ signal: [Float]) -> Float {
        guard signal.count > 1 else { return 0 }

        let n = signal.count - 1

        // Compute x[i] * x[i-1] - negative values indicate zero crossings
        var signProduct = [Float](repeating: 0, count: n)

        signal.withUnsafeBufferPointer { signalPtr in
            signProduct.withUnsafeMutableBufferPointer { prodPtr in
                vDSP_vmul(
                    signalPtr.baseAddress!, 1,      // x[i-1]
                    signalPtr.baseAddress! + 1, 1,  // x[i]
                    prodPtr.baseAddress!, 1,
                    vDSP_Length(n)
                )
            }
        }

        // Count zero crossings (product < 0)
        var crossings: Int = 0
        for i in 0..<n {
            if signProduct[i] < 0 {
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
    /// - Parameter signal: Input audio signal.
    /// - Returns: Total number of zero crossings.
    public static func count(_ signal: [Float]) -> Int {
        guard signal.count > 1 else { return 0 }

        let n = signal.count - 1

        return signal.withUnsafeBufferPointer { signalPtr in
            var signProduct = [Float](repeating: 0, count: n)

            return signProduct.withUnsafeMutableBufferPointer { prodPtr in
                vDSP_vmul(
                    signalPtr.baseAddress!, 1,
                    signalPtr.baseAddress! + 1, 1,
                    prodPtr.baseAddress!, 1,
                    vDSP_Length(n)
                )

                // Count zero crossings (product < 0)
                var crossings: Int = 0
                for i in 0..<n {
                    if prodPtr[i] < 0 {
                        crossings += 1
                    }
                }
                return crossings
            }
        }
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
