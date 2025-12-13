import Foundation

/// Pre-emphasis and de-emphasis filters for audio signals.
///
/// Pre-emphasis is commonly used in speech processing to boost high frequencies
/// before analysis, compensating for the natural spectral tilt of speech.
///
/// ## Example Usage
///
/// ```swift
/// // Apply pre-emphasis before computing MFCCs
/// let emphasized = Emphasis.preemphasis(audio, coef: 0.97)
/// let mfcc = await mfccExtractor.transform(emphasized)
///
/// // Remove pre-emphasis from processed audio
/// let restored = Emphasis.deemphasis(processed, coef: 0.97)
/// ```
public struct Emphasis: Sendable {
    private init() {}

    /// Apply pre-emphasis filter to a signal.
    ///
    /// Pre-emphasis boosts high frequencies using a first-order high-pass filter:
    /// `y[n] = x[n] - coef * x[n-1]`
    ///
    /// This is commonly used before spectral analysis to:
    /// - Compensate for the natural spectral tilt of speech (-6 dB/octave)
    /// - Improve numerical stability in linear prediction
    /// - Balance the frequency spectrum before feature extraction
    ///
    /// - Parameters:
    ///   - signal: Input audio signal.
    ///   - coef: Pre-emphasis coefficient (default: 0.97). Higher values = more emphasis.
    ///           Must be in range [0, 1). Typical values: 0.95-0.97 for speech.
    /// - Returns: Pre-emphasized signal with same length as input.
    ///
    /// - Note: Matches `librosa.effects.preemphasis(y, coef=0.97)`
    public static func preemphasis(_ signal: [Float], coef: Float = 0.97) -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard coef >= 0 && coef < 1 else {
            // Invalid coefficient, return unchanged
            return signal
        }

        var result = [Float](repeating: 0, count: signal.count)

        // First sample is unchanged (or could use x[0] - coef * x[0], but librosa uses x[0])
        result[0] = signal[0]

        // Apply first-order difference filter
        for i in 1..<signal.count {
            result[i] = signal[i] - coef * signal[i - 1]
        }

        return result
    }

    /// Apply de-emphasis filter to a signal.
    ///
    /// De-emphasis is the inverse of pre-emphasis, restoring the original spectral balance:
    /// `y[n] = x[n] + coef * y[n-1]`
    ///
    /// Use this to restore audio that was pre-emphasized before processing.
    ///
    /// - Parameters:
    ///   - signal: Input signal (typically pre-emphasized).
    ///   - coef: De-emphasis coefficient (default: 0.97). Should match the pre-emphasis coef.
    ///           Must be in range [0, 1).
    /// - Returns: De-emphasized signal with same length as input.
    ///
    /// - Note: Matches `librosa.effects.deemphasis(y, coef=0.97)`
    public static func deemphasis(_ signal: [Float], coef: Float = 0.97) -> [Float] {
        guard !signal.isEmpty else { return [] }
        guard coef >= 0 && coef < 1 else {
            // Invalid coefficient, return unchanged
            return signal
        }

        var result = [Float](repeating: 0, count: signal.count)

        // First sample is unchanged
        result[0] = signal[0]

        // Apply first-order IIR filter (recursive)
        for i in 1..<signal.count {
            result[i] = signal[i] + coef * result[i - 1]
        }

        return result
    }
}

// MARK: - Array Extension for Convenience

public extension Array where Element == Float {
    /// Apply pre-emphasis filter.
    ///
    /// - Parameter coef: Pre-emphasis coefficient (default: 0.97).
    /// - Returns: Pre-emphasized signal.
    func preemphasized(coef: Float = 0.97) -> [Float] {
        return Emphasis.preemphasis(self, coef: coef)
    }

    /// Apply de-emphasis filter.
    ///
    /// - Parameter coef: De-emphasis coefficient (default: 0.97).
    /// - Returns: De-emphasized signal.
    func deemphasized(coef: Float = 0.97) -> [Float] {
        return Emphasis.deemphasis(self, coef: coef)
    }
}
