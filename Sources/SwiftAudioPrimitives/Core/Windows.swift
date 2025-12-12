import Accelerate
import Foundation

/// Available window function types.
public enum WindowType: String, CaseIterable, Sendable {
    case hann
    case hamming
    case blackman
    case bartlett
    case rectangular
}

/// Window function generator using Accelerate/vDSP.
public struct Windows: Sendable {
    private init() {}

    /// Generate a window function of the specified type.
    ///
    /// - Parameters:
    ///   - type: The window function type.
    ///   - length: Number of samples in the window.
    ///   - periodic: If true, generates a periodic window (for spectral analysis).
    ///               If false, generates a symmetric window (for filter design).
    /// - Returns: Array of window coefficients.
    public static func generate(_ type: WindowType, length: Int, periodic: Bool = true) -> [Float] {
        guard length > 0 else { return [] }
        guard length > 1 else { return [1.0] }

        let n = periodic ? length : length - 1

        switch type {
        case .hann:
            return generateHann(length: length, n: n)
        case .hamming:
            return generateHamming(length: length, n: n)
        case .blackman:
            return generateBlackman(length: length, n: n)
        case .bartlett:
            return generateBartlett(length: length, n: n)
        case .rectangular:
            return [Float](repeating: 1.0, count: length)
        }
    }

    // MARK: - Private Implementation

    private static func generateHann(length: Int, n: Int) -> [Float] {
        // Hann: 0.5 - 0.5 * cos(2 * pi * k / N)
        var window = [Float](repeating: 0, count: length)
        let scale = 2.0 * Float.pi / Float(n)

        for k in 0..<length {
            let cos_val = cos(scale * Float(k))
            window[k] = 0.5 - 0.5 * cos_val
        }

        return window
    }

    private static func generateHamming(length: Int, n: Int) -> [Float] {
        // Hamming: 0.54 - 0.46 * cos(2 * pi * k / N)
        var window = [Float](repeating: 0, count: length)
        let scale = 2.0 * Float.pi / Float(n)

        for k in 0..<length {
            let cos_val = cos(scale * Float(k))
            window[k] = 0.54 - 0.46 * cos_val
        }

        return window
    }

    private static func generateBlackman(length: Int, n: Int) -> [Float] {
        // Blackman: 0.42 - 0.5 * cos(2*pi*k/N) + 0.08 * cos(4*pi*k/N)
        var window = [Float](repeating: 0, count: length)
        let scale = 2.0 * Float.pi / Float(n)

        for k in 0..<length {
            let angle = scale * Float(k)
            window[k] = 0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle)
        }

        return window
    }

    private static func generateBartlett(length: Int, n: Int) -> [Float] {
        // Bartlett (triangular): 1 - |2k/N - 1|
        var window = [Float](repeating: 0, count: length)
        let halfN = Float(n) / 2.0

        for k in 0..<length {
            let normalized = Float(k) / halfN - 1.0
            window[k] = 1.0 - abs(normalized)
        }

        return window
    }
}
