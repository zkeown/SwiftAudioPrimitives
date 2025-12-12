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

    /// Cache for generated windows to avoid recomputation.
    private static var windowCache: [String: [Float]] = [:]
    private static let cacheLock = NSLock()

    /// Generate a window function of the specified type.
    ///
    /// Results are cached for performance - repeated calls with the same
    /// parameters return the cached window without recomputation.
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

        // Check cache first
        let key = "\(type.rawValue)_\(length)_\(periodic)"
        cacheLock.lock()
        if let cached = windowCache[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let n = periodic ? length : length - 1

        let window: [Float]
        switch type {
        case .hann:
            window = generateHann(length: length, n: n)
        case .hamming:
            window = generateHamming(length: length, n: n)
        case .blackman:
            window = generateBlackman(length: length, n: n)
        case .bartlett:
            window = generateBartlett(length: length, n: n)
        case .rectangular:
            window = [Float](repeating: 1.0, count: length)
        }

        // Store in cache
        cacheLock.lock()
        windowCache[key] = window
        cacheLock.unlock()

        return window
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
        // Note: At k=0, the formula equals exactly 0 (0.42 - 0.5 + 0.08 = 0)
        // but floating point precision can produce tiny negative values.
        // We clamp to ensure non-negativity as required by window function definition.
        var window = [Float](repeating: 0, count: length)
        let scale = 2.0 * Float.pi / Float(n)

        for k in 0..<length {
            let angle = scale * Float(k)
            let value = 0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle)
            window[k] = max(0, value)
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
