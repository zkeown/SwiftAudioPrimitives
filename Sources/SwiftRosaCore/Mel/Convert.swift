import Accelerate
import Foundation

/// dB conversion utilities.
public struct Convert {
    private init() {}

    /// Convert power spectrogram to decibels.
    ///
    /// - Parameters:
    ///   - S: Power spectrogram.
    ///   - ref: Reference power (default: 1.0). Can also be a function like max(S).
    ///   - amin: Minimum amplitude to avoid log(0) (default: 1e-10).
    ///   - topDb: Threshold below max (default: 80.0). Set to nil to disable.
    /// - Returns: Spectrogram in dB.
    public static func powerToDb(
        _ S: [[Float]],
        ref: Float = 1.0,
        amin: Float = 1e-10,
        topDb: Float? = 80.0
    ) -> [[Float]] {
        guard !S.isEmpty else { return [] }

        let rows = S.count
        let cols = S[0].count

        var result = [[Float]]()
        result.reserveCapacity(rows)

        // Compute 10 * log10(max(S, amin) / ref)
        let logRef = 10.0 * log10(max(ref, amin))

        for row in S {
            var dbRow = [Float](repeating: 0, count: cols)

            for i in 0..<cols {
                let clampedValue = max(row[i], amin)
                dbRow[i] = 10.0 * log10(clampedValue) - logRef
            }

            result.append(dbRow)
        }

        // Apply top_db threshold
        if let topDb = topDb {
            let maxVal = result.flatMap { $0 }.max() ?? 0
            let threshold = maxVal - topDb

            for i in 0..<rows {
                for j in 0..<cols {
                    result[i][j] = max(result[i][j], threshold)
                }
            }
        }

        return result
    }

    /// Convert amplitude spectrogram to decibels.
    ///
    /// - Parameters:
    ///   - S: Amplitude (magnitude) spectrogram.
    ///   - ref: Reference amplitude (default: 1.0).
    ///   - amin: Minimum amplitude (default: 1e-5).
    ///   - topDb: Threshold below max (default: 80.0).
    /// - Returns: Spectrogram in dB.
    public static func amplitudeToDb(
        _ S: [[Float]],
        ref: Float = 1.0,
        amin: Float = 1e-5,
        topDb: Float? = 80.0
    ) -> [[Float]] {
        guard !S.isEmpty else { return [] }

        let rows = S.count
        let cols = S[0].count

        var result = [[Float]]()
        result.reserveCapacity(rows)

        // Compute 20 * log10(max(S, amin) / ref)
        let logRef = 20.0 * log10(max(ref, amin))

        for row in S {
            var dbRow = [Float](repeating: 0, count: cols)

            for i in 0..<cols {
                let clampedValue = max(row[i], amin)
                dbRow[i] = 20.0 * log10(clampedValue) - logRef
            }

            result.append(dbRow)
        }

        // Apply top_db threshold
        if let topDb = topDb {
            let maxVal = result.flatMap { $0 }.max() ?? 0
            let threshold = maxVal - topDb

            for i in 0..<rows {
                for j in 0..<cols {
                    result[i][j] = max(result[i][j], threshold)
                }
            }
        }

        return result
    }

    /// Convert decibels to power.
    ///
    /// - Parameters:
    ///   - S_db: Spectrogram in dB.
    ///   - ref: Reference power (default: 1.0).
    /// - Returns: Power spectrogram.
    public static func dbToPower(_ S_db: [[Float]], ref: Float = 1.0) -> [[Float]] {
        guard !S_db.isEmpty else { return [] }

        return S_db.map { row in
            row.map { db in
                ref * pow(10.0, db / 10.0)
            }
        }
    }

    /// Convert decibels to amplitude.
    ///
    /// - Parameters:
    ///   - S_db: Spectrogram in dB.
    ///   - ref: Reference amplitude (default: 1.0).
    /// - Returns: Amplitude spectrogram.
    public static func dbToAmplitude(_ S_db: [[Float]], ref: Float = 1.0) -> [[Float]] {
        guard !S_db.isEmpty else { return [] }

        return S_db.map { row in
            row.map { db in
                ref * pow(10.0, db / 20.0)
            }
        }
    }
}
