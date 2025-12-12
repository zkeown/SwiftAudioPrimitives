import Foundation

/// Padding mode for stack memory edges.
public enum StackPadMode: Sendable {
    /// Pad with constant value.
    case constant(Float)
    /// Pad with edge values.
    case edge
    /// Wrap around (circular).
    case wrap
}

/// Time-delay embedding for feature sequences.
///
/// Stacks multiple time-delayed copies of features to capture temporal context.
/// This is useful for models that need to see a window of past/future frames.
///
/// ## Example Usage
///
/// ```swift
/// // Stack 3 delayed copies (including current frame)
/// let stacked = StackMemory.stack(features, nSteps: 3, delay: 1)
/// // Output shape: (nFeatures * 3, nFrames)
///
/// // Stack with larger delay (skip frames)
/// let stacked = StackMemory.stack(features, nSteps: 5, delay: 2)
///
/// // Stack with zero padding at boundaries
/// let stacked = StackMemory.stack(features, nSteps: 3, delay: 1, mode: .constant(0))
/// ```
public struct StackMemory {
    private init() {}

    /// Stack time-delayed copies of features.
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - nSteps: Number of copies to stack (default: 2).
    ///   - delay: Frame delay between copies (default: 1).
    ///   - mode: Padding mode for boundary frames (default: constant(0)).
    /// - Returns: Stacked features of shape (nFeatures * nSteps, nFrames).
    public static func stack(
        _ features: [[Float]],
        nSteps: Int = 2,
        delay: Int = 1,
        mode: StackPadMode = .constant(0)
    ) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return [] }
        guard nSteps >= 1 else { return features }
        guard delay >= 0 else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count

        var stacked = [[Float]]()
        stacked.reserveCapacity(nFeatures * nSteps)

        // For each step (time offset)
        for step in 0..<nSteps {
            let timeOffset = step * delay

            // For each feature row
            for f in 0..<nFeatures {
                var row = [Float](repeating: 0, count: nFrames)

                for t in 0..<nFrames {
                    // Source time index (looking back in time)
                    let srcT = t - timeOffset
                    row[t] = getValueAt(features[f], index: srcT, count: nFrames, mode: mode)
                }

                stacked.append(row)
            }
        }

        return stacked
    }

    /// Stack features with both past and future context (centered).
    ///
    /// - Parameters:
    ///   - features: Feature matrix of shape (nFeatures, nFrames).
    ///   - contextSize: Number of frames on each side (total = 2*contextSize + 1).
    ///   - mode: Padding mode for boundary frames (default: edge).
    /// - Returns: Stacked features of shape (nFeatures * (2*contextSize + 1), nFrames).
    public static func stackCentered(
        _ features: [[Float]],
        contextSize: Int,
        mode: StackPadMode = .edge
    ) -> [[Float]] {
        guard !features.isEmpty, !features[0].isEmpty else { return [] }
        guard contextSize >= 0 else { return features }

        let nFeatures = features.count
        let nFrames = features[0].count
        let nSteps = 2 * contextSize + 1

        var stacked = [[Float]]()
        stacked.reserveCapacity(nFeatures * nSteps)

        // For each step (time offset from -contextSize to +contextSize)
        for step in -contextSize...contextSize {
            // For each feature row
            for f in 0..<nFeatures {
                var row = [Float](repeating: 0, count: nFrames)

                for t in 0..<nFrames {
                    let srcT = t + step
                    row[t] = getValueAt(features[f], index: srcT, count: nFrames, mode: mode)
                }

                stacked.append(row)
            }
        }

        return stacked
    }

    // MARK: - Private Implementation

    private static func getValueAt(
        _ array: [Float],
        index: Int,
        count: Int,
        mode: StackPadMode
    ) -> Float {
        if index >= 0 && index < count {
            return array[index]
        }

        switch mode {
        case .constant(let value):
            return value

        case .edge:
            if index < 0 {
                return array[0]
            } else {
                return array[count - 1]
            }

        case .wrap:
            var wrappedIndex = index % count
            if wrappedIndex < 0 {
                wrappedIndex += count
            }
            return array[wrappedIndex]
        }
    }
}
