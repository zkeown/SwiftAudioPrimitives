import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaAnalysis

final class TempoDebugTest: XCTestCase {
    func loadReferenceSignal(_ name: String) throws -> [Float] {
        let refPath = "/Users/zakkeown/Code/SwiftAudioPrimitives/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let signals = json["test_signals"] as? [String: Any],
              let signalData = signals[name] as? [String: Any],
              let signalValues = signalData["data"] as? [Double] else {
            throw NSError(domain: "", code: 1, userInfo: [NSLocalizedDescriptionKey: "Signal not found"])
        }
        return signalValues.map { Float($0) }
    }

    func testMultiToneTempo() async throws {
        // Load actual multi_tone from reference data
        let signal = try loadReferenceSignal("multi_tone")
        print("Signal length: \(signal.count)")

        let trackerConfig = BeatTrackerConfig(sampleRate: 22050, hopLength: 512)
        let tracker = BeatTracker(config: trackerConfig)

        let tempo = await tracker.estimateTempo(signal)
        print("Detected tempo: \(tempo) BPM")
        print("Expected: 0 (no tempo)")

        // Use the same onset detector that BeatTracker uses internally
        let onsetConfig = OnsetConfig(
            sampleRate: 22050,
            hopLength: 512,
            nFFT: 2048,
            method: .spectralFlux,
            peakThreshold: 0.3,
            preMax: 3,
            postMax: 3,
            preAvg: 3,
            postAvg: 3,
            delta: 0.05,
            wait: 10
        )
        let onsetDetector = OnsetDetector(config: onsetConfig)
        let onsetEnvelope = await onsetDetector.onsetStrength(signal)

        print("\nOnset envelope stats (BeatTracker's config):")
        print("  Length: \(onsetEnvelope.count)")
        print("  Max: \(onsetEnvelope.max() ?? 0)")
        print("  Min: \(onsetEnvelope.min() ?? 0)")

        // Count how many non-zero values
        let nonZero = onsetEnvelope.filter { $0 > 0.01 }.count
        print("  Non-zero (>0.01) count: \(nonZero)")

        // Show all envelope values
        print("\nAll envelope values:")
        for (i, val) in onsetEnvelope.enumerated() {
            print(String(format: "  Frame %2d: %.4f", i, val))
        }

        // Compute variance / std deviation of envelope (excluding initial onset)
        let steadyState = Array(onsetEnvelope.dropFirst(5))
        let mean = steadyState.reduce(0, +) / Float(steadyState.count)
        let variance = steadyState.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(steadyState.count)
        let stdDev = sqrt(variance)
        print("\nSteady-state stats (frames 5+):")
        print("  Mean: \(mean)")
        print("  Std Dev: \(stdDev)")
        print("  Coefficient of Variation: \(stdDev / max(mean, 0.0001))")

        // Check ratio of initial spike to steady-state
        let maxEnvelope = onsetEnvelope.max() ?? 0
        let spikeRatio = mean > 0 ? maxEnvelope / mean : 0
        print("  Initial spike / mean ratio: \(spikeRatio)")
    }
}
