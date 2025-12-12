import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaAnalysis

final class OnsetDebugTest: XCTestCase {
    func amModulated(carrierFreq: Float = 1000, modulatorFreq: Float = 5, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        let carrierD = Double(carrierFreq)
        let modulatorD = Double(modulatorFreq)
        let srD = Double(sampleRate)
        return (0..<nSamples).map { i in
            let t = Double(i) / srD
            let carrier = sin(2 * Double.pi * carrierD * t)
            let modulator = 0.5 * (1 + sin(2 * Double.pi * modulatorD * t))
            return Float(carrier * modulator)
        }
    }

    func multiTone(fundamentalFrequency: Float = 440, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        let f0 = Double(fundamentalFrequency)
        let srD = Double(sampleRate)
        return (0..<nSamples).map { i in
            let t = Double(i) / srD
            let val = 0.5 * sin(2 * Double.pi * f0 * t) +
                      0.3 * sin(2 * Double.pi * f0 * 2 * t) +
                      0.2 * sin(2 * Double.pi * f0 * 4 * t)
            return Float(val)
        }
    }

    func testMultiToneOnset() async throws {
        // Expected: only 1 onset at frame 3
        let expectedFrames = [3]

        let signal = multiTone()
        print("Signal length: \(signal.count)")
        print("Expected onset frames: \(expectedFrames)")

        let config = OnsetConfig(sampleRate: 22050, hopLength: 512)
        let detector = OnsetDetector(config: config)

        let envelope = await detector.onsetStrength(signal)
        print("\nEnvelope length: \(envelope.count)")
        print("Max value: \(envelope.max() ?? 0)")
        print("Min value: \(envelope.min() ?? 0)")

        // Show first 15 envelope values
        print("\nEnvelope (first 15 frames):")
        for (i, val) in envelope.prefix(15).enumerated() {
            print(String(format: "  Frame %2d: %.4f", i, val))
        }

        let detected = await detector.detect(signal)
        print("\nDetected onset frames: \(detected)")
        print("Detected count: \(detected.count)")
        print("Expected onset frames: \(expectedFrames)")

        // F1 calculation
        let tolerance = 3
        var truePositives = 0
        for expected in expectedFrames {
            if detected.contains(where: { abs($0 - expected) <= tolerance }) {
                truePositives += 1
            }
        }
        let precision = detected.isEmpty ? 0 : Float(truePositives) / Float(detected.count)
        let recall = expectedFrames.isEmpty ? 1 : Float(truePositives) / Float(expectedFrames.count)
        let f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
        print("\nPrecision: \(precision), Recall: \(recall), F1: \(f1)")
    }

    func testOnsetEnvelope() async throws {
        // Expected frames from librosa
        let expectedFrames = [5, 10, 14, 18, 23, 27, 31, 35, 40]

        let signal = amModulated(
            carrierFreq: 1000,
            modulatorFreq: 5,
            sampleRate: 22050,
            duration: 1.0
        )
        print("Signal length: \(signal.count)")
        print("Expected onset frames: \(expectedFrames)")

        let config = OnsetConfig(sampleRate: 22050, hopLength: 512)
        let detector = OnsetDetector(config: config)

        let envelope = await detector.onsetStrength(signal)
        print("\nEnvelope length: \(envelope.count)")
        print("Max value: \(envelope.max() ?? 0)")
        print("Min value: \(envelope.min() ?? 0)")

        print("\nEnvelope values around expected peaks:")
        for frame in expectedFrames {
            let start = max(0, frame - 2)
            let end = min(envelope.count, frame + 3)
            var values: [String] = []
            for i in start..<end {
                let marker = i == frame ? "*" : ""
                values.append(String(format: "%.4f%@", envelope[i], marker))
            }
            print("Frame \(frame): \(values.joined(separator: ", "))")
        }

        let detected = await detector.detect(signal)
        print("\nDetected onset frames: \(detected)")
        print("Expected onset frames: \(expectedFrames)")

        let tolerance = 3
        var matches = 0
        for expected in expectedFrames {
            if detected.contains(where: { abs($0 - expected) <= tolerance }) {
                matches += 1
            }
        }
        print("Matches within tolerance \(tolerance): \(matches)/\(expectedFrames.count)")
    }
}
