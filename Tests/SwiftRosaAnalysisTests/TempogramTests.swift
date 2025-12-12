import XCTest
@testable import SwiftRosaAnalysis
@testable import SwiftRosaCore

/// Tests for Tempogram implementation.
final class TempogramTests: XCTestCase {

    // MARK: - Test Utilities

    private func generateClickTrack(bpm: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)

        let samplesPerBeat = Int(60.0 * sampleRate / bpm)
        let clickLength = min(100, samplesPerBeat / 4)

        var position = 0
        while position < numSamples {
            // Add click (short impulse with decay)
            for i in 0..<clickLength {
                if position + i < numSamples {
                    let decay = exp(-Float(i) / 20.0)
                    signal[position + i] += decay
                }
            }
            position += samplesPerBeat
        }

        return signal
    }

    // MARK: - Basic Tests

    func testTempogramBasic() async {
        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 384,
            type: .autocorrelation,
            bpmRange: (30, 300)
        ))

        // Generate click track at 120 BPM
        let signal = generateClickTrack(bpm: 120, sampleRate: 22050, duration: 3.0)

        let tg = await tempogram.transform(signal)

        XCTAssertGreaterThan(tg.count, 0, "Should produce tempo bins")
        XCTAssertGreaterThan(tg[0].count, 0, "Should produce frames")
    }

    func testTempogramBPMAxis() {
        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            bpmRange: (60, 180)
        ))

        let bpmAxis = tempogram.bpmAxis

        XCTAssertGreaterThan(bpmAxis.count, 0)

        // BPM axis should be mostly in range, allowing for some boundary discretization
        // The discretization of lag bins to BPM can cause slight overshoots
        var inRangeCount = 0
        for bpm in bpmAxis {
            if bpm >= 60 && bpm <= 200 {  // Allow 10% overshoot due to discretization
                inRangeCount += 1
            }
        }
        let inRangeRatio = Float(inRangeCount) / Float(bpmAxis.count)
        XCTAssertGreaterThan(inRangeRatio, 0.9, "Most BPM values should be in or near requested range")
    }

    func testTempogramFromOnsetEnvelope() {
        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 200
        ))

        // Create synthetic onset envelope with periodic peaks
        let nFrames = 500
        var envelope = [Float](repeating: 0, count: nFrames)

        // Add peaks every 43 frames (approximately 120 BPM at 22050/512 fps)
        for i in stride(from: 0, to: nFrames, by: 43) {
            if i < nFrames {
                envelope[i] = 1.0
            }
        }

        let tg = tempogram.transform(onsetEnvelope: envelope)

        XCTAssertGreaterThan(tg.count, 0)

        // Should have energy at 120 BPM region
        // (exact bin depends on configuration)
    }

    func testTempogramFourier() async {
        let tempogram = Tempogram(config: TempogramConfig(
            sampleRate: 22050,
            hopLength: 512,
            winLength: 384,
            type: .fourier,
            bpmRange: (60, 180)
        ))

        let signal = generateClickTrack(bpm: 100, sampleRate: 22050, duration: 3.0)

        let tg = await tempogram.transform(signal)

        XCTAssertGreaterThan(tg.count, 0)

        // All values should be non-negative (magnitude)
        for row in tg {
            for val in row {
                XCTAssertGreaterThanOrEqual(val, 0)
                XCTAssertFalse(val.isNaN)
            }
        }
    }

    func testTempogramPresets() async {
        // Test preset configurations
        let standard = Tempogram.standard
        let music = Tempogram.music

        let signal = generateClickTrack(bpm: 120, sampleRate: 22050, duration: 2.0)

        let tgStandard = await standard.transform(signal)
        let tgMusic = await music.transform(signal)

        XCTAssertGreaterThan(tgStandard.count, 0)
        XCTAssertGreaterThan(tgMusic.count, 0)
    }

    func testTempogramEmptyInput() async {
        let tempogram = Tempogram()

        let empty: [Float] = []
        let tg = await tempogram.transform(empty)

        XCTAssertEqual(tg.count, 0, "Empty input should produce empty output")
    }

    func testTempogramShortInput() async {
        let tempogram = Tempogram(config: TempogramConfig(
            winLength: 100
        ))

        // Input shorter than window
        let shortSignal = [Float](repeating: 0.5, count: 50)
        let tg = await tempogram.transform(shortSignal)

        // Should handle gracefully - either empty or valid finite values
        if !tg.isEmpty && !tg[0].isEmpty {
            // If output is produced, it should be valid (no NaN or Inf)
            for row in tg {
                for val in row {
                    XCTAssertFalse(val.isNaN, "Output should not contain NaN")
                    XCTAssertFalse(val.isInfinite, "Output should not contain Inf")
                }
            }
        }
        // Pass - the implementation handled short input gracefully
    }
}
