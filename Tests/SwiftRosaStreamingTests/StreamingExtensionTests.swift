import XCTest
@testable import SwiftRosaStreaming
import SwiftRosaCore

final class StreamingExtensionTests: XCTestCase {

    // MARK: - Test Signals

    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    // MARK: - StreamingMel Tests

    func testStreamingMelBasic() async {
        let streaming = await StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)

        await streaming.push(signal)
        let frames = await streaming.popFrames()

        XCTAssertGreaterThan(frames.count, 0, "Should produce frames")

        for frame in frames {
            XCTAssertEqual(frame.count, 80, "Each frame should have 80 mel bins")
        }
    }

    func testStreamingMelMatrix() async {
        let streaming = await StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)

        await streaming.push(signal)
        let matrix = await streaming.popMatrix()

        XCTAssertEqual(matrix.count, 80, "Matrix should have 80 mel bins")
        if !matrix.isEmpty {
            XCTAssertGreaterThan(matrix[0].count, 0, "Should have time frames")
        }
    }

    func testStreamingMelIncremental() async {
        let streaming = await StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80
        )

        let chunkSize = 160  // One hop worth
        let totalSamples = 1600  // 10 chunks
        var totalFrames = 0

        for i in stride(from: 0, to: totalSamples, by: chunkSize) {
            let chunk = generateSineWave(frequency: 440, sampleRate: 16000, duration: Float(chunkSize) / 16000)
            await streaming.push(chunk)
            let frames = await streaming.popFrames()
            totalFrames += frames.count
        }

        // Flush remaining
        let remaining = await streaming.flush()
        totalFrames += remaining.count

        XCTAssertGreaterThan(totalFrames, 0, "Should have produced frames incrementally")
    }

    func testStreamingMelFlush() async {
        let streaming = await StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80
        )

        // Push less than one frame worth
        let shortSignal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.01)
        await streaming.push(shortSignal)

        // Should have frames in buffer but not enough for popFrames
        let beforeFlush = await streaming.popFrames()
        let afterFlush = await streaming.flush()

        XCTAssertEqual(beforeFlush.count, 0, "Not enough samples for a full frame")
        XCTAssertGreaterThan(afterFlush.count, 0, "Flush should produce at least one frame")
    }

    func testStreamingMelReset() async {
        let streaming = await StreamingMel(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 80
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)
        await streaming.push(signal)

        await streaming.reset()

        let buffered = await streaming.bufferedSamples
        XCTAssertEqual(buffered, 0, "Buffer should be empty after reset")
    }

    func testStreamingMelPresets() async {
        let speech = await StreamingMel.speechRecognition()
        let music = await StreamingMel.musicAnalysis()
        let whisper = await StreamingMel.whisperCompatible()

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.1)

        await speech.push(signal)
        await music.push(signal)
        await whisper.push(signal)

        XCTAssert(true)
    }

    // MARK: - StreamingMFCC Tests

    func testStreamingMFCCBasic() async {
        let streaming = await StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 40,
            nMFCC: 13
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)

        await streaming.push(signal)
        let frames = await streaming.popFrames()

        XCTAssertGreaterThan(frames.count, 0)

        for frame in frames {
            XCTAssertEqual(frame.count, 13, "Each frame should have 13 MFCC coefficients")
        }
    }

    func testStreamingMFCCMatrix() async {
        let streaming = await StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 40,
            nMFCC: 13
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)

        await streaming.push(signal)
        let matrix = await streaming.popMatrix()

        XCTAssertEqual(matrix.count, 13, "Matrix should have 13 MFCC coefficients")
    }

    func testStreamingMFCCWithDeltas() async {
        let streaming = await StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 40,
            nMFCC: 13
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 1.0)

        await streaming.push(signal)
        let (mfcc, delta, deltaDelta) = await streaming.popFramesWithDeltas()

        XCTAssertEqual(mfcc.count, 13)
        XCTAssertEqual(delta.count, 13)
        XCTAssertEqual(deltaDelta.count, 13)

        if !mfcc.isEmpty && !delta.isEmpty {
            XCTAssertEqual(mfcc[0].count, delta[0].count)
        }
    }

    func testStreamingMFCCStacked() async {
        let streaming = await StreamingMFCC(
            sampleRate: 16000,
            nFFT: 512,
            hopLength: 160,
            nMels: 40,
            nMFCC: 13
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 1.0)

        await streaming.push(signal)
        let stacked = await streaming.popFramesStacked()

        // Stacked should have 13 * 3 = 39 features
        XCTAssertEqual(stacked.count, 39, "Stacked features should have MFCC + delta + delta-delta")
    }

    func testStreamingMFCCPresets() async {
        let speech = await StreamingMFCC.speechRecognition()
        let music = await StreamingMFCC.musicAnalysis()
        let kaldi = await StreamingMFCC.kaldiCompatible()

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.1)

        await speech.push(signal)
        await music.push(signal)
        await kaldi.push(signal)

        XCTAssert(true)
    }

    // MARK: - StreamingPitchDetector Tests

    func testStreamingPitchDetectorBasic() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            fMin: 65,
            fMax: 2000
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        await streaming.push(signal)
        let pitches = await streaming.popPitches()

        XCTAssertGreaterThan(pitches.count, 0)

        // Some pitches should be detected (non-zero)
        let detectedPitches = pitches.filter { $0 > 0 }
        XCTAssertGreaterThan(detectedPitches.count, 0, "Should detect some pitches")

        // Average detected pitch should be near 440 Hz
        if !detectedPitches.isEmpty {
            let avgPitch = detectedPitches.reduce(0, +) / Float(detectedPitches.count)
            XCTAssertEqual(avgPitch, 440, accuracy: 30, "Average pitch should be near 440 Hz")
        }
    }

    func testStreamingPitchDetectorWithConfidence() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        await streaming.push(signal)
        let result = await streaming.popPitchesWithConfidence()

        XCTAssertEqual(result.pitches.count, result.confidences.count)

        // Check confidences are in valid range
        for confidence in result.confidences {
            XCTAssertGreaterThanOrEqual(confidence, 0)
            XCTAssertLessThanOrEqual(confidence, 1)
        }
    }

    func testStreamingPitchDetectorSmoothed() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512,
            medianFilterSize: 5
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        await streaming.push(signal)
        let smoothed = await streaming.popPitchesSmoothed()

        XCTAssertGreaterThan(smoothed.count, 0)
    }

    func testStreamingPitchDetectorWithNotes() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)

        await streaming.push(signal)
        let notes = await streaming.popPitchesWithNotes()

        XCTAssertGreaterThan(notes.count, 0)

        // Check structure
        for (pitch, midiNote, noteName, confidence) in notes {
            if pitch > 0 {
                XCTAssertGreaterThan(midiNote, 0)
                XCTAssertNotEqual(noteName, "-")
                XCTAssertGreaterThan(confidence, 0)
            }
        }
    }

    func testStreamingPitchDetectorCurrentPitch() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)
        await streaming.push(signal)

        let currentPitch = await streaming.currentPitch()

        if let pitch = currentPitch {
            XCTAssertEqual(pitch, 440, accuracy: 30)
        }
    }

    func testStreamingPitchDetectorCurrentNote() async {
        let streaming = await StreamingPitchDetector(
            sampleRate: 22050,
            frameLength: 2048,
            hopLength: 512
        )

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)
        await streaming.push(signal)

        let currentNote = await streaming.currentNote()

        if let (pitch, noteName) = currentNote {
            XCTAssertGreaterThan(pitch, 0)
            XCTAssertEqual(noteName, "A4", "440 Hz should be A4")
        }
    }

    func testStreamingPitchDetectorPresets() async {
        let voice = await StreamingPitchDetector.voice()
        let music = await StreamingPitchDetector.music()
        let singing = await StreamingPitchDetector.singing()
        let tuner = await StreamingPitchDetector.tuner()

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.2)

        await voice.push(signal)
        await music.push(signal)
        await singing.push(signal)
        await tuner.push(signal)

        XCTAssert(true)
    }

    // MARK: - Statistics Tests

    func testStreamingMelStats() async {
        let streaming = await StreamingMel(sampleRate: 16000, nFFT: 512, hopLength: 160, nMels: 80)

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)
        await streaming.push(signal)
        let _ = await streaming.popFrames()

        let stats = await streaming.stats

        XCTAssertGreaterThan(stats.totalSamplesReceived, 0)
        XCTAssertGreaterThan(stats.totalFramesOutput, 0)
    }

    func testStreamingMFCCStats() async {
        let streaming = await StreamingMFCC(sampleRate: 16000, nFFT: 512, hopLength: 160, nMels: 40, nMFCC: 13)

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)
        await streaming.push(signal)
        let _ = await streaming.popFrames()

        let stats = await streaming.stats

        XCTAssertGreaterThan(stats.totalSamplesReceived, 0)
        XCTAssertGreaterThan(stats.totalFramesOutput, 0)
    }

    func testStreamingPitchStats() async {
        let streaming = await StreamingPitchDetector(sampleRate: 22050, frameLength: 2048, hopLength: 512)

        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 0.5)
        await streaming.push(signal)
        let _ = await streaming.popPitches()

        let stats = await streaming.stats

        XCTAssertGreaterThan(stats.totalSamplesReceived, 0)
    }

    // MARK: - Edge Cases

    func testEmptyInput() async {
        let streaming = await StreamingMel(sampleRate: 16000, nFFT: 512, hopLength: 160, nMels: 80)

        await streaming.push([])
        let frames = await streaming.popFrames()

        XCTAssertEqual(frames.count, 0)
    }

    func testProcessComplete() async {
        let streaming = await StreamingMel(sampleRate: 16000, nFFT: 512, hopLength: 160, nMels: 80)

        let signal = generateSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)
        let frames = await streaming.processComplete(signal)

        XCTAssertGreaterThan(frames.count, 0)
    }
}
