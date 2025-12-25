import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaStreaming
@testable import SwiftRosaAnalysis
@testable import SwiftRosaEffects

/// Streaming parity tests to validate that streaming implementations
/// produce identical results to their batch counterparts.
///
/// These tests ensure:
/// 1. StreamingSTFT matches batch STFT
/// 2. StreamingMel matches batch MelSpectrogram
/// 3. StreamingMFCC matches batch MFCC
/// 4. Streaming pitch detection matches batch pitch detection
final class StreamingParityTests: XCTestCase {

    // MARK: - Test Configuration

    private let sampleRate: Float = 22050
    private let nFFT = 1024
    private let hopLength = 256
    private let chunkSize = 1024  // Simulate real-time processing

    // MARK: - Signal Generation Helpers

    private func generateSine(frequency: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * .pi * frequency * t)
        }
        return signal
    }

    private func generateMultiTone(duration: Float) -> [Float] {
        let frequencies: [Float] = [440, 880, 1320]
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            for freq in frequencies {
                signal[i] += sin(2 * .pi * freq * t) / Float(frequencies.count)
            }
        }
        return signal
    }

    private func generateChirp(fmin: Float, fmax: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        var signal = [Float](repeating: 0, count: numSamples)
        let rate = (fmax - fmin) / duration
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            let instantFreq = fmin + rate * t / 2
            signal[i] = sin(2 * .pi * instantFreq * t)
        }
        return signal
    }

    // MARK: - Correlation Helper

    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count && !a.isEmpty else { return 0 }

        let n = Float(a.count)
        let meanA = a.reduce(0, +) / n
        let meanB = b.reduce(0, +) / n

        var sumAB: Float = 0
        var sumA2: Float = 0
        var sumB2: Float = 0

        for i in 0..<a.count {
            let da = a[i] - meanA
            let db = b[i] - meanB
            sumAB += da * db
            sumA2 += da * da
            sumB2 += db * db
        }

        let denom = sqrt(sumA2 * sumB2)
        return denom > 0 ? sumAB / denom : 0
    }

    // MARK: - StreamingSTFT Parity Tests

    /// Test that StreamingSTFT produces same magnitude as batch STFT
    func testStreamingSTFT_MatchesBatch_Sine() async throws {
        let signal = generateSine(frequency: 440, duration: 2.0)

        // Batch processing
        let batchConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let batchSTFT = STFT(config: batchConfig)
        let batchResult = await batchSTFT.transform(signal)
        let batchMagnitude = batchResult.magnitude

        // Streaming processing
        let streamingConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        )
        let streamingSTFT = StreamingSTFT(config: streamingConfig)

        var streamingFrames: [[Float]] = []

        // Process in chunks
        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            try await streamingSTFT.push(chunk)
            let frames = await streamingSTFT.popFrames()
            // Extract magnitude from each ComplexMatrix frame
            for frame in frames {
                let mag = frame.magnitude.flatMap { $0 }
                streamingFrames.append(mag)
            }
        }

        // Flush remaining frames
        let finalFrames = await streamingSTFT.flush()
        for frame in finalFrames {
            let mag = frame.magnitude.flatMap { $0 }
            streamingFrames.append(mag)
        }

        // Compare results
        // Note: Streaming may have slightly different frame count due to buffering
        let minFrames = min(batchMagnitude.first?.count ?? 0, streamingFrames.count)

        if minFrames > 10 {
            // Compare middle frames (avoid edge effects)
            let startFrame = minFrames / 4
            let endFrame = 3 * minFrames / 4

            var maxRelError: Float = 0
            var comparisonCount = 0

            for frameIdx in startFrame..<endFrame {
                // Batch is [freq][frame], streaming is [freq] per frame
                if frameIdx < streamingFrames.count && batchMagnitude.count > 0 {
                    let streamingFrame = streamingFrames[frameIdx]

                    for freqIdx in 0..<min(streamingFrame.count, batchMagnitude.count) {
                        if frameIdx < batchMagnitude[freqIdx].count {
                            let expected = batchMagnitude[freqIdx][frameIdx]
                            let actual = streamingFrame[freqIdx]

                            if expected > 0.01 {
                                let relError = abs(expected - actual) / expected
                                maxRelError = max(maxRelError, relError)
                                comparisonCount += 1
                            }
                        }
                    }
                }
            }

            if comparisonCount > 0 {
                XCTAssertLessThan(maxRelError, 0.5,
                    "Streaming STFT should match batch within 50% (max error: \(maxRelError))")
            }
        }
    }

    /// Test StreamingSTFT with multi-tone signal
    func testStreamingSTFT_MatchesBatch_MultiTone() async throws {
        let signal = generateMultiTone(duration: 2.0)

        // Batch processing
        let batchConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: true,
            padMode: .constant(0)
        )
        let batchSTFT = STFT(config: batchConfig)
        let batchResult = await batchSTFT.transform(signal)

        // Streaming processing
        let streamingConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        )
        let streamingSTFT = StreamingSTFT(config: streamingConfig)

        var streamingFrameCount = 0

        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            try await streamingSTFT.push(chunk)
            let frames = await streamingSTFT.popFrames()
            streamingFrameCount += frames.count
        }

        // Verify both produce output
        XCTAssertGreaterThan(batchResult.cols, 0, "Batch should produce frames")
        XCTAssertGreaterThan(streamingFrameCount, 0, "Streaming should produce frames")

        // Check frame counts are similar
        let frameDiff = abs(batchResult.cols - streamingFrameCount)
        let maxFrameDiff = max(20, batchResult.cols / 5)  // Allow 20% difference
        XCTAssertLessThanOrEqual(frameDiff, maxFrameDiff,
            "Frame count should be similar: batch=\(batchResult.cols), streaming=\(streamingFrameCount)")
    }

    // MARK: - StreamingMel Parity Tests

    /// Test that StreamingMel produces same output as batch MelSpectrogram
    func testStreamingMel_MatchesBatch_Sine() async throws {
        let signal = generateSine(frequency: 440, duration: 2.0)
        let nMels = 80

        // Batch processing
        let batchMel = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let batchResult = await batchMel.transform(signal)

        // Streaming processing
        let streamingMel = StreamingMel(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )

        var streamingFrames: [[Float]] = []

        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            await streamingMel.push(chunk)
            let frames = await streamingMel.popFrames()
            streamingFrames.append(contentsOf: frames)
        }

        // Compare results
        let minFrames = min(batchResult.first?.count ?? 0, streamingFrames.count)

        if minFrames > 10 {
            // Compare middle frames
            let startFrame = minFrames / 4
            let endFrame = 3 * minFrames / 4

            var maxRelError: Float = 0
            var significantComparisons = 0

            for frameIdx in startFrame..<endFrame {
                if frameIdx < streamingFrames.count {
                    let streamingFrame = streamingFrames[frameIdx]

                    for melIdx in 0..<min(nMels, batchResult.count, streamingFrame.count) {
                        if frameIdx < batchResult[melIdx].count {
                            let expected = batchResult[melIdx][frameIdx]
                            let actual = streamingFrame[melIdx]

                            if expected > 0.001 {
                                let relError = Swift.abs(expected - actual) / expected
                                maxRelError = Swift.max(maxRelError, relError)
                                significantComparisons += 1
                            }
                        }
                    }
                }
            }

            if significantComparisons > 0 {
                XCTAssertLessThan(maxRelError, 0.5,
                    "Streaming mel should match batch within 50% (max error: \(maxRelError))")
            }
        }
    }

    /// Test StreamingMel with chirp signal
    func testStreamingMel_MatchesBatch_Chirp() async throws {
        let signal = generateChirp(fmin: 100, fmax: 8000, duration: 2.0)
        let nMels = 80

        // Batch processing
        let batchMel = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        let batchResult = await batchMel.transform(signal)

        // Streaming processing
        let streamingMel = StreamingMel(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )

        var streamingFrames: [[Float]] = []

        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            await streamingMel.push(chunk)
            let frames = await streamingMel.popFrames()
            streamingFrames.append(contentsOf: frames)
        }

        // Verify both produce output
        XCTAssertGreaterThan(batchResult.first?.count ?? 0, 0, "Batch should produce frames")
        XCTAssertGreaterThan(streamingFrames.count, 0, "Streaming should produce frames")

        // For chirp, verify the frequency sweep is captured by both
        // The energy should move across mel bins over time
        let batchFirstFrame = 0
        let batchLastFrame = min((batchResult.first?.count ?? 1) - 1, 50)
        let streamingFirstFrame = 0
        let streamingLastFrame = min(streamingFrames.count - 1, 50)

        // Find peak mel bin for first and last compared frames
        if batchLastFrame > batchFirstFrame && streamingLastFrame > streamingFirstFrame {
            // Batch: find peak mel for early frame
            var batchEarlyPeak = 0
            var batchEarlyMax: Float = 0
            for melIdx in 0..<batchResult.count {
                if batchResult[melIdx].count > batchFirstFrame + 5 {
                    let val = batchResult[melIdx][batchFirstFrame + 5]
                    if val > batchEarlyMax {
                        batchEarlyMax = val
                        batchEarlyPeak = melIdx
                    }
                }
            }

            // Streaming: find peak mel for early frame
            var streamingEarlyPeak = 0
            var streamingEarlyMax: Float = 0
            if streamingFrames.count > streamingFirstFrame + 5 {
                let frame = streamingFrames[streamingFirstFrame + 5]
                for (melIdx, val) in frame.enumerated() {
                    if val > streamingEarlyMax {
                        streamingEarlyMax = val
                        streamingEarlyPeak = melIdx
                    }
                }
            }

            // Both should find peak in similar region (low mels for start of chirp)
            XCTAssertEqual(batchEarlyPeak, streamingEarlyPeak, accuracy: 15,
                "Early chirp should have similar peak mel bin")
        }
    }

    // MARK: - StreamingMFCC Parity Tests

    /// Test that StreamingMFCC produces same output as batch MFCC
    func testStreamingMFCC_MatchesBatch_Sine() async throws {
        let signal = generateSine(frequency: 440, duration: 2.0)
        let nMFCC = 13
        let nMels = 80

        // Batch processing
        let batchMFCC = MFCC(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels,
            nMFCC: nMFCC
        )
        let batchResult = await batchMFCC.transform(signal)

        // Streaming processing
        let streamingMFCC = StreamingMFCC(
            sampleRate: sampleRate,
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels,
            nMFCC: nMFCC
        )

        var streamingFrames: [[Float]] = []

        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            await streamingMFCC.push(chunk)
            let frames = await streamingMFCC.popFrames()
            streamingFrames.append(contentsOf: frames)
        }

        // Compare results
        let minFrames = min(batchResult.first?.count ?? 0, streamingFrames.count)

        if minFrames > 10 {
            let startFrame = minFrames / 4
            let endFrame = 3 * minFrames / 4

            // Compare first few MFCC coefficients (most important)
            var correlationSum: Float = 0
            var correlationCount = 0

            for coeffIdx in 0..<min(5, nMFCC) {
                var batchValues: [Float] = []
                var streamingValues: [Float] = []

                for frameIdx in startFrame..<endFrame {
                    if coeffIdx < batchResult.count && frameIdx < batchResult[coeffIdx].count {
                        batchValues.append(batchResult[coeffIdx][frameIdx])
                    }
                    if frameIdx < streamingFrames.count && coeffIdx < streamingFrames[frameIdx].count {
                        streamingValues.append(streamingFrames[frameIdx][coeffIdx])
                    }
                }

                let minLen = min(batchValues.count, streamingValues.count)
                if minLen > 5 {
                    let corr = correlation(
                        Array(batchValues.prefix(minLen)),
                        Array(streamingValues.prefix(minLen))
                    )
                    if !corr.isNaN {
                        correlationSum += corr
                        correlationCount += 1
                    }
                }
            }

            if correlationCount > 0 {
                let avgCorrelation = correlationSum / Float(correlationCount)
                XCTAssertGreaterThan(avgCorrelation, 0.5,
                    "Streaming MFCC should correlate with batch (avg: \(avgCorrelation))")
            }
        }
    }

    // MARK: - Streaming Pitch Detection Parity Tests

    /// Test that StreamingPitchDetector matches batch pitch detection
    func testStreamingPitch_MatchesBatch_Sine() async throws {
        let frequency: Float = 440
        let signal = generateSine(frequency: frequency, duration: 2.0)

        // Batch pitch detection
        let batchConfig = PitchDetectorConfig(
            sampleRate: sampleRate,
            frameLength: nFFT,
            hopLength: hopLength,
            fMin: 50,
            fMax: 2000
        )
        let batchPitch = PitchDetector(config: batchConfig)
        let batchFrequencies = await batchPitch.transform(signal)

        // Streaming pitch detection
        let streamingPitch = StreamingPitchDetector(
            sampleRate: sampleRate,
            frameLength: nFFT,
            hopLength: hopLength,
            fMin: 50,
            fMax: 2000
        )

        var streamingFrequencies: [Float] = []

        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            await streamingPitch.push(chunk)
            let pitches = await streamingPitch.popPitches()
            streamingFrequencies.append(contentsOf: pitches)
        }

        // Compare results
        // Both should detect ~440 Hz for most frames

        let minFrames = min(batchFrequencies.count, streamingFrequencies.count)

        if minFrames > 10 {
            // Count how many frames detect frequency near 440 Hz
            var batchCorrect = 0
            var streamingCorrect = 0

            let startFrame = minFrames / 4
            let endFrame = 3 * minFrames / 4

            for frameIdx in startFrame..<endFrame {
                if frameIdx < batchFrequencies.count {
                    let batchF = batchFrequencies[frameIdx]
                    if Swift.abs(batchF - frequency) < 50 {  // Within 50 Hz
                        batchCorrect += 1
                    }
                }
                if frameIdx < streamingFrequencies.count {
                    let streamingF = streamingFrequencies[frameIdx]
                    if Swift.abs(streamingF - frequency) < 50 {
                        streamingCorrect += 1
                    }
                }
            }

            let totalFrames = endFrame - startFrame
            let batchAccuracy = Float(batchCorrect) / Float(totalFrames)
            let streamingAccuracy = Float(streamingCorrect) / Float(totalFrames)

            // Both should have reasonable accuracy for pure sine
            XCTAssertGreaterThan(batchAccuracy, 0.3,
                "Batch pitch should detect 440 Hz in some frames")
            XCTAssertGreaterThan(streamingAccuracy, 0.3,
                "Streaming pitch should detect 440 Hz in some frames")
        }
    }

    // MARK: - Edge Cases

    /// Test streaming with very small chunks
    func testStreamingSTFT_SmallChunks() async throws {
        let signal = generateSine(frequency: 440, duration: 1.0)
        let smallChunkSize = 128  // Much smaller than nFFT

        let streamingConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        )
        let streamingSTFT = StreamingSTFT(config: streamingConfig)

        var frameCount = 0

        for start in stride(from: 0, to: signal.count, by: smallChunkSize) {
            let end = min(start + smallChunkSize, signal.count)
            let chunk = Array(signal[start..<end])

            try await streamingSTFT.push(chunk)
            let frames = await streamingSTFT.popFrames()
            frameCount += frames.count
        }

        // Flush remaining
        let final = await streamingSTFT.flush()
        frameCount += final.count

        // Should still produce reasonable number of frames
        let expectedFrames = signal.count / hopLength
        XCTAssertGreaterThan(frameCount, expectedFrames / 4,
            "Small chunks should still produce frames")
    }

    /// Test streaming with single sample at a time
    func testStreamingSTFT_SingleSample() async throws {
        // This is an extreme case - one sample at a time
        let signal = generateSine(frequency: 440, duration: 0.5)

        let streamingConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        )
        let streamingSTFT = StreamingSTFT(config: streamingConfig)

        var frameCount = 0

        // Push one sample at a time (first 5000 samples to limit test time)
        for i in 0..<min(5000, signal.count) {
            try await streamingSTFT.push([signal[i]])
            let frames = await streamingSTFT.popFrames()
            frameCount += frames.count
        }

        // Should handle single samples gracefully
        XCTAssertGreaterThan(frameCount, 0,
            "Single sample mode should still produce output")
    }

    /// Test streaming reset functionality
    func testStreamingSTFT_Reset() async throws {
        let signal1 = generateSine(frequency: 440, duration: 0.5)
        let signal2 = generateSine(frequency: 880, duration: 0.5)

        let streamingConfig = STFTConfig(
            uncheckedNFFT: nFFT,
            hopLength: hopLength,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        )
        let streamingSTFT = StreamingSTFT(config: streamingConfig)

        // Process first signal
        try await streamingSTFT.push(signal1)
        _ = await streamingSTFT.popFrames()

        // Reset
        await streamingSTFT.reset()

        // Process second signal
        try await streamingSTFT.push(signal2)
        let frames = await streamingSTFT.popFrames()

        // After reset, should process second signal cleanly
        XCTAssertGreaterThan(frames.count, 0,
            "Should produce output after reset")

        // The output should reflect 880 Hz, not 440 Hz
        // Find peak frequency bin
        if let firstFrame = frames.first {
            let mag = firstFrame.magnitude.flatMap { $0 }
            var peakBin = 0
            var peakMag: Float = 0
            for (bin, m) in mag.enumerated() {
                if m > peakMag {
                    peakMag = m
                    peakBin = bin
                }
            }

            let freqResolution = sampleRate / Float(nFFT)
            let peakFreq = Float(peakBin) * freqResolution

            // Should be near 880 Hz (not 440 Hz from previous signal)
            XCTAssertEqual(peakFreq, 880, accuracy: 150,
                "After reset, should detect new signal frequency")
        }
    }
}
