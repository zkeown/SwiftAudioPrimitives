import XCTest
@testable import SwiftRosaStreaming
import SwiftRosaCore

final class StreamingTests: XCTestCase {

    // MARK: - Accuracy Thresholds
    //
    // For COLA-compliant streaming (Hann with 75% overlap):
    // - Round-trip error: < 1e-4 (matching batch STFT/ISTFT)
    // - Streaming vs batch difference: < 1e-5 (should be identical)
    //
    // For non-centered mode (edge effects expected):
    // - Round-trip error in valid region: < 1e-3
    private let colaRoundTripError: Float = 1e-4
    private let streamingBatchTolerance: Float = 1e-5
    private let nonCenteredTolerance: Float = 1e-3

    // MARK: - StreamingSTFT Tests

    func testStreamingSTFTBasic() async {
        let config = STFTConfig(
            nFFT: 256,
            hopLength: 64,
            windowType: .hann,
            center: false
        )

        let streaming = StreamingSTFT(config: config)

        // Generate test signal
        var signal = [Float](repeating: 0, count: 512)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        // Push samples
        await streaming.push(signal)

        // Should have frames available
        let frames = await streaming.popFrames()
        XCTAssertGreaterThan(frames.count, 0, "Should produce frames")

        // Each frame should have correct dimensions
        for frame in frames {
            XCTAssertEqual(frame.rows, config.nFreqs)
            XCTAssertEqual(frame.cols, 1)
        }
    }

    func testStreamingSTFTIncremental() async {
        let config = STFTConfig(
            nFFT: 256,
            hopLength: 64,
            windowType: .hann,
            center: false
        )

        let streaming = StreamingSTFT(config: config)

        // Push samples in chunks
        let chunkSize = 64
        var totalFrames = 0

        for _ in 0..<10 {
            var chunk = [Float](repeating: 0, count: chunkSize)
            for i in 0..<chunkSize {
                chunk[i] = Float.random(in: -1...1)
            }

            await streaming.push(chunk)
            let frames = await streaming.popFrames()
            totalFrames += frames.count
        }

        // Flush remaining
        let remaining = await streaming.flush()
        totalFrames += remaining.count

        XCTAssertGreaterThan(totalFrames, 0, "Should produce frames from incremental input")
    }

    func testStreamingSTFTEquivalenceToBatch() async {
        // Streaming STFT should produce IDENTICAL results to batch STFT (without centering)
        // Note: streaming flush() may produce one extra zero-padded frame
        let config = STFTConfig(
            nFFT: 256,
            hopLength: 64,
            windowType: .hann,
            center: false
        )

        // Generate test signal (seeded for reproducibility)
        srand48(42)
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05) + 0.5 * sin(Float(i) * 0.1)
        }

        // Batch STFT
        let batchSTFT = STFT(config: config)
        let batchResult = await batchSTFT.transform(signal)

        // Streaming STFT - push all at once (should be identical to batch)
        let streaming = StreamingSTFT(config: config)
        await streaming.push(signal)
        let streamingFrames = await streaming.flush()

        // Frame count should match exactly or differ by at most 1 (due to flush padding)
        let frameDiff = abs(streamingFrames.count - batchResult.cols)
        XCTAssertLessThanOrEqual(frameDiff, 1,
            "Frame count difference: streaming=\(streamingFrames.count), batch=\(batchResult.cols)")

        // Compare ALL values (real and imaginary) for overlapping frames
        let minFrames = min(streamingFrames.count, batchResult.cols)
        var maxMagError: Float = 0
        var maxRealError: Float = 0
        var maxImagError: Float = 0

        for frameIdx in 0..<minFrames {
            let streamFrame = streamingFrames[frameIdx]

            for freq in 0..<min(streamFrame.rows, batchResult.rows) {
                // Compare real parts directly
                let realError = abs(streamFrame.real[freq][0] - batchResult.real[freq][frameIdx])
                maxRealError = max(maxRealError, realError)

                // Compare imaginary parts directly
                let imagError = abs(streamFrame.imag[freq][0] - batchResult.imag[freq][frameIdx])
                maxImagError = max(maxImagError, imagError)

                // Compare magnitudes
                let streamMag = sqrt(
                    streamFrame.real[freq][0] * streamFrame.real[freq][0] +
                    streamFrame.imag[freq][0] * streamFrame.imag[freq][0]
                )
                let batchMag = sqrt(
                    batchResult.real[freq][frameIdx] * batchResult.real[freq][frameIdx] +
                    batchResult.imag[freq][frameIdx] * batchResult.imag[freq][frameIdx]
                )
                maxMagError = max(maxMagError, abs(streamMag - batchMag))
            }
        }

        // Streaming should be nearly identical to batch
        XCTAssertLessThan(maxRealError, streamingBatchTolerance,
            "Real part max error: \(maxRealError)")
        XCTAssertLessThan(maxImagError, streamingBatchTolerance,
            "Imaginary part max error: \(maxImagError)")
        XCTAssertLessThan(maxMagError, streamingBatchTolerance,
            "Magnitude max error: \(maxMagError)")
    }

    func testStreamingSTFTIncrementalEquivalence() async {
        // Verify that pushing samples incrementally produces same result as batch
        let config = STFTConfig(
            nFFT: 256,
            hopLength: 64,
            windowType: .hann,
            center: false
        )

        // Generate test signal
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)
        }

        // Streaming STFT - push all at once
        let streamingAll = StreamingSTFT(config: config)
        await streamingAll.push(signal)
        let framesAll = await streamingAll.flush()

        // Streaming STFT - push in chunks
        let streamingChunks = StreamingSTFT(config: config)
        let chunkSize = 128
        for start in stride(from: 0, to: signal.count, by: chunkSize) {
            let end = min(start + chunkSize, signal.count)
            await streamingChunks.push(Array(signal[start..<end]))
        }
        var framesChunks: [ComplexMatrix] = []
        framesChunks.append(contentsOf: await streamingChunks.flush())

        // Should produce same frame count
        XCTAssertEqual(framesAll.count, framesChunks.count,
            "Incremental should produce same frame count: all=\(framesAll.count), chunks=\(framesChunks.count)")

        // Compare values
        var maxError: Float = 0
        for frameIdx in 0..<min(framesAll.count, framesChunks.count) {
            for freq in 0..<framesAll[frameIdx].rows {
                let realError = abs(framesAll[frameIdx].real[freq][0] - framesChunks[frameIdx].real[freq][0])
                let imagError = abs(framesAll[frameIdx].imag[freq][0] - framesChunks[frameIdx].imag[freq][0])
                maxError = max(maxError, max(realError, imagError))
            }
        }

        // Should be exactly identical (no floating point differences expected)
        XCTAssertLessThan(maxError, streamingBatchTolerance,
            "Incremental vs all-at-once max error: \(maxError)")
    }

    func testStreamingSTFTReset() async {
        let streaming = StreamingSTFT(config: STFTConfig(nFFT: 256, hopLength: 64, center: false))

        // Push some samples
        let signal = [Float](repeating: 0.5, count: 512)
        await streaming.push(signal)

        // Reset
        await streaming.reset()

        // Buffer should be empty
        let stats = await streaming.stats
        XCTAssertEqual(stats.currentBufferLevel, 0)
        XCTAssertEqual(stats.totalSamplesReceived, 0)
        XCTAssertEqual(stats.totalFramesOutput, 0)
    }

    // MARK: - StreamingISTFT Tests

    func testStreamingISTFTBasic() async {
        let config = STFTConfig(
            nFFT: 256,
            hopLength: 64,
            windowType: .hann,
            center: false
        )

        let streamingISTFT = StreamingISTFT(config: config)

        // Create simple test frames
        var frames: [ComplexMatrix] = []
        for _ in 0..<8 {
            let real = [[Float]](repeating: [Float.random(in: -1...1)], count: config.nFreqs)
            let imag = [[Float]](repeating: [Float.random(in: -1...1)], count: config.nFreqs)
            frames.append(ComplexMatrix(real: real, imag: imag))
        }

        // Push frames
        let output = await streamingISTFT.push(frames)

        // Should produce some output (may buffer some)
        // Note: ISTFT needs overlap to output, so initial frames may be buffered

        // Flush to get all remaining samples
        let remaining = await streamingISTFT.flush()
        let totalOutput = output.count + remaining.count

        XCTAssertGreaterThan(totalOutput, 0, "Should produce output samples")
    }

    func testStreamingISTFTLatency() async {
        let config = STFTConfig(nFFT: 2048, hopLength: 512)
        let streaming = StreamingISTFT(config: config)

        let latency = await streaming.latencySamples
        XCTAssertEqual(latency, config.nFFT)

        let latencySeconds = await streaming.latencySeconds(sampleRate: 44100)
        let expectedLatency = Float(config.nFFT) / 44100.0
        XCTAssertEqual(latencySeconds, expectedLatency, accuracy: 0.0001)
    }

    // MARK: - Round-Trip Streaming Tests

    func testStreamingRoundTrip() async {
        // Use COLA-compliant config (Hann with 75% overlap) for best reconstruction
        let config = STFTConfig(
            nFFT: 512,
            hopLength: 128,  // 75% overlap
            windowType: .hann,
            center: false
        )

        // Generate test signal
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)
        }

        // Forward: Streaming STFT
        let streamingSTFT = StreamingSTFT(config: config)
        await streamingSTFT.push(signal)
        let frames = await streamingSTFT.flush()

        // Inverse: Streaming ISTFT
        let streamingISTFT = StreamingISTFT(config: config)
        var reconstructed = await streamingISTFT.push(frames)
        reconstructed.append(contentsOf: await streamingISTFT.flush())

        // Compare (accounting for edge effects in non-centered mode)
        let compareLength = min(signal.count, reconstructed.count)
        let startOffset = config.nFFT / 2  // Skip initial transient

        guard compareLength > startOffset * 2 else {
            XCTFail("Not enough samples to compare")
            return
        }

        var maxError: Float = 0
        var sumSquaredError: Float = 0
        var compareCount = 0
        for i in startOffset..<(compareLength - startOffset) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
            compareCount += 1
        }
        let rmse = sqrt(sumSquaredError / Float(compareCount))

        // With COLA-compliant config, error should be small
        // Non-centered mode has some edge effects, so use nonCenteredTolerance
        XCTAssertLessThan(maxError, nonCenteredTolerance,
            "Streaming round-trip max error: \(maxError)")
        XCTAssertLessThan(rmse, nonCenteredTolerance / 10,
            "Streaming round-trip RMSE: \(rmse)")
    }

    func testStreamingRoundTripCOLA() async {
        // Test with different COLA configurations
        let configs: [(Int, Int, String)] = [
            (512, 128, "Hann 75% overlap"),  // nFFT/4
            (512, 256, "Hann 50% overlap"),  // nFFT/2
        ]

        for (nFFT, hopLength, description) in configs {
            let config = STFTConfig(
                nFFT: nFFT,
                hopLength: hopLength,
                windowType: .hann,
                center: false
            )

            // Generate test signal
            var signal = [Float](repeating: 0, count: 4096)
            for i in 0..<signal.count {
                signal[i] = sin(Float(i) * 0.05) + 0.3 * cos(Float(i) * 0.1)
            }

            // Round trip with streaming
            let stft = StreamingSTFT(config: config)
            await stft.push(signal)
            let frames = await stft.flush()

            let istft = StreamingISTFT(config: config)
            var reconstructed = await istft.push(frames)
            reconstructed.append(contentsOf: await istft.flush())

            // Calculate error in valid region
            let margin = nFFT
            let validStart = margin
            let validEnd = min(signal.count, reconstructed.count) - margin

            guard validEnd > validStart else { continue }

            var maxError: Float = 0
            var maxIdx = validStart
            for i in validStart..<validEnd {
                let err = abs(signal[i] - reconstructed[i])
                if err > maxError {
                    maxError = err
                    maxIdx = i
                }
            }

            XCTAssertLessThan(maxError, nonCenteredTolerance,
                "\(description) round-trip error: \(maxError)")
        }
    }

    // MARK: - StreamingProcessor Tests

    func testStreamingProcessorPassthrough() async {
        let config = STFTConfig(
            nFFT: 512,
            hopLength: 128,
            windowType: .hann,
            center: false
        )

        let processor = StreamingProcessor(config: config)

        // Generate test signal
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        // Process with identity transform
        let output = await processor.process(signal) { frames in
            frames  // Pass through unchanged
        }

        let flushed = await processor.flush()
        let totalOutput = output + flushed

        XCTAssertGreaterThan(totalOutput.count, 0, "Should produce output")
    }

    func testStreamingProcessorReset() async {
        let processor = StreamingProcessor(config: STFTConfig(nFFT: 256, hopLength: 64))

        let signal = [Float](repeating: 0.5, count: 512)
        _ = await processor.process(signal) { $0 }

        // Reset and process again
        await processor.reset()

        let signal2 = [Float](repeating: 0.3, count: 512)
        let output = await processor.process(signal2) { $0 }
        let flushed = await processor.flush()

        XCTAssertGreaterThan(output.count + flushed.count, 0)
    }

    // MARK: - Banquet Preset Tests

    func testBanquetPresetSTFT() async {
        let streaming = StreamingSTFT.forBanquet

        let config = await streaming.config
        XCTAssertEqual(config.nFFT, 2048)
        XCTAssertEqual(config.hopLength, 512)
        XCTAssertEqual(config.windowType, .hann)
        XCTAssertEqual(config.center, false)
    }

    func testBanquetPresetISTFT() async {
        let streaming = StreamingISTFT.forBanquet

        let config = await streaming.config
        XCTAssertEqual(config.nFFT, 2048)
        XCTAssertEqual(config.hopLength, 512)
        XCTAssertEqual(config.windowType, .hann)
        XCTAssertEqual(config.center, false)
    }
}
