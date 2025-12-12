import XCTest
@testable import SwiftAudioPrimitives

final class MetalEngineTests: XCTestCase {

    // MARK: - Tolerance Analysis
    //
    // Expected error bounds for GPU vs CPU:
    // - Single precision float: ~7 significant digits (ulp ≈ 1e-7)
    // - Accumulated error after N additions: O(sqrt(N) * ulp) for random signs
    // - For overlap-add with k overlapping frames: error ≈ k * ulp
    //
    // With frameLength=2048, hopLength=512, max overlap = frameLength/hopLength = 4
    // Expected max error ≈ 4 * 1e-6 ≈ 4e-6
    // We use 1e-4 to account for implementation differences
    private let gpuCpuTolerance: Float = 1e-4

    // For exact integer/simple operations, use tighter tolerance
    private let exactTolerance: Float = 1e-6

    // MARK: - Availability Tests

    func testMetalAvailability() {
        // Metal should be available on modern Apple devices
        // This test documents the check, actual availability depends on hardware
        let isAvailable = MetalEngine.isAvailable
        print("Metal availability: \(isAvailable)")
        // No assertion - just documenting behavior
    }

    func testShouldUseGPUHeuristic() {
        // Small data: prefer CPU
        XCTAssertFalse(MetalEngine.shouldUseGPU(elementCount: 1000))
        XCTAssertFalse(MetalEngine.shouldUseGPU(elementCount: 50_000))

        // Large data: prefer GPU
        XCTAssertTrue(MetalEngine.shouldUseGPU(elementCount: 100_001))
        XCTAssertTrue(MetalEngine.shouldUseGPU(elementCount: 1_000_000))
    }

    // MARK: - Overlap-Add Tests

    func testOverlapAddCorrectness() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available on this device")
        }

        let engine = try MetalEngine()

        // Create simple test frames with known values for analytical verification
        let frameLength = 256
        let frameCount = 8
        let hopLength = 64
        let overlapRatio = frameLength / hopLength  // = 4 overlapping frames

        var frames: [[Float]] = []
        for f in 0..<frameCount {
            var frame = [Float](repeating: 0, count: frameLength)
            // Simple pattern: each frame has constant value (f+1)*0.1
            for i in 0..<frameLength {
                frame[i] = Float(f + 1) * 0.1
            }
            frames.append(frame)
        }

        // Rectangular window for easy analytical verification
        let window = [Float](repeating: 1.0, count: frameLength)
        let outputLength = (frameCount - 1) * hopLength + frameLength

        let result = try await engine.overlapAdd(
            frames: frames,
            window: window,
            hopLength: hopLength,
            outputLength: outputLength
        )

        XCTAssertEqual(result.count, outputLength)

        // Analytical verification: at any position, the value should be the sum
        // of contributions from all frames that overlap at that position
        var maxError: Float = 0
        for pos in 0..<outputLength {
            var expected: Float = 0
            for f in 0..<frameCount {
                let frameStart = f * hopLength
                let frameEnd = frameStart + frameLength
                if pos >= frameStart && pos < frameEnd {
                    // This frame contributes at this position
                    expected += Float(f + 1) * 0.1
                }
            }
            let error = abs(result[pos] - expected)
            maxError = max(maxError, error)

            // Verify each position individually for small errors
            XCTAssertEqual(result[pos], expected, accuracy: exactTolerance,
                "Mismatch at position \(pos): got \(result[pos]), expected \(expected)")
        }

        // Overall max error should be very small
        XCTAssertLessThan(maxError, exactTolerance,
            "Overall overlap-add max error: \(maxError)")

        // Verify specific positions analytically:
        // Position 0: only frame 0 contributes -> 0.1
        XCTAssertEqual(result[0], 0.1, accuracy: exactTolerance)

        // Position hopLength*3 (=192): frames 0,1,2,3 all contribute -> 0.1+0.2+0.3+0.4 = 1.0
        let midPos = hopLength * 3
        if midPos < outputLength {
            XCTAssertEqual(result[midPos], 1.0, accuracy: exactTolerance,
                "At position \(midPos), expected sum of frames 0-3 = 1.0")
        }
    }

    func testOverlapAddEmptyInput() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()
        let result = try await engine.overlapAdd(
            frames: [],
            window: [1.0, 1.0, 1.0, 1.0],
            hopLength: 2,
            outputLength: 0
        )

        XCTAssertTrue(result.isEmpty)
    }

    func testOverlapAddCOLAProperty() async throws {
        // Test that overlap-add with COLA-compliant window produces constant sum
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Hann window with 50% overlap satisfies COLA: sum of overlapped windows = constant
        let frameLength = 512
        let hopLength = 256  // 50% overlap
        let frameCount = 10
        let outputLength = (frameCount - 1) * hopLength + frameLength

        // All frames contain 1.0 - with COLA, output should be constant
        var frames: [[Float]] = []
        for _ in 0..<frameCount {
            frames.append([Float](repeating: 1.0, count: frameLength))
        }

        // Hann window
        let window = Windows.generate(.hann, length: frameLength)

        let result = try await engine.overlapAdd(
            frames: frames,
            window: window,
            hopLength: hopLength,
            outputLength: outputLength
        )

        // In the central region (avoiding edges), the sum should be constant
        // For Hann with 50% overlap, the constant is 1.0
        let margin = frameLength
        let validStart = margin
        let validEnd = outputLength - margin

        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude
        for i in validStart..<validEnd {
            minVal = min(minVal, result[i])
            maxVal = max(maxVal, result[i])
        }

        // COLA property: variation should be minimal
        let variation = maxVal - minVal
        XCTAssertLessThan(variation, 0.01,
            "COLA property violated: variation \(variation), range [\(minVal), \(maxVal)]")

        // Mean should be close to 1.0 for Hann with 50% overlap
        let centralRegion = Array(result[validStart..<validEnd])
        let mean = centralRegion.reduce(0, +) / Float(centralRegion.count)
        XCTAssertEqual(mean, 1.0, accuracy: 0.01,
            "COLA mean should be 1.0 for Hann 50% overlap, got \(mean)")
    }

    // MARK: - Mel Filterbank Tests

    func testMelFilterbankCorrectness() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Simple test case
        let nFreqs = 4
        let nFrames = 3
        let nMels = 2

        // Spectrogram: identity-like pattern
        let spectrogram: [[Float]] = [
            [1.0, 0.0, 0.0],  // freq 0
            [0.0, 1.0, 0.0],  // freq 1
            [0.0, 0.0, 1.0],  // freq 2
            [1.0, 1.0, 1.0],  // freq 3
        ]

        // Filterbank: simple weighting
        let filterbank: [[Float]] = [
            [1.0, 0.5, 0.0, 0.0],  // mel 0: weights freqs 0, 1
            [0.0, 0.5, 1.0, 0.5],  // mel 1: weights freqs 1, 2, 3
        ]

        let result = try await engine.applyMelFilterbank(
            spectrogram: spectrogram,
            filterbank: filterbank
        )

        XCTAssertEqual(result.count, nMels)
        XCTAssertEqual(result[0].count, nFrames)

        // Manual calculation verification:
        // mel[0][0] = 1.0*1.0 + 0.5*0.0 + 0.0*0.0 + 0.0*1.0 = 1.0
        // mel[0][1] = 1.0*0.0 + 0.5*1.0 + 0.0*0.0 + 0.0*1.0 = 0.5
        // mel[0][2] = 1.0*0.0 + 0.5*0.0 + 0.0*1.0 + 0.0*1.0 = 0.0
        XCTAssertEqual(result[0][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0][1], 0.5, accuracy: 0.001)
        XCTAssertEqual(result[0][2], 0.0, accuracy: 0.001)

        // mel[1][0] = 0.0*1.0 + 0.5*0.0 + 1.0*0.0 + 0.5*1.0 = 0.5
        // mel[1][1] = 0.0*0.0 + 0.5*1.0 + 1.0*0.0 + 0.5*1.0 = 1.0
        // mel[1][2] = 0.0*0.0 + 0.5*0.0 + 1.0*1.0 + 0.5*1.0 = 1.5
        XCTAssertEqual(result[1][0], 0.5, accuracy: 0.001)
        XCTAssertEqual(result[1][1], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[1][2], 1.5, accuracy: 0.001)
    }

    func testMelFilterbankEmptyInput() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()
        let result = try await engine.applyMelFilterbank(
            spectrogram: [],
            filterbank: []
        )

        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Complex Multiplication Tests

    func testComplexMultiplicationCorrectness() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Test (3+4i) * (1+2i) = (3-8) + (6+4)i = -5 + 10i
        let aReal: [[Float]] = [[3.0]]
        let aImag: [[Float]] = [[4.0]]
        let bReal: [[Float]] = [[1.0]]
        let bImag: [[Float]] = [[2.0]]

        let result = try await engine.complexMultiply(
            aReal: aReal,
            aImag: aImag,
            bReal: bReal,
            bImag: bImag
        )

        XCTAssertEqual(result.real[0][0], -5.0, accuracy: 0.001)
        XCTAssertEqual(result.imag[0][0], 10.0, accuracy: 0.001)
    }

    func testComplexMultiplicationBatch() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Test multiple elements
        let aReal: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let aImag: [[Float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let bReal: [[Float]] = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
        let bImag: [[Float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        let result = try await engine.complexMultiply(
            aReal: aReal,
            aImag: aImag,
            bReal: bReal,
            bImag: bImag
        )

        // Real * Real with zero imaginary = Real result
        XCTAssertEqual(result.real[0][0], 2.0, accuracy: 0.001)
        XCTAssertEqual(result.real[0][1], 4.0, accuracy: 0.001)
        XCTAssertEqual(result.real[0][2], 6.0, accuracy: 0.001)
        XCTAssertEqual(result.real[1][0], 8.0, accuracy: 0.001)
        XCTAssertEqual(result.real[1][1], 10.0, accuracy: 0.001)
        XCTAssertEqual(result.real[1][2], 12.0, accuracy: 0.001)

        // Imaginary parts should be zero
        for row in result.imag {
            for val in row {
                XCTAssertEqual(val, 0.0, accuracy: 0.001)
            }
        }
    }

    func testComplexMultiplicationEmptyInput() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()
        let result = try await engine.complexMultiply(
            aReal: [],
            aImag: [],
            bReal: [],
            bImag: []
        )

        XCTAssertTrue(result.real.isEmpty)
        XCTAssertTrue(result.imag.isEmpty)
    }

    // MARK: - Performance Comparison Tests

    func testOverlapAddGPUvsCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Larger test for meaningful comparison
        let frameLength = 2048
        let frameCount = 100
        let hopLength = 512
        let maxOverlap = frameLength / hopLength  // = 4

        // Use seeded random for reproducibility
        srand48(42)
        var frames: [[Float]] = []
        for _ in 0..<frameCount {
            var frame = [Float](repeating: 0, count: frameLength)
            for i in 0..<frameLength {
                frame[i] = Float(drand48() * 2 - 1)
            }
            frames.append(frame)
        }

        let window = Windows.generate(.hann, length: frameLength)
        let outputLength = (frameCount - 1) * hopLength + frameLength

        // GPU version
        let gpuResult = try await engine.overlapAdd(
            frames: frames,
            window: window,
            hopLength: hopLength,
            outputLength: outputLength
        )

        // CPU version (reference implementation using Accelerate for accuracy)
        var cpuResult = [Float](repeating: 0, count: outputLength)
        for (frameIdx, frame) in frames.enumerated() {
            let startIdx = frameIdx * hopLength
            for i in 0..<frameLength {
                let outIdx = startIdx + i
                if outIdx < outputLength {
                    cpuResult[outIdx] += frame[i] * window[i]
                }
            }
        }

        // Results should match
        XCTAssertEqual(gpuResult.count, cpuResult.count)

        var maxError: Float = 0
        var sumSquaredError: Float = 0
        for i in 0..<min(gpuResult.count, cpuResult.count) {
            let error = abs(gpuResult[i] - cpuResult[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
        }
        let rmse = sqrt(sumSquaredError / Float(cpuResult.count))

        // Expected error: O(maxOverlap * ulp) for accumulated floating point
        // With maxOverlap=4 and typical values ~1.0, error should be < 1e-5
        // We use gpuCpuTolerance (1e-4) to be safe
        XCTAssertLessThan(maxError, gpuCpuTolerance,
            "GPU/CPU max error: \(maxError), expected < \(gpuCpuTolerance)")
        XCTAssertLessThan(rmse, gpuCpuTolerance / 10,
            "GPU/CPU RMSE: \(rmse)")
    }

    func testMelFilterbankGPUvsCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Create realistic mel filterbank test
        let nFreqs = 513  // nFFT/2 + 1 for nFFT=1024
        let nFrames = 50
        let nMels = 80

        // Random spectrogram (seeded for reproducibility)
        srand48(123)
        var spectrogram: [[Float]] = []
        for _ in 0..<nFreqs {
            var row = [Float](repeating: 0, count: nFrames)
            for j in 0..<nFrames {
                row[j] = Float(drand48())  // 0 to 1
            }
            spectrogram.append(row)
        }

        // Create simple filterbank (normalized triangular)
        var filterbank: [[Float]] = []
        for m in 0..<nMels {
            var filter = [Float](repeating: 0, count: nFreqs)
            let center = nFreqs * m / nMels
            let width = max(nFreqs / nMels, 3)
            for f in max(0, center - width)..<min(nFreqs, center + width) {
                let dist = abs(f - center)
                filter[f] = 1.0 - Float(dist) / Float(width)
            }
            filterbank.append(filter)
        }

        // GPU version
        let gpuResult = try await engine.applyMelFilterbank(
            spectrogram: spectrogram,
            filterbank: filterbank
        )

        // CPU reference: mel = filterbank @ spectrogram
        var cpuResult: [[Float]] = []
        for m in 0..<nMels {
            var melRow = [Float](repeating: 0, count: nFrames)
            for t in 0..<nFrames {
                var sum: Float = 0
                for f in 0..<nFreqs {
                    sum += filterbank[m][f] * spectrogram[f][t]
                }
                melRow[t] = sum
            }
            cpuResult.append(melRow)
        }

        // Compare
        var maxError: Float = 0
        for m in 0..<nMels {
            for t in 0..<nFrames {
                let error = abs(gpuResult[m][t] - cpuResult[m][t])
                maxError = max(maxError, error)
            }
        }

        // Error should be small (sum of nFreqs multiplications)
        // Expected: O(sqrt(nFreqs) * ulp) ≈ sqrt(513) * 1e-7 ≈ 2e-6
        XCTAssertLessThan(maxError, gpuCpuTolerance,
            "Mel filterbank GPU/CPU max error: \(maxError)")
    }

    // MARK: - Complex Number Edge Cases

    func testComplexMultiplicationWithLargeValues() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // Test with larger values to check overflow handling
        let aReal: [[Float]] = [[1000.0, -1000.0]]
        let aImag: [[Float]] = [[500.0, -500.0]]
        let bReal: [[Float]] = [[100.0, 100.0]]
        let bImag: [[Float]] = [[50.0, 50.0]]

        let result = try await engine.complexMultiply(
            aReal: aReal,
            aImag: aImag,
            bReal: bReal,
            bImag: bImag
        )

        // (1000 + 500i) * (100 + 50i) = 100000 - 25000 + (50000 + 50000)i = 75000 + 100000i
        XCTAssertEqual(result.real[0][0], 75000.0, accuracy: 1.0)
        XCTAssertEqual(result.imag[0][0], 100000.0, accuracy: 1.0)

        // (-1000 - 500i) * (100 + 50i) = -100000 + 25000 + (-50000 - 50000)i = -75000 - 100000i
        XCTAssertEqual(result.real[0][1], -75000.0, accuracy: 1.0)
        XCTAssertEqual(result.imag[0][1], -100000.0, accuracy: 1.0)
    }

    func testComplexMultiplicationUnitCircle() async throws {
        // Test multiplication on unit circle (should preserve magnitude)
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let engine = try MetalEngine()

        // e^(iπ/4) * e^(iπ/4) = e^(iπ/2) = i
        let cos45 = cos(Float.pi / 4)
        let sin45 = sin(Float.pi / 4)

        let aReal: [[Float]] = [[cos45]]
        let aImag: [[Float]] = [[sin45]]
        let bReal: [[Float]] = [[cos45]]
        let bImag: [[Float]] = [[sin45]]

        let result = try await engine.complexMultiply(
            aReal: aReal,
            aImag: aImag,
            bReal: bReal,
            bImag: bImag
        )

        // Result should be approximately i = (0 + 1i)
        XCTAssertEqual(result.real[0][0], 0.0, accuracy: exactTolerance,
            "Real part should be 0 for e^(iπ/2)")
        XCTAssertEqual(result.imag[0][0], 1.0, accuracy: exactTolerance,
            "Imaginary part should be 1 for e^(iπ/2)")

        // Verify magnitude is preserved (should be 1)
        let magnitude = sqrt(result.real[0][0] * result.real[0][0] + result.imag[0][0] * result.imag[0][0])
        XCTAssertEqual(magnitude, 1.0, accuracy: exactTolerance,
            "Unit circle multiplication should preserve magnitude")
    }
}
