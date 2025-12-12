import XCTest
@testable import SwiftRosaCore

/// Tests that verify GPU (Metal) and CPU implementations produce identical results.
///
/// These tests ensure mathematical consistency between execution paths, which is critical
/// for deterministic behavior regardless of hardware/thresholds.
final class GPUCPUParityTests: XCTestCase {

    /// Tolerance for GPU/CPU parity - should be very tight since both use Float32.
    /// Any difference beyond this indicates a bug, not precision loss.
    let parityTolerance: Float = 1e-5

    // MARK: - Mel Filterbank Parity Tests

    func testMelFilterbankParitySineTone() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Generate test signal
        let sampleRate: Float = 22050
        let duration: Float = 0.5
        let frequency: Float = 440
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * .pi * frequency * Float(i) / sampleRate)
        }

        // Run mel spectrogram with GPU disabled
        let melSpecCPU = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 1024,
            hopLength: 256,
            nMels: 64,
            useGPU: false
        )
        let cpuResult = await melSpecCPU.transform(signal)

        // Run mel spectrogram with GPU enabled AND forced
        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let melSpecGPU = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 1024,
            hopLength: 256,
            nMels: 64,
            useGPU: true
        )
        let gpuResult = await melSpecGPU.transform(signal)

        // Compare dimensions
        XCTAssertEqual(cpuResult.count, gpuResult.count, "Mel band count mismatch")
        guard !cpuResult.isEmpty else {
            XCTFail("Empty results")
            return
        }
        XCTAssertEqual(cpuResult[0].count, gpuResult[0].count, "Frame count mismatch")

        // Compare values
        var maxError: Float = 0
        var maxRelError: Float = 0

        for mel in 0..<cpuResult.count {
            for frame in 0..<cpuResult[mel].count {
                let cpu = cpuResult[mel][frame]
                let gpu = gpuResult[mel][frame]
                let absError = abs(cpu - gpu)
                maxError = max(maxError, absError)

                if abs(cpu) > 1e-6 {
                    let relError = absError / abs(cpu)
                    maxRelError = max(maxRelError, relError)
                }
            }
        }

        XCTAssertLessThan(maxError, parityTolerance,
            "Mel filterbank GPU/CPU max absolute error \(maxError) exceeds tolerance")

        // Log info for diagnostics
        print("Mel filterbank parity - max abs error: \(maxError), max rel error: \(maxRelError)")
    }

    func testMelFilterbankParityWhiteNoise() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Generate white noise
        let nSamples = 8192
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = Float.random(in: -1...1)
        }

        // CPU path
        let melSpecCPU = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 512,
            hopLength: 128,
            nMels: 40,
            useGPU: false
        )
        let cpuResult = await melSpecCPU.transform(signal)

        // GPU path (forced)
        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let melSpecGPU = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 512,
            hopLength: 128,
            nMels: 40,
            useGPU: true
        )
        let gpuResult = await melSpecGPU.transform(signal)

        // Compare
        XCTAssertEqual(cpuResult.count, gpuResult.count)
        guard !cpuResult.isEmpty else { return }

        var maxError: Float = 0
        for mel in 0..<cpuResult.count {
            for frame in 0..<cpuResult[mel].count {
                let error = abs(cpuResult[mel][frame] - gpuResult[mel][frame])
                maxError = max(maxError, error)
            }
        }

        XCTAssertLessThan(maxError, parityTolerance,
            "White noise mel filterbank GPU/CPU error \(maxError) exceeds tolerance")
    }

    // MARK: - Overlap-Add Parity Tests

    func testOverlapAddParitySineTone() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Generate test signal
        let sampleRate: Float = 22050
        let duration: Float = 0.25
        let frequency: Float = 440
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * .pi * frequency * Float(i) / sampleRate)
        }

        // Perform STFT
        let stftConfig = STFTConfig(nFFT: 1024, hopLength: 256)
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // CPU ISTFT
        let istftCPU = ISTFT(config: stftConfig, useGPU: false)
        let cpuResult = await istftCPU.transform(spectrogram, length: nSamples)

        // GPU ISTFT (forced)
        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let istftGPU = ISTFT(config: stftConfig, useGPU: true)
        let gpuResult = await istftGPU.transform(spectrogram, length: nSamples)

        // Compare
        XCTAssertEqual(cpuResult.count, gpuResult.count, "Output length mismatch")

        var maxError: Float = 0
        for i in 0..<min(cpuResult.count, gpuResult.count) {
            let error = abs(cpuResult[i] - gpuResult[i])
            maxError = max(maxError, error)
        }

        XCTAssertLessThan(maxError, parityTolerance,
            "Overlap-add GPU/CPU max error \(maxError) exceeds tolerance")

        print("Overlap-add parity - max error: \(maxError)")
    }

    func testOverlapAddParityRoundTrip() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Generate complex signal
        let nSamples = 4096
        var signal = [Float](repeating: 0, count: nSamples)
        let sampleRate: Float = 22050

        // Multi-frequency signal
        for i in 0..<nSamples {
            let t = Float(i) / sampleRate
            signal[i] = 0.5 * sin(2 * .pi * 220 * t) +
                        0.3 * sin(2 * .pi * 440 * t) +
                        0.2 * sin(2 * .pi * 880 * t)
        }

        let stftConfig = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // CPU round-trip
        let istftCPU = ISTFT(config: stftConfig, useGPU: false)
        let cpuRoundTrip = await istftCPU.transform(spectrogram, length: nSamples)

        // GPU round-trip (forced)
        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let istftGPU = ISTFT(config: stftConfig, useGPU: true)
        let gpuRoundTrip = await istftGPU.transform(spectrogram, length: nSamples)

        // Both round-trips should be identical
        var maxError: Float = 0
        for i in 0..<min(cpuRoundTrip.count, gpuRoundTrip.count) {
            let error = abs(cpuRoundTrip[i] - gpuRoundTrip[i])
            maxError = max(maxError, error)
        }

        XCTAssertLessThan(maxError, parityTolerance,
            "Round-trip GPU/CPU parity error \(maxError) exceeds tolerance")
    }

    // MARK: - Complex Multiply Parity Tests

    func testComplexMultiplyParity() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        let engine = try MetalEngine()

        // Generate test complex matrices
        let rows = 64
        let cols = 32

        var aReal = [[Float]]()
        var aImag = [[Float]]()
        var bReal = [[Float]]()
        var bImag = [[Float]]()

        for _ in 0..<rows {
            var ar = [Float](repeating: 0, count: cols)
            var ai = [Float](repeating: 0, count: cols)
            var br = [Float](repeating: 0, count: cols)
            var bi = [Float](repeating: 0, count: cols)

            for j in 0..<cols {
                ar[j] = Float.random(in: -1...1)
                ai[j] = Float.random(in: -1...1)
                br[j] = Float.random(in: -1...1)
                bi[j] = Float.random(in: -1...1)
            }

            aReal.append(ar)
            aImag.append(ai)
            bReal.append(br)
            bImag.append(bi)
        }

        // GPU complex multiply
        let gpuResult = try await engine.complexMultiply(
            aReal: aReal, aImag: aImag,
            bReal: bReal, bImag: bImag
        )

        // CPU complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        var cpuReal = [[Float]]()
        var cpuImag = [[Float]]()

        for r in 0..<rows {
            var realRow = [Float](repeating: 0, count: cols)
            var imagRow = [Float](repeating: 0, count: cols)

            for c in 0..<cols {
                let ar = aReal[r][c]
                let ai = aImag[r][c]
                let br = bReal[r][c]
                let bi = bImag[r][c]

                realRow[c] = ar * br - ai * bi
                imagRow[c] = ar * bi + ai * br
            }

            cpuReal.append(realRow)
            cpuImag.append(imagRow)
        }

        // Compare
        var maxRealError: Float = 0
        var maxImagError: Float = 0

        for r in 0..<rows {
            for c in 0..<cols {
                let realError = abs(gpuResult.real[r][c] - cpuReal[r][c])
                let imagError = abs(gpuResult.imag[r][c] - cpuImag[r][c])
                maxRealError = max(maxRealError, realError)
                maxImagError = max(maxImagError, imagError)
            }
        }

        XCTAssertLessThan(maxRealError, parityTolerance,
            "Complex multiply real part GPU/CPU error \(maxRealError) exceeds tolerance")
        XCTAssertLessThan(maxImagError, parityTolerance,
            "Complex multiply imag part GPU/CPU error \(maxImagError) exceeds tolerance")

        print("Complex multiply parity - real error: \(maxRealError), imag error: \(maxImagError)")
    }

    func testComplexMultiplyParityEdgeCases() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        let engine = try MetalEngine()

        // Test special values
        let testCases: [(name: String, a: (Float, Float), b: (Float, Float))] = [
            ("pure real", (2.0, 0.0), (3.0, 0.0)),
            ("pure imaginary", (0.0, 2.0), (0.0, 3.0)),
            ("unit complex", (1.0, 0.0), (0.0, 1.0)),
            ("conjugate pair", (1.0, 2.0), (1.0, -2.0)),
            ("small values", (1e-6, 1e-6), (1e-6, 1e-6)),
            ("large values", (1e6, 1e6), (1e-6, 1e-6)),
        ]

        for (name, a, b) in testCases {
            let aReal = [[a.0]]
            let aImag = [[a.1]]
            let bReal = [[b.0]]
            let bImag = [[b.1]]

            let gpuResult = try await engine.complexMultiply(
                aReal: aReal, aImag: aImag,
                bReal: bReal, bImag: bImag
            )

            // Expected: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            let expectedReal = a.0 * b.0 - a.1 * b.1
            let expectedImag = a.0 * b.1 + a.1 * b.0

            let realError = abs(gpuResult.real[0][0] - expectedReal)
            let imagError = abs(gpuResult.imag[0][0] - expectedImag)

            // Use relative error for large values
            let tolerance: Float = max(parityTolerance, parityTolerance * max(abs(expectedReal), abs(expectedImag)))

            XCTAssertLessThan(realError, tolerance,
                "Complex multiply '\(name)' real error \(realError)")
            XCTAssertLessThan(imagError, tolerance,
                "Complex multiply '\(name)' imag error \(imagError)")
        }
    }

    // MARK: - Large Data Parity Tests

    func testMelFilterbankParityLargeData() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Generate longer signal to ensure GPU threshold is exceeded
        let sampleRate: Float = 22050
        let duration: Float = 2.0  // 2 seconds (still exceeds GPU threshold)
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / sampleRate
            // Chirp signal
            let freq = 200 + 1000 * t / duration
            signal[i] = sin(2 * .pi * freq * t)
        }

        // CPU path
        let melSpecCPU = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            useGPU: false
        )
        let cpuResult = await melSpecCPU.transform(signal)

        // GPU path (forced)
        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let melSpecGPU = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            useGPU: true
        )
        let gpuResult = await melSpecGPU.transform(signal)

        // Compare
        XCTAssertEqual(cpuResult.count, gpuResult.count, "Mel band count mismatch")
        guard !cpuResult.isEmpty else { return }
        XCTAssertEqual(cpuResult[0].count, gpuResult[0].count, "Frame count mismatch")

        var maxError: Float = 0
        var totalError: Double = 0
        var count = 0

        for mel in 0..<cpuResult.count {
            for frame in 0..<cpuResult[mel].count {
                let error = abs(cpuResult[mel][frame] - gpuResult[mel][frame])
                maxError = max(maxError, error)
                totalError += Double(error)
                count += 1
            }
        }

        let avgError = totalError / Double(count)

        XCTAssertLessThan(maxError, parityTolerance,
            "Large data mel filterbank max error \(maxError) exceeds tolerance")

        print("Large mel filterbank parity - max error: \(maxError), avg error: \(avgError), elements: \(count)")
    }

    // MARK: - Determinism Tests

    func testGPUDeterminism() async throws {
        try XCTSkipUnless(MetalEngine.isAvailable, "Metal not available")

        // Run the same computation multiple times on GPU
        // Results should be bit-identical each time

        let savedConfig = MetalEngine.thresholdConfig
        MetalEngine.thresholdConfig = GPUThresholdConfig(forceEnableGPU: true)
        defer { MetalEngine.thresholdConfig = savedConfig }

        let sampleRate: Float = 22050
        let nSamples = 4096

        // Use deterministic signal
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * .pi * 440 * Float(i) / sampleRate)
        }

        let melSpec = MelSpectrogram(
            sampleRate: sampleRate,
            nFFT: 1024,
            hopLength: 256,
            nMels: 64,
            useGPU: true
        )

        // Run multiple times
        let result1 = await melSpec.transform(signal)
        let result2 = await melSpec.transform(signal)
        let result3 = await melSpec.transform(signal)

        // All results should be identical
        for mel in 0..<result1.count {
            for frame in 0..<result1[mel].count {
                XCTAssertEqual(result1[mel][frame], result2[mel][frame],
                    "GPU non-deterministic between run 1 and 2 at [\(mel)][\(frame)]")
                XCTAssertEqual(result2[mel][frame], result3[mel][frame],
                    "GPU non-deterministic between run 2 and 3 at [\(mel)][\(frame)]")
            }
        }
    }
}
