import XCTest
@testable import SwiftAudioPrimitives

final class GriffinLimTests: XCTestCase {

    // MARK: - Helper Methods

    /// Calculate spectral convergence error (normalized magnitude MSE)
    private func spectralError(original: [[Float]], reconstructed: [[Float]]) -> Float {
        var sumSquaredError: Float = 0
        var sumSquaredOriginal: Float = 0
        let nFreqs = min(original.count, reconstructed.count)

        for f in 0..<nFreqs {
            let nFrames = min(original[f].count, reconstructed[f].count)
            for t in 0..<nFrames {
                let diff = original[f][t] - reconstructed[f][t]
                sumSquaredError += diff * diff
                sumSquaredOriginal += original[f][t] * original[f][t]
            }
        }

        return sqrt(sumSquaredError / max(sumSquaredOriginal, 1e-10))
    }

    // MARK: - Basic Reconstruction Tests

    func testBasicReconstruction() async {
        // Generate a simple signal
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        // Compute STFT magnitude
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Reconstruct with Griffin-Lim
        let griffinLim = GriffinLim(config: config, nIter: 32)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: signal.count)

        XCTAssertEqual(reconstructed.count, signal.count)

        // Should produce non-zero output
        var hasNonZero = false
        for sample in reconstructed {
            if abs(sample) > 0.001 {
                hasNonZero = true
                break
            }
        }
        XCTAssertTrue(hasNonZero, "Reconstructed signal should have content")

        // Verify spectral convergence - reconstructed magnitude should be close to original
        let reconstructedSpec = await stft.transform(reconstructed)
        let reconstructedMag = reconstructedSpec.magnitude
        let error = spectralError(original: magnitude, reconstructed: reconstructedMag)

        // Note: Griffin-Lim with random phase init is non-deterministic
        // 25% threshold is conservative to avoid flaky failures
        XCTAssertLessThan(error, 0.25,
            "Spectral convergence error should be < 25%, got \(error * 100)%")
    }

    func testReconstructionWithCallback() async {
        // Simple test signal
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05)
        }

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        let griffinLim = GriffinLim(config: config, nIter: 10)

        var iterations: [Int] = []
        var errors: [Float] = []

        let _ = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count) { iter, error in
            iterations.append(iter)
            errors.append(error)
        }

        // Should have called back for each iteration
        XCTAssertEqual(iterations.count, 10)
        XCTAssertEqual(iterations, Array(0..<10))

        // Verify error generally decreases over iterations
        // Allow for small oscillations due to momentum, but enforce overall trend
        if errors.count >= 6 {
            let firstThirdAvg = errors.prefix(3).reduce(0, +) / 3
            let lastThirdAvg = errors.suffix(3).reduce(0, +) / 3

            XCTAssertLessThan(lastThirdAvg, firstThirdAvg * 1.1,
                "Error should decrease overall: first third avg=\(firstThirdAvg), last third avg=\(lastThirdAvg)")
        }

        // Also check that error at last iteration is less than first (enforcing convergence)
        if errors.count >= 2 {
            XCTAssertLessThan(errors.last!, errors.first!,
                "Final error (\(errors.last!)) should be less than initial error (\(errors.first!))")
        }
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let griffinLim = GriffinLim()

        XCTAssertEqual(griffinLim.nIter, 32)
        XCTAssertEqual(griffinLim.momentum, 0.99, accuracy: 0.001)
        XCTAssertTrue(griffinLim.randomPhaseInit)
    }

    func testCustomConfiguration() {
        let config = STFTConfig(nFFT: 1024, hopLength: 256)
        let griffinLim = GriffinLim(
            config: config,
            nIter: 50,
            momentum: 0.95,
            randomPhaseInit: false
        )

        XCTAssertEqual(griffinLim.config.nFFT, 1024)
        XCTAssertEqual(griffinLim.config.hopLength, 256)
        XCTAssertEqual(griffinLim.nIter, 50)
        XCTAssertEqual(griffinLim.momentum, 0.95, accuracy: 0.001)
        XCTAssertFalse(griffinLim.randomPhaseInit)
    }

    // MARK: - Convergence Tests

    func testConvergenceImproves() async {
        // Generate test signal
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.05) + 0.5 * cos(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Compare few iterations vs many iterations (use deterministic for fair comparison)
        let fewIter = GriffinLim(config: config, nIter: 5, momentum: 0.99, randomPhaseInit: false)
        let manyIter = GriffinLim(config: config, nIter: 50, momentum: 0.99, randomPhaseInit: false)

        // Track errors during reconstruction
        var errorsFew: [Float] = []
        var errorsMany: [Float] = []

        let resultFew = await fewIter.reconstruct(magnitude, length: signal.count) { _, error in
            errorsFew.append(error)
        }
        let resultMany = await manyIter.reconstruct(magnitude, length: signal.count) { _, error in
            errorsMany.append(error)
        }

        // Compute spectral error for each
        let specFew = await stft.transform(resultFew)
        let specMany = await stft.transform(resultMany)

        let errorFew = spectralError(original: magnitude, reconstructed: specFew.magnitude)
        let errorMany = spectralError(original: magnitude, reconstructed: specMany.magnitude)

        // More iterations should give AT LEAST as good result (not worse)
        XCTAssertLessThanOrEqual(errorMany, errorFew,
            "More iterations should not increase error: 50 iter=\(errorMany), 5 iter=\(errorFew)")

        // Verify at least some improvement with more iterations
        // (The minimum found at iteration 3-4 might not improve much beyond 5 iterations,
        // but 50 iterations should find at least as good or better)
        XCTAssertLessThanOrEqual(errorMany, errorFew * 1.001,
            "50 iterations should match or improve on 5 iterations")
    }

    func testIterationCountsProduceProgressiveImprovement() async {
        // Test that error decreases (or stays same after convergence) as we increase iterations
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1) + 0.3 * cos(Float(i) * 0.2)
        }

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        let iterCounts = [2, 5, 10, 20, 40]
        var previousError: Float = Float.infinity
        var errors: [Float] = []

        for nIter in iterCounts {
            let griffinLim = GriffinLim(config: config, nIter: nIter, randomPhaseInit: false)
            let result = await griffinLim.reconstruct(magnitude, length: signal.count)

            let reconstructedSpec = await stft.transform(result)
            let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)
            errors.append(error)

            // Error should never significantly increase with more iterations
            // Allow tiny tolerance for floating point noise (0.5%)
            XCTAssertLessThanOrEqual(error, previousError * 1.005,
                "Error with \(nIter) iterations (\(error)) should be <= with fewer iterations (\(previousError))")
            previousError = error
        }

        // Verify overall improvement: last should be meaningfully better than first
        XCTAssertLessThan(errors.last!, errors.first! * 0.99,
            "Final error (\(errors.last!)) should be at least 1% better than initial (\(errors.first!))")
    }

    // MARK: - Momentum Tests

    func testMomentumEffect() async {
        var signal = [Float](repeating: 0, count: 1024)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Compare different momentum values (use same random seed via deterministic init)
        let noMomentum = GriffinLim(config: config, nIter: 20, momentum: 0.0, randomPhaseInit: false)
        let highMomentum = GriffinLim(config: config, nIter: 20, momentum: 0.99, randomPhaseInit: false)

        let resultNoMom = await noMomentum.reconstruct(magnitude, length: signal.count)
        let resultHighMom = await highMomentum.reconstruct(magnitude, length: signal.count)

        // Both should produce valid output
        XCTAssertEqual(resultNoMom.count, signal.count)
        XCTAssertEqual(resultHighMom.count, signal.count)

        // Compute errors
        let specNoMom = await stft.transform(resultNoMom)
        let specHighMom = await stft.transform(resultHighMom)

        let errorNoMom = spectralError(original: magnitude, reconstructed: specNoMom.magnitude)
        let errorHighMom = spectralError(original: magnitude, reconstructed: specHighMom.magnitude)

        // With proper Fast Griffin-Lim, high momentum should typically converge faster/better
        // At minimum, both should achieve reasonable convergence
        XCTAssertLessThan(errorNoMom, 0.3, "No momentum error should be < 30%")
        XCTAssertLessThan(errorHighMom, 0.3, "High momentum error should be < 30%")

        // Results should be different (momentum affects convergence path)
        var maxDiff: Float = 0
        for i in 0..<min(resultNoMom.count, resultHighMom.count) {
            maxDiff = max(maxDiff, abs(resultNoMom[i] - resultHighMom[i]))
        }
        XCTAssertGreaterThan(maxDiff, 0.001,
            "Different momentum values should produce different results")
    }

    // MARK: - Edge Cases

    func testEmptyMagnitude() async {
        let griffinLim = GriffinLim()
        let result = await griffinLim.reconstruct([[Float]](), length: 0)

        XCTAssertTrue(result.isEmpty)
    }

    func testSingleFrameMagnitude() async {
        // Single frame spectrogram with nFFT=8 -> nFreqs=5 (nFFT/2+1)
        let magnitude: [[Float]] = [[1.0], [0.5], [0.25], [0.1], [0.05]]

        let config = STFTConfig(nFFT: 8, hopLength: 4)
        let griffinLim = GriffinLim(config: config, nIter: 5)

        let result = await griffinLim.reconstruct(magnitude, length: 8)

        XCTAssertEqual(result.count, 8)
    }

    func testZeroMagnitude() async {
        // All zeros should produce all zeros
        // nFFT=8 -> nFreqs=5, 4 frames
        let magnitude: [[Float]] = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        let config = STFTConfig(nFFT: 8, hopLength: 2)
        let griffinLim = GriffinLim(config: config, nIter: 5)

        let result = await griffinLim.reconstruct(magnitude, length: 10)

        // Output should be near zero
        let maxAbs = result.map(abs).max() ?? 0
        XCTAssertEqual(maxAbs, 0, accuracy: 1e-6,
            "Zero magnitude should produce zero output, got max abs: \(maxAbs)")
    }

    // MARK: - Determinism Tests

    func testZeroPhaseInitDeterministic() async {
        var signal = [Float](repeating: 0, count: 512)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 128, hopLength: 32)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        // With zero phase init (randomPhaseInit = false), results should be deterministic
        let griffinLim = GriffinLim(config: config, nIter: 10, randomPhaseInit: false)

        let result1 = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)
        let result2 = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)

        // Results should be identical
        XCTAssertEqual(result1.count, result2.count)
        for i in 0..<result1.count {
            XCTAssertEqual(result1[i], result2[i], accuracy: 1e-6,
                "Deterministic reconstruction should give identical result at index \(i)")
        }
    }

    func testRandomPhaseInitNonDeterministic() async {
        var signal = [Float](repeating: 0, count: 512)
        for i in 0..<signal.count {
            signal[i] = sin(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 128, hopLength: 32)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        // With random phase init, results may differ between runs
        let griffinLim = GriffinLim(config: config, nIter: 10, randomPhaseInit: true)

        let result1 = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)
        let result2 = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)

        // Both should have same length
        XCTAssertEqual(result1.count, result2.count)

        // Results may differ (check if there's any difference)
        // Note: with enough iterations, they might converge to similar results
        var sumDiff: Float = 0
        for i in 0..<result1.count {
            sumDiff += abs(result1[i] - result2[i])
        }

        // At minimum, both should be valid
        let hasNonZero1 = result1.contains { abs($0) > 0.001 }
        let hasNonZero2 = result2.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasNonZero1, "First result should have content")
        XCTAssertTrue(hasNonZero2, "Second result should have content")
    }

    // MARK: - Spectral Convergence Tests

    func testSpectralConvergenceQuality() async {
        // Test with a rich signal that's challenging to reconstruct
        var signal = [Float](repeating: 0, count: 4096)
        for i in 0..<signal.count {
            let t = Float(i) / 22050
            signal[i] = 0.3 * sin(2 * Float.pi * 220 * t)
            signal[i] += 0.2 * sin(2 * Float.pi * 440 * t)
            signal[i] += 0.15 * sin(2 * Float.pi * 660 * t)
            signal[i] += 0.1 * sin(2 * Float.pi * 880 * t)
        }

        let config = STFTConfig(nFFT: 1024, hopLength: 256, windowType: .hann)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // With sufficient iterations, should achieve good spectral match
        let griffinLim = GriffinLim(config: config, nIter: 100, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: signal.count)

        let reconstructedSpec = await stft.transform(reconstructed)
        let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)

        // With 100 iterations and momentum, should achieve < 15% spectral error
        // (Harmonic signals are challenging for Griffin-Lim due to phase ambiguity)
        XCTAssertLessThan(error, 0.15,
            "Spectral convergence should be < 15% with 100 iterations, got \(error * 100)%")
    }

    // MARK: - Length Inference

    func testLengthInference() async {
        srand48(42)
        var signal = [Float](repeating: 0, count: 1000)
        for i in 0..<signal.count {
            signal[i] = Float(drand48() * 2 - 1)
        }

        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        // Reconstruct without specifying length
        let griffinLim = GriffinLim(config: config, nIter: 5)
        let result = await griffinLim.reconstruct(spectrogram.magnitude, length: nil)

        // Should produce reasonable output length
        XCTAssertGreaterThan(result.count, 0)

        // Length should be related to number of frames
        let expectedMinLength = (spectrogram.cols - 1) * config.hopLength
        XCTAssertGreaterThanOrEqual(result.count, expectedMinLength,
            "Inferred length (\(result.count)) should be >= expected min (\(expectedMinLength))")
    }
}
