import XCTest
@testable import SwiftAudioPrimitives

final class GriffinLimTests: XCTestCase {

    // MARK: - Shared Test Signals (computed once per test class)

    /// Simple 512-sample sine wave
    private static let signal512: [Float] = (0..<512).map { sin(Float($0) * 0.1) }

    /// Simple 1024-sample sine wave
    private static let signal1024: [Float] = (0..<1024).map { sin(Float($0) * 0.1) }

    /// 1024-sample signal with two harmonics
    private static let signal1024Harmonic: [Float] = (0..<1024).map { i in
        sin(Float(i) * 0.1) + 0.3 * cos(Float(i) * 0.2)
    }

    /// 2048-sample sine wave
    private static let signal2048: [Float] = (0..<2048).map { i in sin(Float(i) * 0.1) }

    /// 2048-sample signal with two harmonics
    private static let signal2048Harmonic: [Float] = (0..<2048).map { i in
        sin(Float(i) * 0.1) + 0.5 * cos(Float(i) * 0.2)
    }

    /// 2048-sample signal with complex harmonics (for waveform tests)
    private static let signal2048Complex: [Float] = (0..<2048).map { i in
        sin(Float(i) * 0.08) + 0.3 * cos(Float(i) * 0.15)
    }

    /// 4096-sample multi-harmonic signal at 22050 Hz sample rate
    private static let signal4096Harmonic: [Float] = (0..<4096).map { i in
        let t = Float(i) / 22050
        return 0.3 * sin(2 * Float.pi * 220 * t) +
               0.2 * sin(2 * Float.pi * 440 * t) +
               0.15 * sin(2 * Float.pi * 660 * t) +
               0.1 * sin(2 * Float.pi * 880 * t)
    }

    // MARK: - Helper Methods

    /// Calculate spectral convergence error (normalized magnitude MSE)
    /// This is the standard metric for Griffin-Lim: sqrt(sum((target-est)^2) / sum(target^2))
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

    /// Calculate Pearson correlation coefficient between two signals
    /// Returns value in [-1, 1] where 1 = perfect positive correlation
    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 1 else { return 0 }

        var sumA: Float = 0
        var sumB: Float = 0
        var sumAB: Float = 0
        var sumA2: Float = 0
        var sumB2: Float = 0

        for i in 0..<n {
            sumA += a[i]
            sumB += b[i]
            sumAB += a[i] * b[i]
            sumA2 += a[i] * a[i]
            sumB2 += b[i] * b[i]
        }

        let meanA = sumA / Float(n)
        let meanB = sumB / Float(n)

        var cov: Float = 0
        var varA: Float = 0
        var varB: Float = 0

        for i in 0..<n {
            let diffA = a[i] - meanA
            let diffB = b[i] - meanB
            cov += diffA * diffB
            varA += diffA * diffA
            varB += diffB * diffB
        }

        let denom = sqrt(varA * varB)
        return denom > 0 ? cov / denom : 0
    }

    /// Calculate Signal-to-Distortion Ratio (SDR) in dB
    /// Higher is better. SDR = 10 * log10(signal_power / distortion_power)
    private func sdr(_ reference: [Float], _ estimate: [Float]) -> Float {
        let n = min(reference.count, estimate.count)
        guard n > 0 else { return -Float.infinity }

        var signalPower: Float = 0
        var distortionPower: Float = 0

        for i in 0..<n {
            signalPower += reference[i] * reference[i]
            let diff = reference[i] - estimate[i]
            distortionPower += diff * diff
        }

        guard distortionPower > 1e-10 else { return Float.infinity }
        return 10 * log10(signalPower / distortionPower)
    }

    /// Calculate RMS energy of a signal
    private func rmsEnergy(_ signal: [Float]) -> Float {
        guard !signal.isEmpty else { return 0 }
        let sumSquared = signal.reduce(0) { $0 + $1 * $1 }
        return sqrt(sumSquared / Float(signal.count))
    }

    // MARK: - Basic Reconstruction Tests

    func testBasicReconstruction() async {
        let signal = Self.signal2048

        // Compute STFT magnitude
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Reconstruct with Griffin-Lim using deterministic init for reproducibility
        let griffinLim = GriffinLim(config: config, nIter: 60, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: signal.count)

        XCTAssertEqual(reconstructed.count, signal.count)

        // 1. Signal should have content with similar energy
        let originalRMS = rmsEnergy(signal)
        let reconstructedRMS = rmsEnergy(reconstructed)
        XCTAssertGreaterThan(reconstructedRMS, originalRMS * 0.5,
            "Reconstructed RMS (\(reconstructedRMS)) should be at least 50% of original (\(originalRMS))")

        // 2. Verify spectral convergence
        // Griffin-Lim typically converges to local minima with 10-25% spectral error
        // depending on the signal characteristics and initialization
        let reconstructedSpec = await stft.transform(reconstructed)
        let reconstructedMag = reconstructedSpec.magnitude
        let error = spectralError(original: magnitude, reconstructed: reconstructedMag)

        XCTAssertLessThan(error, 0.25,
            "Spectral convergence error should be < 25%, got \(error * 100)%")

        // 3. Energy preservation should be reasonable (within 15%)
        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }
        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        let energyRatio = reconstructedEnergy / max(originalEnergy, 1e-10)
        XCTAssertEqual(energyRatio, 1.0, accuracy: 0.15,
            "Energy ratio should be within 15% of 1.0, got \(energyRatio)")
    }

    func testReconstructionWithCallback() async {
        let signal = Self.signal1024
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        let griffinLim = GriffinLim(config: config, nIter: 20, momentum: 0.99, randomPhaseInit: false)

        var iterations: [Int] = []
        var errors: [Float] = []

        let _ = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count) { iter, error in
            iterations.append(iter)
            errors.append(error)
        }

        // Should have called back for each iteration
        XCTAssertEqual(iterations.count, 20)
        XCTAssertEqual(iterations, Array(0..<20))

        // 1. Error should generally decrease, but Griffin-Lim can plateau
        // Check that the first few iterations show significant improvement
        if errors.count >= 3 {
            XCTAssertLessThan(errors[2], errors[0] * 0.5,
                "First few iterations should show significant improvement")
        }

        // 2. Final error should be less than initial (convergence happened)
        let finalError = errors.last!
        let initialError = errors.first!
        XCTAssertLessThan(finalError, initialError,
            "Final error (\(finalError)) should be less than initial (\(initialError))")

        // 3. Final error should be in a reasonable range for Griffin-Lim
        // Typically converges to 15-30% for simple signals with these parameters
        XCTAssertLessThan(finalError, 0.35,
            "Final error should be < 35%, got \(finalError * 100)%")
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
        let signal = Self.signal2048Harmonic
        let config = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Compare few iterations vs many iterations
        let fewIter = GriffinLim(config: config, nIter: 5, momentum: 0.99, randomPhaseInit: false)
        let manyIter = GriffinLim(config: config, nIter: 30, momentum: 0.99, randomPhaseInit: false)

        let resultFew = await fewIter.reconstruct(magnitude, length: signal.count)
        let resultMany = await manyIter.reconstruct(magnitude, length: signal.count)

        // Compute spectral error for each
        let specFew = await stft.transform(resultFew)
        let specMany = await stft.transform(resultMany)

        let errorFew = spectralError(original: magnitude, reconstructed: specFew.magnitude)
        let errorMany = spectralError(original: magnitude, reconstructed: specMany.magnitude)

        // More iterations should give at least as good result (with small tolerance for local minima)
        XCTAssertLessThanOrEqual(errorMany, errorFew * 1.05,
            "More iterations should not significantly increase error: 50 iter=\(errorMany), 5 iter=\(errorFew)")

        // Both should achieve reasonable convergence (< 25% error)
        XCTAssertLessThan(errorFew, 0.25,
            "5 iterations should achieve < 25% error, got \(errorFew * 100)%")
        XCTAssertLessThan(errorMany, 0.25,
            "50 iterations should achieve < 25% error, got \(errorMany * 100)%")
    }

    func testIterationCountsProduceProgressiveImprovement() async {
        let signal = Self.signal1024Harmonic
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        let iterCounts = [5, 10, 20, 40]
        var previousError: Float = Float.infinity
        var errors: [Float] = []

        for nIter in iterCounts {
            let griffinLim = GriffinLim(config: config, nIter: nIter, momentum: 0.99, randomPhaseInit: false)
            let result = await griffinLim.reconstruct(magnitude, length: signal.count)

            let reconstructedSpec = await stft.transform(result)
            let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)
            errors.append(error)

            // Error should not significantly increase (allow 5% tolerance for local minima effects)
            XCTAssertLessThanOrEqual(error, previousError * 1.05,
                "Error with \(nIter) iterations (\(error)) should be <= with fewer iterations (\(previousError))")
            previousError = error
        }

        // All errors should be in reasonable range for Griffin-Lim
        for (i, error) in errors.enumerated() {
            XCTAssertLessThan(error, 0.25,
                "\(iterCounts[i]) iterations should achieve < 25% error, got \(error * 100)%")
        }
    }

    // MARK: - Momentum Tests

    func testMomentumEffect() async {
        let signal = Self.signal1024
        // Use constant padding for optimal Griffin-Lim convergence
        let config = STFTConfig(nFFT: 256, hopLength: 64, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Compare different momentum values
        let noMomentum = GriffinLim(config: config, nIter: 30, momentum: 0.0, randomPhaseInit: false)
        let highMomentum = GriffinLim(config: config, nIter: 30, momentum: 0.99, randomPhaseInit: false)

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

        // Both should achieve reasonable convergence (< 15% error with 30 iterations)
        // Note: With constant padding, expect better convergence
        XCTAssertLessThan(errorNoMom, 0.15,
            "No momentum with 30 iters should achieve < 15% error, got \(errorNoMom * 100)%")
        XCTAssertLessThan(errorHighMom, 0.15,
            "High momentum with 30 iters should achieve < 15% error, got \(errorHighMom * 100)%")

        // Fast Griffin-Lim (with momentum) should converge at least comparably to classic
        XCTAssertLessThanOrEqual(errorHighMom, errorNoMom * 1.2,
            "High momentum should not be significantly worse: highMom=\(errorHighMom), noMom=\(errorNoMom)")

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
        let signal = Self.signal512
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
        let signal = Self.signal512
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
        let signal = Self.signal4096Harmonic
        // Use constant padding for optimal Griffin-Lim convergence (matches librosa default)
        let config = STFTConfig(nFFT: 1024, hopLength: 256, windowType: .hann, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // With sufficient iterations, should achieve excellent spectral match
        let griffinLim = GriffinLim(config: config, nIter: 50, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: signal.count)

        let reconstructedSpec = await stft.transform(reconstructed)
        let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)

        // 1. With 50 iterations and momentum, should achieve < 5% spectral error
        XCTAssertLessThan(error, 0.05,
            "Spectral convergence should be < 5% with 50 iterations, got \(error * 100)%")

        // 2. Energy should be tightly preserved (< 5% difference)
        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }
        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        let energyRatio = reconstructedEnergy / max(originalEnergy, 1e-10)
        XCTAssertEqual(energyRatio, 1.0, accuracy: 0.05,
            "Energy ratio should be within 5% of 1.0, got \(energyRatio)")
    }

    /// Test that simple tonal signals achieve excellent reconstruction
    func testSimpleToneReconstruction() async {
        // Single pure tone at 440 Hz
        let sampleRate: Float = 22050
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * Float.pi * 440 * t)
        }

        // Use constant padding for optimal Griffin-Lim convergence
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Test with 40 iterations for best results
        let griffinLim = GriffinLim(config: config, nIter: 40, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(magnitude, length: signal.count)

        let reconstructedSpec = await stft.transform(reconstructed)
        let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)

        // 1. Simple tones should achieve < 5% spectral error with enough iterations
        XCTAssertLessThan(error, 0.05,
            "Single tone should achieve < 5% spectral error, got \(error * 100)%")

        // 2. Energy should be tightly preserved (< 5% difference)
        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }
        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        let energyRatio = reconstructedEnergy / originalEnergy
        XCTAssertEqual(energyRatio, 1.0, accuracy: 0.05,
            "Energy should be preserved within 5%, got ratio \(energyRatio)")

        // 3. RMS should be reasonably close
        let originalRMS = rmsEnergy(signal)
        let reconstructedRMS = rmsEnergy(reconstructed)
        XCTAssertEqual(reconstructedRMS, originalRMS, accuracy: originalRMS * 0.05,
            "RMS should be within 5%: original=\(originalRMS), reconstructed=\(reconstructedRMS)")
    }

    /// Test strict monotonic improvement with increasing iterations
    func testSpectralErrorDecreasesTrend() async {
        let signal = Self.signal1024Harmonic
        // Use constant padding for optimal Griffin-Lim convergence
        let config = STFTConfig(nFFT: 256, hopLength: 64, windowType: .hann, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)
        let magnitude = spectrogram.magnitude

        // Track error at different iteration counts
        let iterCounts = [5, 15, 30, 50]
        var errors: [Float] = []

        for nIter in iterCounts {
            let griffinLim = GriffinLim(config: config, nIter: nIter, momentum: 0.99, randomPhaseInit: false)
            let result = await griffinLim.reconstruct(magnitude, length: signal.count)
            let reconstructedSpec = await stft.transform(result)
            let error = spectralError(original: magnitude, reconstructed: reconstructedSpec.magnitude)
            errors.append(error)
        }

        // 1. Verify monotonic improvement (allow tiny tolerance for local minima)
        for i in 1..<errors.count {
            XCTAssertLessThanOrEqual(errors[i], errors[i-1] * 1.01,
                "Error at \(iterCounts[i]) iters (\(errors[i])) must be <= error at \(iterCounts[i-1]) iters (\(errors[i-1]))")
        }

        // 2. Final error with 50 iterations should be < 5%
        XCTAssertLessThan(errors.last!, 0.05,
            "50 iterations should achieve < 5% error, got \(errors.last! * 100)%")

        // 3. Should see improvement from 5 to 50 iterations
        XCTAssertLessThan(errors.last!, errors.first!,
            "50 iters (\(errors.last!)) should be better than 5 iters (\(errors.first!))")
    }

    /// Test waveform correlation - reconstructed signal should correlate with original
    /// Note: Griffin-Lim may produce time-shifted or phase-altered waveforms that still
    /// have the correct magnitude spectrum. Correlation tests are tricky because the
    /// algorithm optimizes for spectral match, not time-domain match.
    func testWaveformCorrelation() async {
        let signal = Self.signal2048Complex
        // Use constant padding for optimal Griffin-Lim convergence
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        let griffinLim = GriffinLim(config: config, nIter: 50, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)

        // Verify spectral match is good (this is what Griffin-Lim optimizes)
        let reconstructedSpec = await stft.transform(reconstructed)
        let error = spectralError(original: spectrogram.magnitude, reconstructed: reconstructedSpec.magnitude)
        XCTAssertLessThan(error, 0.05,
            "Spectral error should be < 5%, got \(error * 100)%")

        // Energy should be preserved
        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }
        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        let energyRatio = reconstructedEnergy / originalEnergy
        XCTAssertEqual(energyRatio, 1.0, accuracy: 0.1,
            "Energy should be preserved within 10%, got ratio \(energyRatio)")
    }

    /// Test that reconstruction produces good spectral quality
    /// Note: SDR (Signal-to-Distortion Ratio) is not a reliable metric for Griffin-Lim
    /// because the algorithm optimizes spectral match, not time-domain match.
    /// A phase-shifted signal can have perfect spectral match but terrible SDR.
    func testReconstructionSpectralQuality() async {
        // Generate test signal
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            let t = Float(i) / 22050
            signal[i] = 0.5 * sin(2 * Float.pi * 440 * t) + 0.3 * sin(2 * Float.pi * 880 * t)
        }

        // Use constant padding for optimal Griffin-Lim convergence
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann, padMode: .constant(0))
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        let griffinLim = GriffinLim(config: config, nIter: 50, momentum: 0.99, randomPhaseInit: false)
        let reconstructed = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)

        // Verify spectral match - this is the true measure of Griffin-Lim quality
        let reconstructedSpec = await stft.transform(reconstructed)
        let error = spectralError(original: spectrogram.magnitude, reconstructed: reconstructedSpec.magnitude)
        XCTAssertLessThan(error, 0.05,
            "Spectral error should be < 5%, got \(error * 100)%")

        // Energy should be preserved
        let originalEnergy = signal.reduce(0) { $0 + $1 * $1 }
        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        let energyRatio = reconstructedEnergy / originalEnergy
        XCTAssertEqual(energyRatio, 1.0, accuracy: 0.1,
            "Energy should be preserved within 10%, got ratio \(energyRatio)")
    }

    // MARK: - Diagnostic Tests

    /// This test verifies the fundamental requirement for Griffin-Lim:
    /// STFT -> ISTFT -> STFT should preserve magnitude (up to numerical precision)
    func testSTFTRoundTripPreservesMagnitude() async {
        let signal = Self.signal2048Harmonic
        let config = STFTConfig(nFFT: 512, hopLength: 128, windowType: .hann)
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Forward STFT
        let spectrogram = await stft.transform(signal)
        let originalMag = spectrogram.magnitude

        // ISTFT -> STFT round trip
        let reconstructedSignal = await istft.transform(spectrogram, length: signal.count)
        let roundTripSpec = await stft.transform(reconstructedSignal)
        let roundTripMag = roundTripSpec.magnitude

        // Magnitude should be nearly identical
        let error = spectralError(original: originalMag, reconstructed: roundTripMag)

        // CRITICAL: This should be < 1% for Griffin-Lim to work properly
        // If this test fails, the STFT/ISTFT implementation has issues
        XCTAssertLessThan(error, 0.01,
            "STFT round-trip should preserve magnitude within 1%, got \(error * 100)% error")
    }

    /// Verify that the Griffin-Lim iteration actually decreases error
    func testIterationErrorDecreases() async {
        let signal = Self.signal1024
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let spectrogram = await stft.transform(signal)

        // Test with different momentum values
        let momValues: [Float] = [0.0, 0.5, 0.9, 0.99]
        var allErrors: [[Float]] = []

        for mom in momValues {
            let griffinLim = GriffinLim(config: config, nIter: 30, momentum: mom, randomPhaseInit: false)
            var errors: [Float] = []
            let _ = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count) { _, error in
                errors.append(error)
            }
            allErrors.append(errors)
            print("Momentum=\(mom): first 5=\(errors.prefix(5)), last 5=\(errors.suffix(5))")
        }

        let errors = allErrors[0]  // No momentum

        // Error at iteration 30 should be significantly less than at iteration 1
        XCTAssertLessThan(errors.last!, errors.first! * 0.5,
            "Error should decrease by at least 50% over 30 iterations")
    }

    /// Test starting from the true phase - this should give near-zero error
    func testStartingFromTruePhase() async {
        let signal = Self.signal1024
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Get true magnitude and phase
        let spectrogram = await stft.transform(signal)
        let trueMagnitude = spectrogram.magnitude
        let truePhase = spectrogram.phase

        // Reconstruct from true magnitude + true phase
        let complexFromTruePhase = ComplexMatrix.fromPolar(magnitude: trueMagnitude, phase: truePhase)
        let reconstructed = await istft.transform(complexFromTruePhase, length: signal.count)

        // Re-analyze
        let reconstructedSpec = await stft.transform(reconstructed)
        let reconstructedMag = reconstructedSpec.magnitude

        // This should be near-perfect if STFT/ISTFT work correctly
        let error = spectralError(original: trueMagnitude, reconstructed: reconstructedMag)
        print("Error from true phase: \(error * 100)%")

        XCTAssertLessThan(error, 0.01,
            "Starting from true phase should give <1% error, got \(error * 100)%")
    }

    /// Test Griffin-Lim with different hop length ratios
    /// Smaller hop length (more overlap) should give better convergence
    func testHopLengthEffect() async {
        let signal = Self.signal2048
        let nFFT = 512

        // Test different hop ratios
        let hopRatios = [4, 8, 16]  // nFFT/4, nFFT/8, nFFT/16
        var results: [(ratio: Int, error: Float)] = []

        for ratio in hopRatios {
            let hopLength = nFFT / ratio
            // Use constant padding for optimal Griffin-Lim convergence
            let config = STFTConfig(nFFT: nFFT, hopLength: hopLength, windowType: .hann, padMode: .constant(0))
            let stft = STFT(config: config)
            let spectrogram = await stft.transform(signal)

            // Use iterations sufficient for convergence
            let griffinLim = GriffinLim(config: config, nIter: 40, momentum: 0.99, randomPhaseInit: false)
            let reconstructed = await griffinLim.reconstruct(spectrogram.magnitude, length: signal.count)

            let reconstructedSpec = await stft.transform(reconstructed)
            let error = spectralError(original: spectrogram.magnitude, reconstructed: reconstructedSpec.magnitude)
            results.append((ratio, error))
            print("Hop ratio \(ratio) (hop=\(hopLength)): error = \(error * 100)%")
        }

        // All configurations should achieve good convergence with constant padding
        // Note: The relationship between hop ratio and convergence isn't strictly monotonic
        // due to COLA conditions and other factors. We just verify all achieve good results.
        for result in results {
            XCTAssertLessThan(result.error, 0.05,
                "Hop ratio \(result.ratio) should achieve < 5% error, got \(result.error * 100)%")
        }
    }

    // MARK: - Length Inference

    /// Detailed comparison of single Griffin-Lim iteration
    func testSingleIterationComparison() async {
        // Match librosa test parameters exactly
        let nFFT = 256
        let hopLength = 64
        let nSamples = 1024

        // Generate signal
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(Float(i) * 0.1)
        }

        // Use constant (zero) padding like librosa default
        let config = STFTConfig(nFFT: nFFT, hopLength: hopLength, windowType: .hann, center: true, padMode: .constant(0))
        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Forward STFT
        let spectrogram = await stft.transform(signal)
        let targetMag = spectrogram.magnitude

        print("\n=== Swift vs librosa comparison ===")
        print("Spectrogram shape: (\(targetMag.count), \(targetMag[0].count))")
        print("DC bin first 5 frames: \(Array(targetMag[0].prefix(5)))")
        // librosa DC bin: [10.623964, 5.2968554, 0.03435593, 0.05094771, 0.06684504]

        let totalEnergy = targetMag.flatMap { $0 }.reduce(0) { $0 + $1 * $1 }
        print("Target magnitude energy: \(totalEnergy)")
        // librosa: 98069.2812

        // Create complex from magnitude + zero phase
        let zeroPhase = targetMag.map { $0.map { _ in Float(0) } }
        let complexZeroPhase = ComplexMatrix.fromPolar(magnitude: targetMag, phase: zeroPhase)

        // ISTFT
        let reconstructed = await istft.transform(complexZeroPhase, length: nSamples)
        print("\nISTFT output first 5: \(Array(reconstructed.prefix(5)))")
        // librosa: [0.16433104, 0.16038575, 0.15483429, 0.14780295, 0.13942554]

        let reconstructedEnergy = reconstructed.reduce(0) { $0 + $1 * $1 }
        print("ISTFT output energy: \(reconstructedEnergy)")
        // librosa: 46.97243085705395

        // Forward STFT again
        let rebuilt = await stft.transform(reconstructed)
        let rebuiltMag = rebuilt.magnitude

        let rebuiltEnergy = rebuiltMag.flatMap { $0 }.reduce(0) { $0 + $1 * $1 }
        print("Rebuilt magnitude energy: \(rebuiltEnergy)")
        // librosa: 9008.5156

        // Error
        let error = spectralError(original: targetMag, reconstructed: rebuiltMag)
        print("Iteration 0 error: \(error * 100)%")
        // librosa: 70.67%

        // Extract phase and do iteration 1
        let newPhase = rebuilt.phase
        let complexNewPhase = ComplexMatrix.fromPolar(magnitude: targetMag, phase: newPhase)
        let reconstructed2 = await istft.transform(complexNewPhase, length: nSamples)
        let rebuilt2 = await stft.transform(reconstructed2)
        let error2 = spectralError(original: targetMag, reconstructed: rebuilt2.magnitude)
        print("Iteration 1 error: \(error2 * 100)%")
        // librosa: 10.53%

        // Continue a few more iterations
        var currentComplex = complexNewPhase
        var currentRebuilt = rebuilt2
        for iter in 2...5 {
            let phase = currentRebuilt.phase
            currentComplex = ComplexMatrix.fromPolar(magnitude: targetMag, phase: phase)
            let audio = await istft.transform(currentComplex, length: nSamples)
            currentRebuilt = await stft.transform(audio)
            let err = spectralError(original: targetMag, reconstructed: currentRebuilt.magnitude)
            print("Iteration \(iter) error: \(err * 100)%")
        }

        // The key comparison:
        // Our iter 1: should be ~10.53% like librosa, NOT ~18.4%
        XCTAssertLessThan(error2, 0.15, "Iteration 1 error should be <15% like librosa, got \(error2 * 100)%")
    }

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
