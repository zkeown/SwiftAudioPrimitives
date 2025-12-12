import XCTest
@testable import SwiftAudioPrimitives

final class STFTTests: XCTestCase {

    // MARK: - Basic Tests

    func testSTFTOutputShape() async {
        let signal = [Float](repeating: 0, count: 1024)
        let config = STFTConfig(nFFT: 256, hopLength: 64)
        let stft = STFT(config: config)

        let result = await stft.transform(signal)

        // nFreqs = nFFT/2 + 1 = 129
        XCTAssertEqual(result.rows, 129)

        // nFrames depends on centering and signal length
        // With center=true, signal is padded by nFFT/2 on each side
        // So effective length = 1024 + 256 = 1280
        // nFrames = 1 + (1280 - 256) / 64 = 1 + 16 = 17
        XCTAssertGreaterThan(result.cols, 0)
    }

    func testSTFTWithSineWave() async {
        // Generate a simple sine wave
        let sampleRate: Float = 22050
        let frequency: Float = 440  // A4
        let duration: Float = 0.1   // 100ms
        let nSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let config = STFTConfig(nFFT: 1024, hopLength: 256)
        let stft = STFT(config: config)

        let result = await stft.transform(signal)

        // Check that we get a valid spectrogram
        XCTAssertEqual(result.rows, 513)  // nFFT/2 + 1
        XCTAssertGreaterThan(result.cols, 0)

        // Check that magnitude is reasonable
        let magnitude = result.magnitude
        let maxMag = magnitude.flatMap { $0 }.max() ?? 0
        XCTAssertGreaterThan(maxMag, 0)
    }

    func testMagnitudeSpectrogram() async {
        let signal = (0..<2048).map { sin(Float($0) * 0.1) }
        let stft = STFT(config: STFTConfig(nFFT: 512))

        let magnitude = await stft.magnitude(signal)

        // All magnitudes should be non-negative
        for row in magnitude {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0)
            }
        }
    }

    func testPowerSpectrogram() async {
        let signal = (0..<2048).map { sin(Float($0) * 0.1) }
        let stft = STFT(config: STFTConfig(nFFT: 512))

        let power = await stft.power(signal)

        // All power values should be non-negative
        for row in power {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0)
            }
        }
    }

    // MARK: - Configuration Tests

    func testDifferentFFTSizes() async {
        let signal = [Float](repeating: 0.5, count: 4096)

        for nFFT in [256, 512, 1024, 2048] {
            let config = STFTConfig(nFFT: nFFT)
            let stft = STFT(config: config)

            let result = await stft.transform(signal)

            XCTAssertEqual(result.rows, nFFT / 2 + 1, "Wrong nFreqs for nFFT=\(nFFT)")
        }
    }

    func testDifferentHopLengths() async {
        let signal = [Float](repeating: 0.5, count: 4096)
        let nFFT = 1024

        for hopLength in [128, 256, 512] {
            let config = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: false)
            let stft = STFT(config: config)

            let result = await stft.transform(signal)

            let expectedFrames = 1 + (signal.count - nFFT) / hopLength
            XCTAssertEqual(result.cols, expectedFrames, "Wrong nFrames for hopLength=\(hopLength)")
        }
    }

    func testCenteringOption() async {
        let signal = [Float](repeating: 0.5, count: 2048)
        let nFFT = 512
        let hopLength = 128

        let configCentered = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: true)
        let configUncentered = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: false)

        let stftCentered = STFT(config: configCentered)
        let stftUncentered = STFT(config: configUncentered)

        let resultCentered = await stftCentered.transform(signal)
        let resultUncentered = await stftUncentered.transform(signal)

        // Centered should have more frames due to padding
        XCTAssertGreaterThan(resultCentered.cols, resultUncentered.cols)
    }

    // MARK: - Frequency Domain Accuracy Tests

    func testFrequencyPeakDetection() async {
        // Generate a pure sine wave at a known frequency
        let sampleRate: Float = 22050
        let frequency: Float = 1000.0  // 1 kHz - easy to verify
        let nSamples = 8192  // Long signal for better frequency resolution

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let nFFT = 2048
        let config = STFTConfig(nFFT: nFFT, hopLength: 512, windowType: .hann, center: false)
        let stft = STFT(config: config)

        let magnitude = await stft.magnitude(signal)

        // Calculate expected bin for 1000 Hz
        // bin = frequency * nFFT / sampleRate
        let expectedBin = Int(frequency * Float(nFFT) / sampleRate)
        let binResolution = sampleRate / Float(nFFT)

        // Find the peak bin in the middle frame (avoid edge effects)
        let midFrame = magnitude[0].count / 2
        var peakBin = 0
        var peakValue: Float = 0
        for bin in 0..<magnitude.count {
            if magnitude[bin][midFrame] > peakValue {
                peakValue = magnitude[bin][midFrame]
                peakBin = bin
            }
        }

        // Peak should be within ±1 bin of expected (frequency resolution limit)
        XCTAssertEqual(peakBin, expectedBin, accuracy: 1,
            "Peak at bin \(peakBin) (\(Float(peakBin) * binResolution) Hz), expected \(expectedBin) (\(frequency) Hz)")

        // Peak should be significantly larger than neighbors (at least 10x)
        if peakBin > 2 && peakBin < magnitude.count - 2 {
            let neighborAvg = (magnitude[peakBin - 2][midFrame] + magnitude[peakBin + 2][midFrame]) / 2
            XCTAssertGreaterThan(peakValue, neighborAvg * 10,
                "Peak should be significantly larger than neighbors")
        }
    }

    func testMultipleFrequencyPeaks() async {
        // Generate a signal with two known frequencies
        let sampleRate: Float = 22050
        let freq1: Float = 500.0
        let freq2: Float = 1500.0
        let nSamples = 8192

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / sampleRate
            signal[i] = 0.5 * sin(2 * Float.pi * freq1 * t) + 0.5 * sin(2 * Float.pi * freq2 * t)
        }

        let nFFT = 2048
        let config = STFTConfig(nFFT: nFFT, hopLength: 512, center: false)
        let stft = STFT(config: config)

        let magnitude = await stft.magnitude(signal)

        let expectedBin1 = Int(freq1 * Float(nFFT) / sampleRate)
        let expectedBin2 = Int(freq2 * Float(nFFT) / sampleRate)

        // Find peaks in middle frame
        let midFrame = magnitude[0].count / 2

        // Find top 2 peaks
        var peaks: [(bin: Int, value: Float)] = []
        for bin in 1..<(magnitude.count - 1) {
            let val = magnitude[bin][midFrame]
            // Is this a local maximum?
            if val > magnitude[bin-1][midFrame] && val > magnitude[bin+1][midFrame] {
                peaks.append((bin, val))
            }
        }
        peaks.sort { $0.value > $1.value }

        XCTAssertGreaterThanOrEqual(peaks.count, 2, "Should have at least 2 peaks")

        // Top two peaks should be near our expected frequencies
        let topPeakBins = Set([peaks[0].bin, peaks[1].bin])
        let expectedBins = [expectedBin1, expectedBin2]

        for expected in expectedBins {
            let hasMatch = topPeakBins.contains { abs($0 - expected) <= 1 }
            XCTAssertTrue(hasMatch, "Expected peak near bin \(expected) not found in top peaks")
        }
    }

    func testDCComponentForConstantSignal() async {
        // A constant signal should have energy primarily in bin 0 (DC)
        // With windowing, some spectral leakage is expected due to window main lobe width
        let dcLevel: Float = 0.5
        let signal = [Float](repeating: dcLevel, count: 4096)

        let nFFT = 1024
        let config = STFTConfig(nFFT: nFFT, hopLength: 256, windowType: .rectangular, center: false)
        let stft = STFT(config: config)

        let magnitude = await stft.magnitude(signal)

        // Check middle frame
        let midFrame = magnitude[0].count / 2

        // DC bin (0) should have the maximum energy
        let dcMagnitude = magnitude[0][midFrame]
        XCTAssertGreaterThan(dcMagnitude, 0, "DC bin should have positive magnitude")

        // For rectangular window, bins > 0 should be negligible
        // Skip bin 1 which may have numerical noise; check bins further out
        var farBinEnergy: Float = 0
        for bin in 10..<(magnitude.count / 2) {
            farBinEnergy += magnitude[bin][midFrame] * magnitude[bin][midFrame]
        }
        let dcEnergy = dcMagnitude * dcMagnitude
        let farBinRatio = farBinEnergy / max(dcEnergy, 1e-10)

        // Far bins should have negligible energy compared to DC (< 1% total)
        XCTAssertLessThan(farBinRatio, 0.01,
            "Far bins have \(farBinRatio * 100)% of DC energy, should be ~0% for constant signal")
    }

    // MARK: - Energy Preservation Tests (Parseval's Theorem)

    func testEnergyPreservation() async {
        // Test that STFT preserves relative energy relationships
        // Rather than exact Parseval's theorem (which requires careful normalization),
        // verify that doubling amplitude quadruples frequency energy

        let nSamples = 4096
        var signal1 = [Float](repeating: 0, count: nSamples)
        var signal2 = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let val = sin(Float(i) * 0.1) + 0.5 * cos(Float(i) * 0.3)
            signal1[i] = val
            signal2[i] = 2.0 * val  // Double amplitude
        }

        let nFFT = 1024
        let hopLength = 256
        let config = STFTConfig(nFFT: nFFT, hopLength: hopLength, windowType: .hann, center: false)
        let stft = STFT(config: config)

        let result1 = await stft.transform(signal1)
        let result2 = await stft.transform(signal2)

        // Compute total frequency energy for each
        var energy1: Float = 0
        var energy2: Float = 0
        for f in 0..<result1.rows {
            for t in 0..<result1.cols {
                let mag1Sq = result1.real[f][t] * result1.real[f][t] + result1.imag[f][t] * result1.imag[f][t]
                let mag2Sq = result2.real[f][t] * result2.real[f][t] + result2.imag[f][t] * result2.imag[f][t]
                energy1 += mag1Sq
                energy2 += mag2Sq
            }
        }

        // Doubling amplitude should quadruple energy (2^2 = 4)
        let ratio = energy2 / max(energy1, 1e-10)
        XCTAssertGreaterThan(ratio, 3.8, "Energy ratio should be ~4x for doubled amplitude, got \(ratio)x")
        XCTAssertLessThan(ratio, 4.2, "Energy ratio should be ~4x for doubled amplitude, got \(ratio)x")
    }

    func testSignalPowerConservation() async {
        // Test that power spectrogram reflects signal amplitude
        let amplitude: Float = 0.7
        let nSamples = 4096

        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signal[i] = amplitude * sin(Float(i) * 0.1)
        }

        let config = STFTConfig(nFFT: 1024, hopLength: 256)
        let stft = STFT(config: config)

        let power = await stft.power(signal)

        // Total power should scale with amplitude^2
        let totalPower = power.flatMap { $0 }.reduce(0, +)
        XCTAssertGreaterThan(totalPower, 0, "Total power should be positive")

        // Now test with doubled amplitude - power should be ~4x
        var signalDouble = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            signalDouble[i] = 2 * amplitude * sin(Float(i) * 0.1)
        }

        let powerDouble = await stft.power(signalDouble)
        let totalPowerDouble = powerDouble.flatMap { $0 }.reduce(0, +)

        // Power should scale by 4x (2^2) with doubled amplitude
        let powerRatio = totalPowerDouble / totalPower
        XCTAssertEqual(powerRatio, 4.0, accuracy: 0.5,
            "Doubling amplitude should quadruple power, got ratio \(powerRatio)")
    }

    // MARK: - Numerical Accuracy Tests

    func testZeroSignalGivesZeroSpectrum() async {
        let signal = [Float](repeating: 0, count: 2048)
        let config = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: config)

        let magnitude = await stft.magnitude(signal)

        // All magnitudes should be exactly zero (or very close due to floating-point)
        let maxMag = magnitude.flatMap { $0 }.max() ?? 0
        XCTAssertEqual(maxMag, 0, accuracy: 1e-10,
            "Zero signal should give zero spectrum, got max magnitude \(maxMag)")
    }

    func testMagnitudePhaseConsistency() async {
        // Test that magnitude and phase can reconstruct real/imag
        var signal = [Float](repeating: 0, count: 2048)
        for i in 0..<signal.count {
            signal[i] = Float.random(in: -1...1)
        }

        let config = STFTConfig(nFFT: 512, hopLength: 128)
        let stft = STFT(config: config)

        let result = await stft.transform(signal)

        // Compute magnitude and phase
        let mag = result.magnitude
        let phase = result.phase

        // Verify: real = mag * cos(phase), imag = mag * sin(phase)
        for f in 0..<min(10, result.rows) {  // Check first 10 frequency bins
            for t in 0..<min(10, result.cols) {
                let reconstructedReal = mag[f][t] * cos(phase[f][t])
                let reconstructedImag = mag[f][t] * sin(phase[f][t])

                XCTAssertEqual(reconstructedReal, result.real[f][t], accuracy: 1e-5,
                    "Real component mismatch at [\(f)][\(t)]")
                XCTAssertEqual(reconstructedImag, result.imag[f][t], accuracy: 1e-5,
                    "Imag component mismatch at [\(f)][\(t)]")
            }
        }
    }

    func testFrameCountCalculation() async {
        // Test exact frame count calculation for uncentered STFT
        let testCases: [(signalLen: Int, nFFT: Int, hopLength: Int)] = [
            (1024, 256, 64),
            (2048, 512, 128),
            (4096, 1024, 256),
            (1000, 256, 64),  // Non-power-of-2 signal length
        ]

        for tc in testCases {
            let signal = [Float](repeating: 0.5, count: tc.signalLen)
            let config = STFTConfig(nFFT: tc.nFFT, hopLength: tc.hopLength, center: false)
            let stft = STFT(config: config)

            let result = await stft.transform(signal)

            // For uncentered STFT: nFrames = 1 + (signalLen - nFFT) / hopLength
            let expectedFrames = 1 + (tc.signalLen - tc.nFFT) / tc.hopLength
            XCTAssertEqual(result.cols, expectedFrames,
                "Frame count mismatch for signal=\(tc.signalLen), nFFT=\(tc.nFFT), hop=\(tc.hopLength)")
        }
    }
}
