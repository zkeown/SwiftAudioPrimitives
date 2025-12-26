import XCTest
@testable import SwiftRosaCore
@testable import SwiftRosaEffects

/// Diagnostic tests to identify where energy is lost in phase vocoder pipeline
final class PhaseVocoderDiagnosticTest: XCTestCase {

    /// Generate a pure sine wave
    private func generateSine(frequency: Float = 440, sampleRate: Int = 22050, duration: Float = 1.0) -> [Float] {
        let nSamples = Int(Float(sampleRate) * duration)
        return (0..<nSamples).map { i in
            let t = Float(i) / Float(sampleRate)
            return sin(2 * Float.pi * frequency * t)
        }
    }

    /// Compute total energy of signal
    private func totalEnergy(_ signal: [Float]) -> Float {
        signal.reduce(0) { $0 + $1 * $1 }
    }

    /// Compute STFT energy
    private func stftEnergy(_ spec: ComplexMatrix) -> Float {
        var energy: Float = 0
        for f in 0..<spec.rows {
            for t in 0..<spec.cols {
                energy += spec.real[f][t] * spec.real[f][t] + spec.imag[f][t] * spec.imag[f][t]
            }
        }
        return energy
    }

    /// Test ISTFT directly with fewer frames
    func testISTFTWithFewerFrames() async throws {
        let signal = generateSine()
        let signalEnergy = totalEnergy(signal)
        print("\n=== ISTFT With Fewer Frames Test ===")
        print("Input signal energy: \(signalEnergy)")

        let config = STFTConfig(
            uncheckedNFFT: 2048,
            hopLength: 512,
            winLength: 2048,
            windowType: .hann,
            center: true,
            padMode: .reflect
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Forward STFT
        let spec = await stft.transform(signal)
        let fullStftE = stftEnergy(spec)
        print("Full STFT: (\(spec.rows), \(spec.cols)), energy=\(fullStftE)")

        // Reconstruct from full spectrogram
        let reconFull = await istft.transform(spec)
        let reconFullE = totalEnergy(reconFull)
        print("Reconstructed from full: length=\(reconFull.count), energy=\(reconFullE)")

        // Take just the first half of the frames
        let halfCols = spec.cols / 2
        var halfReal = [[Float]](repeating: [], count: spec.rows)
        var halfImag = [[Float]](repeating: [], count: spec.rows)
        for f in 0..<spec.rows {
            halfReal[f] = Array(spec.real[f].prefix(halfCols))
            halfImag[f] = Array(spec.imag[f].prefix(halfCols))
        }
        let halfSpec = ComplexMatrix(real: halfReal, imag: halfImag)
        let halfStftE = stftEnergy(halfSpec)

        print("\nHalf STFT: (\(halfSpec.rows), \(halfSpec.cols)), energy=\(halfStftE)")
        print("STFT energy ratio: \(halfStftE / fullStftE)")

        // Reconstruct from half spectrogram
        let reconHalf = await istft.transform(halfSpec)
        let reconHalfE = totalEnergy(reconHalf)

        print("Reconstructed from half: length=\(reconHalf.count), energy=\(reconHalfE)")
        print("Recon energy ratio: \(reconHalfE / reconFullE)")
        print("Expected ratio: ~0.5")

        // The issue should be visible here if ISTFT is the problem
        XCTAssertGreaterThan(reconHalfE / reconFullE, 0.4, "ISTFT energy ratio too low for fewer frames")

        // Now test with phase-vocoder stretched spectrogram
        print("\n--- Comparing with phase-vocoder stretched ---")
        let vocoder = PhaseVocoder(config: PhaseVocoderConfig(nFFT: 2048, hopLength: 512))
        let stretchedSpec = vocoder.stretchSpectrogram(spec, rate: 2.0)
        let stretchedStftE = stftEnergy(stretchedSpec)

        print("Stretched STFT: (\(stretchedSpec.rows), \(stretchedSpec.cols)), energy=\(stretchedStftE)")
        print("Stretched STFT energy ratio: \(stretchedStftE / fullStftE)")

        // Check if the stretched spectrogram has similar properties to the half spectrogram
        print("\nComparing bin-by-bin magnitudes for DOMINANT bins (40,41,42), first 5 frames:")
        for f in [40, 41, 42] {
            print("  Freq bin \(f) (~\(Float(f) * 22050 / 2048) Hz):")
            for t in 0..<min(5, halfSpec.cols) {
                let halfMag = sqrt(halfSpec.real[f][t] * halfSpec.real[f][t] + halfSpec.imag[f][t] * halfSpec.imag[f][t])
                let stretchMag = sqrt(stretchedSpec.real[f][t] * stretchedSpec.real[f][t] + stretchedSpec.imag[f][t] * stretchedSpec.imag[f][t])
                print("    t=\(t): half=\(String(format: "%.2f", halfMag)), stretched=\(String(format: "%.2f", stretchMag))")
            }
        }

        // Also check total energy per frame
        print("\nEnergy per frame:")
        for t in 0..<min(5, halfSpec.cols) {
            var halfFrameE: Float = 0
            var stretchFrameE: Float = 0
            for f in 0..<spec.rows {
                halfFrameE += halfSpec.real[f][t] * halfSpec.real[f][t] + halfSpec.imag[f][t] * halfSpec.imag[f][t]
                stretchFrameE += stretchedSpec.real[f][t] * stretchedSpec.real[f][t] + stretchedSpec.imag[f][t] * stretchedSpec.imag[f][t]
            }
            print("  t=\(t): half=\(String(format: "%.2f", halfFrameE)), stretched=\(String(format: "%.2f", stretchFrameE))")
        }

        // Compare phases for dominant bin
        print("\nPhases for bin 41 (440 Hz) - ORIGINAL STFT first:")
        for t in 0..<min(8, spec.cols) {
            let origPhase = atan2(spec.imag[41][t], spec.real[41][t])
            let origMag = sqrt(spec.real[41][t] * spec.real[41][t] + spec.imag[41][t] * spec.imag[41][t])
            print("  orig t=\(t): phase=\(String(format: "%.4f", origPhase)), mag=\(String(format: "%.2f", origMag))")
        }

        print("\nPhases for bin 41 (440 Hz) - half vs stretched:")
        for t in 0..<min(8, halfSpec.cols) {
            let halfPhase = atan2(halfSpec.imag[41][t], halfSpec.real[41][t])
            let stretchPhase = atan2(stretchedSpec.imag[41][t], stretchedSpec.real[41][t])
            print("  t=\(t): half=\(String(format: "%.4f", halfPhase)), stretched=\(String(format: "%.4f", stretchPhase))")
        }

        // Compare phases for all dominant bins at frame 1
        print("\nPhases at t=1 for dominant bins:")
        for f in [39, 40, 41, 42, 43] {
            let halfPhase = atan2(halfSpec.imag[f][1], halfSpec.real[f][1])
            let stretchPhase = atan2(stretchedSpec.imag[f][1], stretchedSpec.real[f][1])
            print("  bin \(f): half=\(String(format: "%.4f", halfPhase)), stretched=\(String(format: "%.4f", stretchPhase))")
        }

        // Reconstruct from stretched spectrogram
        let reconStretched = await istft.transform(stretchedSpec)
        let reconStretchedE = totalEnergy(reconStretched)

        print("\nReconstructed from stretched: length=\(reconStretched.count), energy=\(reconStretchedE)")
        print("Stretched recon energy ratio: \(reconStretchedE / reconFullE)")
        print("Expected: ~0.5, like half spectrogram")

        // Compare first few samples
        print("\nFirst 20 samples of reconstructed signals:")
        print("  Half spec recon: \(reconHalf.prefix(20).map { String(format: "%.4f", $0) }.joined(separator: ", "))")
        print("  Stretched recon: \(reconStretched.prefix(20).map { String(format: "%.4f", $0) }.joined(separator: ", "))")
    }

    /// Test STFT matches librosa
    func testSTFTMatchesLibrosa() async throws {
        let signal = generateSine()
        print("\n=== Checking if Swift STFT matches librosa ===")

        let config = STFTConfig(
            uncheckedNFFT: 2048,
            hopLength: 512,
            winLength: 2048,
            windowType: .hann,
            center: true,
            padMode: .reflect
        )

        let stft = STFT(config: config)
        let spec = await stft.transform(signal)

        // librosa's expected phases for bin 41 (from Python trace):
        // frame 0: phase=1.4544, mag=254.82
        // frame 1: phase=2.9088, mag=461.26
        // frame 2: phase=-1.9883, mag=506.20

        print("Swift STFT for bin 41:")
        for t in 0..<5 {
            let phase = atan2(spec.imag[41][t], spec.real[41][t])
            let mag = sqrt(spec.real[41][t] * spec.real[41][t] + spec.imag[41][t] * spec.imag[41][t])
            print("  frame \(t): phase=\(String(format: "%.4f", phase)), mag=\(String(format: "%.2f", mag))")
        }

        print("\nExpected (from librosa):")
        print("  frame 0: phase=1.4544, mag=254.82")
        print("  frame 1: phase=2.9088, mag=461.26")
        print("  frame 2: phase=-1.9883, mag=506.20")
    }

    func testDiagnosePipelineEnergy() async throws {
        let signal = generateSine()
        let signalEnergy = totalEnergy(signal)
        print("\n=== Phase Vocoder Energy Diagnostic ===")
        print("Input signal energy: \(signalEnergy)")
        print("Input signal length: \(signal.count)")

        // Create STFT/ISTFT with matching config
        let config = STFTConfig(
            uncheckedNFFT: 2048,
            hopLength: 512,
            winLength: 2048,
            windowType: .hann,
            center: true,
            padMode: .reflect
        )

        let stft = STFT(config: config)
        let istft = ISTFT(config: config)

        // Step 1: Forward STFT
        let spec = await stft.transform(signal)
        let stftE = stftEnergy(spec)
        print("\nAfter STFT:")
        print("  Shape: (\(spec.rows), \(spec.cols))")
        print("  STFT energy: \(stftE)")

        // Step 2: Reconstruct without stretching
        let reconstructed = await istft.transform(spec)
        let reconEnergy = totalEnergy(reconstructed)
        print("\nAfter ISTFT (no stretching):")
        print("  Length: \(reconstructed.count)")
        print("  Energy: \(reconEnergy)")
        print("  Ratio to input: \(reconEnergy / signalEnergy)")

        // Step 3: Phase vocoder for different rates
        let vocoder = PhaseVocoder(config: PhaseVocoderConfig(nFFT: 2048, hopLength: 512))

        for rate: Float in [0.5, 2.0] {
            print("\n--- Rate = \(rate) ---")

            // Stretch spectrogram
            let stretchedSpec = vocoder.stretchSpectrogram(spec, rate: rate)
            let stretchedStftE = stftEnergy(stretchedSpec)
            let frameRatio = Float(stretchedSpec.cols) / Float(spec.cols)
            let stftEnergyRatio = stretchedStftE / stftE

            print("After stretchSpectrogram:")
            print("  Shape: (\(stretchedSpec.rows), \(stretchedSpec.cols))")
            print("  Frame ratio: \(String(format: "%.4f", frameRatio))")
            print("  STFT energy: \(stretchedStftE)")
            print("  STFT energy ratio: \(String(format: "%.4f", stftEnergyRatio))")
            print("  Energy/Frame ratio: \(String(format: "%.4f", stftEnergyRatio / frameRatio))")

            // Reconstruct
            let expectedLength = Int(round(Float(signal.count) / rate))
            let stretched = await istft.transform(stretchedSpec, length: expectedLength)
            let stretchedEnergy = totalEnergy(stretched)
            let expectedEnergy = signalEnergy / rate
            let energyRatio = stretchedEnergy / expectedEnergy

            print("After ISTFT:")
            print("  Length: \(stretched.count) (expected: \(expectedLength))")
            print("  Energy: \(stretchedEnergy)")
            print("  Expected energy (input/rate): \(expectedEnergy)")
            print("  Energy ratio: \(String(format: "%.4f", energyRatio))")

            // Also test the full timeStretch function
            let fullStretched = await vocoder.timeStretch(signal, rate: rate)
            let fullEnergy = totalEnergy(fullStretched)
            print("Full timeStretch:")
            print("  Length: \(fullStretched.count)")
            print("  Energy: \(fullEnergy)")
            print("  Energy ratio: \(String(format: "%.4f", fullEnergy / expectedEnergy))")
        }
    }
}
