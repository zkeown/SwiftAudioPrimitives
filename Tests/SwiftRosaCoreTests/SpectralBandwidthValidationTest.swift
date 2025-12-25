import XCTest
import Foundation
@testable import SwiftRosaCore

/// Validate spectral bandwidth against librosa reference values
final class SpectralBandwidthValidationTest: XCTestCase {

    func testSpectralBandwidthAgainstLibrosa() async throws {
        // Generate test signal using TestSignalGenerator (Double precision)
        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 1.0)

        let config = SpectralFeaturesConfig(sampleRate: 22050, nFFT: 2048, hopLength: 512)
        let features = SpectralFeatures(config: config)

        let bw = await features.bandwidth(signal)

        // Check frame 5 (steady state, not affected by edge padding)
        let frameIdx = 5

        // librosa reference: 11.86 Hz for 440Hz pure sine
        let expectedBW: Float = 11.86
        // Use narrowband tolerance for pure tones (sensitive to FFT noise)
        let tolerance: Float = Float(ValidationTolerances.spectralBandwidthNarrowband)

        let error = abs(bw[frameIdx] - expectedBW) / expectedBW
        print("Spectral bandwidth validation:")
        print("  Frame \(frameIdx) bandwidth: \(String(format: "%.4f", bw[frameIdx])) Hz")
        print("  Expected (librosa): \(expectedBW) Hz")
        print("  Relative error: \(String(format: "%.2e", error))")

        XCTAssertLessThan(error, tolerance, "Bandwidth should match librosa within tolerance")
    }

    func testSpectralCentroidAgainstLibrosa() async throws {
        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 1.0)

        let config = SpectralFeaturesConfig(sampleRate: 22050, nFFT: 2048, hopLength: 512)
        let features = SpectralFeatures(config: config)

        let cent = await features.centroid(signal)

        let frameIdx = 5
        // librosa reference: 440.25 Hz for 440Hz pure sine (slight offset due to bin spacing)
        let expectedCent: Float = 440.25
        let tolerance: Float = Float(ValidationTolerances.spectralCentroid)

        let error = abs(cent[frameIdx] - expectedCent) / expectedCent
        print("\nSpectral centroid validation:")
        print("  Frame \(frameIdx) centroid: \(String(format: "%.2f", cent[frameIdx])) Hz")
        print("  Expected (librosa): \(expectedCent) Hz")
        print("  Relative error: \(String(format: "%.2e", error))")

        XCTAssertLessThan(error, tolerance, "Centroid should match librosa within tolerance")
    }

    func testSpectralRolloffAgainstLibrosa() async throws {
        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 1.0)

        let config = SpectralFeaturesConfig(sampleRate: 22050, nFFT: 2048, hopLength: 512)
        let features = SpectralFeatures(config: config)

        let roll = await features.rolloff(signal)

        let frameIdx = 5
        // For a pure tone, rolloff should be near the signal frequency
        let expectedRoll: Float = 452.20  // bin 42 * 22050/2048 (next bin above 440Hz contains 85% power)
        let tolerance: Float = Float(ValidationTolerances.spectralRolloff)

        let error = abs(roll[frameIdx] - expectedRoll) / expectedRoll
        print("\nSpectral rolloff validation:")
        print("  Frame \(frameIdx) rolloff: \(String(format: "%.2f", roll[frameIdx])) Hz")
        print("  Expected (librosa): ~\(expectedRoll) Hz")
        print("  Relative error: \(String(format: "%.2e", error))")

        // Rolloff can vary more depending on exact energy distribution
        XCTAssertLessThan(error, tolerance, "Rolloff should match librosa within tolerance")
    }

    func testSTFTNoiseFloorWithDoubleSignal() async throws {
        // Verify that the signal generation with Double precision produces low noise
        let signal = TestSignalGenerator.sine(frequency: 440, sampleRate: 22050, duration: 1.0)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512, center: true, padMode: .constant(0)))
        let mag = await stft.magnitude(signal)

        let frameIdx = 5
        let signalBin = 41
        let noiseBin = 893

        let signalMag = mag[signalBin][frameIdx]
        let noiseMag = mag[noiseBin][frameIdx]
        let snr = signalMag / noiseMag

        print("\nSTFT noise floor check:")
        print("  Signal bin \(signalBin) magnitude: \(signalMag)")
        print("  Noise bin \(noiseBin) magnitude: \(String(format: "%.9f", noiseMag))")
        print("  Signal/Noise ratio: \(String(format: "%.0f", snr))")

        // With Double precision signal generation, noise should be ~1e-6 or less
        // SNR should be at least 1e8 (100 million to 1)
        XCTAssertGreaterThan(snr, 1e7, "SNR should be very high with Double precision signal")
        XCTAssertLessThan(noiseMag, 1e-5, "Noise floor should be very low")
    }
}
