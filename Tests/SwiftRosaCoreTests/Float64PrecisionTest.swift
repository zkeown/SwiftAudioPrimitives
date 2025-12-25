import SwiftRosaCore
import Foundation
import Testing

@Suite("Float64 Precision Tests")
struct Float64PrecisionTests {

    @Test("Float64 FFT provides better STFT-ISTFT roundtrip precision")
    func testFloat64RoundtripPrecision() async {
        // Create test signal - 1 second of 440Hz sine wave
        let sampleRate = 22050
        let duration = 1.0
        let numSamples = Int(Double(sampleRate) * duration)

        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            signal[i] = sin(2 * Float.pi * 440 * t)
        }

        // Test STFT -> ISTFT roundtrip with Float32
        let config32 = STFTConfig(uncheckedNFFT: 2048, precision: .float32)
        let stft32 = STFT(config: config32)
        let istft32 = ISTFT(config: config32)

        let spec32 = await stft32.transform(signal)
        let recon32 = await istft32.transform(spec32, length: numSamples)

        // Test STFT -> ISTFT roundtrip with Float64
        let config64 = STFTConfig(uncheckedNFFT: 2048, precision: .float64)
        let stft64 = STFT(config: config64)
        let istft64 = ISTFT(config: config64)

        let spec64 = await stft64.transform(signal)
        let recon64 = await istft64.transform(spec64, length: numSamples)

        // Calculate errors
        var error32: Double = 0
        var error64: Double = 0
        var maxError32: Double = 0
        var maxError64: Double = 0

        for i in 0..<min(signal.count, recon32.count, recon64.count) {
            let e32 = abs(Double(signal[i]) - Double(recon32[i]))
            let e64 = abs(Double(signal[i]) - Double(recon64[i]))
            error32 += e32 * e32
            error64 += e64 * e64
            maxError32 = max(maxError32, e32)
            maxError64 = max(maxError64, e64)
        }

        let rmse32 = sqrt(error32 / Double(signal.count))
        let rmse64 = sqrt(error64 / Double(signal.count))

        print("")
        print("STFT -> ISTFT Roundtrip Precision Comparison")
        print("=============================================")
        print("Signal: \(numSamples) samples of 440Hz sine wave")
        print("")
        print("Float32 FFT:")
        print("  RMSE: \(rmse32)")
        print("  Max Error: \(maxError32)")
        print("")
        print("Float64 FFT:")
        print("  RMSE: \(rmse64)")
        print("  Max Error: \(maxError64)")
        print("")
        print("Improvement:")
        print("  RMSE: \(rmse32 / rmse64)x better")
        print("  Max Error: \(maxError32 / maxError64)x better")

        // Float64 should be significantly better
        #expect(rmse64 < rmse32, "Float64 RMSE should be lower than Float32")
        #expect(maxError64 < maxError32, "Float64 max error should be lower than Float32")

        // Both should be reasonably accurate
        #expect(rmse32 < 1e-5, "Float32 RMSE should be < 1e-5")
        #expect(rmse64 < 1e-6, "Float64 RMSE should be < 1e-6 (10x better)")
    }

    @Test("Float64 maintains compatibility with Float32 output")
    func testFloat64OutputCompatibility() async {
        // Create test signal
        let sampleRate = 22050
        let numSamples = 4096

        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            signal[i] = sin(2 * Float.pi * 440 * t) + 0.3 * sin(2 * Float.pi * 880 * t)
        }

        // Float32 and Float64 STFT
        let config32 = STFTConfig(uncheckedNFFT: 1024, precision: .float32)
        let config64 = STFTConfig(uncheckedNFFT: 1024, precision: .float64)

        let stft32 = STFT(config: config32)
        let stft64 = STFT(config: config64)

        let spec32 = await stft32.transform(signal)
        let spec64 = await stft64.transform(signal)

        // Output should have same dimensions
        #expect(spec32.real.count == spec64.real.count, "Same number of frequency bins")
        #expect(spec32.real[0].count == spec64.real[0].count, "Same number of frames")

        // Magnitude should be very similar (both represent the same signal)
        let mag32 = spec32.magnitude
        let mag64 = spec64.magnitude

        var totalDiff: Double = 0
        var maxDiff: Double = 0

        for f in 0..<mag32.count {
            for t in 0..<mag32[f].count {
                let diff = abs(Double(mag32[f][t]) - Double(mag64[f][t]))
                totalDiff += diff
                maxDiff = max(maxDiff, diff)
            }
        }

        let avgDiff = totalDiff / Double(mag32.count * mag32[0].count)

        print("")
        print("Float32 vs Float64 STFT Magnitude Comparison")
        print("=============================================")
        print("Average difference: \(avgDiff)")
        print("Max difference: \(maxDiff)")

        // The difference should be small (due to Float32 vs Float64 FFT precision)
        #expect(avgDiff < 1e-4, "Average magnitude difference should be small")
        #expect(maxDiff < 1e-3, "Max magnitude difference should be small")
    }
}
