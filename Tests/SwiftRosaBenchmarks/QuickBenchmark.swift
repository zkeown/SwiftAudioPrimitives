import XCTest
import SwiftRosaCore
import SwiftRosaEffects
import Foundation

final class QuickBenchmarkTests: XCTestCase {

    func testISTFTBenchmark() async throws {
        print("\n=== ISTFT Benchmark ===")
        let sampleRate: Float = 22050
        let duration: Float = 10.0
        let numSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * .pi * 440 * t)
        }

        let stftConfig = STFTConfig(uncheckedNFFT: 2048, hopLength: 512, windowType: .hann, center: true)
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)
        print("Spectrogram shape: (\(spectrogram.rows), \(spectrogram.cols))")

        let istft = ISTFT(config: stftConfig)
        let start = CFAbsoluteTimeGetCurrent()
        let reconstructed = await istft.transform(spectrogram, length: numSamples)
        let istftTime = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print("ISTFT: \(String(format: "%.1f", istftTime))ms (librosa: 3.9ms)")
        print("Speedup: \(String(format: "%.1fx", 3.9 / istftTime)) vs librosa")
        XCTAssertEqual(reconstructed.count, numSamples)
    }

    func testChromaBenchmark() async throws {
        print("\n=== Chroma STFT Benchmark ===")
        let sampleRate: Float = 22050
        let duration: Float = 10.0
        let numSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * .pi * 440 * t)
        }

        let chromaConfig = ChromagramConfig(
            sampleRate: sampleRate,
            hopLength: 512,
            nChroma: 12,
            useCQT: false,
            norm: .l2,
            nFFT: 4096
        )
        let chromagram = Chromagram(config: chromaConfig)

        // Time STFT separately (4096)
        let stftConfig = STFTConfig(uncheckedNFFT: 4096, hopLength: 512, windowType: .hann, center: true)
        let stft = STFT(config: stftConfig)

        let stftStart = CFAbsoluteTimeGetCurrent()
        let stftMag = await stft.magnitude(signal)
        let stftTime = (CFAbsoluteTimeGetCurrent() - stftStart) * 1000

        // Also time STFT with 2048 for comparison
        let stftConfig2k = STFTConfig(uncheckedNFFT: 2048, hopLength: 512, windowType: .hann, center: true)
        let stft2k = STFT(config: stftConfig2k)
        let stft2kStart = CFAbsoluteTimeGetCurrent()
        _ = await stft2k.magnitude(signal)
        let stft2kTime = (CFAbsoluteTimeGetCurrent() - stft2kStart) * 1000

        // Time filterbank application separately
        let fbStart = CFAbsoluteTimeGetCurrent()
        let chromaFromSTFT = chromagram.fromSTFT(stftMag)
        let fbTime = (CFAbsoluteTimeGetCurrent() - fbStart) * 1000

        // Total time
        let start = CFAbsoluteTimeGetCurrent()
        let chroma = await chromagram.transform(signal)
        let chromaTime = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print("  STFT (4096): \(String(format: "%.1f", stftTime))ms")
        print("  STFT (2048): \(String(format: "%.1f", stft2kTime))ms")
        print("  Filterbank:  \(String(format: "%.1f", fbTime))ms")
        print("Chroma STFT: \(String(format: "%.1f", chromaTime))ms (librosa: 7.2ms)")
        print("Speedup: \(String(format: "%.1fx", 7.2 / chromaTime)) vs librosa")
        XCTAssertEqual(chroma.count, 12)
        XCTAssertEqual(chromaFromSTFT.count, 12)
    }

    func testHPSSBenchmark() async throws {
        print("\n=== HPSS Benchmark ===")
        let sampleRate: Float = 22050
        let duration: Float = 10.0
        let numSamples = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * .pi * 440 * t)
        }

        let hpss = HPSS.standard  // 31x31 kernel
        let start = CFAbsoluteTimeGetCurrent()
        let result = await hpss.separate(signal)
        let hpssTime = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print("HPSS (31x31): \(String(format: "%.1f", hpssTime))ms (librosa: 215ms)")
        print("Speedup: \(String(format: "%.1fx", 215.0 / hpssTime)) vs librosa")
        XCTAssertEqual(result.harmonic.count, numSamples)
    }
}
