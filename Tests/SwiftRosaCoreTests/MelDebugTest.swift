import XCTest
@testable import SwiftRosaCore

final class MelDebugTest: XCTestCase {
    func loadReferenceSignal(_ name: String) throws -> [Float] {
        let refPath = findReferencePath()
        let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let signals = json["test_signals"] as? [String: Any],
              let signalData = signals[name] as? [String: Any],
              let signalValues = signalData["data"] as? [Double] else {
            throw NSError(domain: "", code: 1, userInfo: [NSLocalizedDescriptionKey: "Signal not found"])
        }
        return signalValues.map { Float($0) }
    }

    func findReferencePath() -> String {
        let possiblePaths = [
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceData/librosa_reference.json")
                .path,
            FileManager.default.currentDirectoryPath + "/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json",
        ]
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return possiblePaths[0]
    }

    func loadMelReference(_ name: String) throws -> [[Double]] {
        let refPath = findReferencePath()
        let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let melSpec = json["mel_spectrogram"] as? [String: Any],
              let sigRef = melSpec[name] as? [String: Any],
              let mel = sigRef["mel_spectrogram"] as? [[Double]] else {
            throw NSError(domain: "", code: 1, userInfo: [NSLocalizedDescriptionKey: "Mel reference not found"])
        }
        return mel
    }

    func testPowerSpectrogramDebug() async throws {
        // Test with impulse first (no window, rectangular)
        var impulse = [Float](repeating: 0, count: 2048)
        impulse[0] = 1.0

        // Use rectangular window to match librosa's window='ones'
        let stftImpulse = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: 2048,
            windowType: .rectangular,
            center: false,
            padMode: .constant(0)
        ))

        let powerImpulse = await stftImpulse.power(impulse)
        print("=== Impulse test (unit impulse, no window) ===")
        print("Swift power of impulse (first 5 bins):", powerImpulse.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        print("librosa power of impulse (first 5 bins): 1.0000, 1.0000, 1.0000, 1.0000, 1.0000")
        print("Scale factor: \(powerImpulse[0][0])")

        // Now test with actual signal
        let signal = try loadReferenceSignal("sine_440hz")

        let stft = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: true,
            padMode: .reflect
        ))

        let powerSpec = await stft.power(signal)
        print("\n=== Sine wave test (with hann window) ===")
        print("Swift power spec shape: freqs=\(powerSpec.count), frames=\(powerSpec.first?.count ?? 0)")
        print("Swift power spec (first 5 bins, first frame):", powerSpec.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))

        // librosa power spec: [63.450577 63.596638 63.756046 64.213264 64.68579 ]
        print("librosa power spec (first 5 bins, first frame): 63.4506, 63.5966, 63.7560, 64.2133, 64.6858")

        let swiftVal = Double(powerSpec[0][0])
        let librosaVal = 63.450577
        print("\nScale ratio (librosa/swift): \(librosaVal / swiftVal)")
    }

    func testMelSpectrogramDebug() async throws {
        let signal = try loadReferenceSignal("sine_440hz")
        let expectedMel = try loadMelReference("sine_440hz")

        print("Signal length: \(signal.count)")
        print("Expected mel shape: frames=\(expectedMel.count), mels=\(expectedMel.first?.count ?? 0)")

        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )

        let result = await melSpec.transform(signal)
        print("Swift mel shape: mels=\(result.count), frames=\(result.first?.count ?? 0)")

        // Compare first frame
        print("\n--- First frame comparison (frame 0) ---")
        print("librosa (first 10 mels):", expectedMel[0].prefix(10).map { String(format: "%.4f", $0) }.joined(separator: ", "))
        print("Swift   (first 10 mels):", result.prefix(10).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))

        // Compare middle frame
        let midFrame = expectedMel.count / 2
        print("\n--- Middle frame comparison (frame \(midFrame)) ---")
        print("librosa (first 10 mels):", expectedMel[midFrame].prefix(10).map { String(format: "%.4f", $0) }.joined(separator: ", "))
        print("Swift   (first 10 mels):", result.prefix(10).map { String(format: "%.4f", $0[midFrame]) }.joined(separator: ", "))

        // Calculate scale factor
        var ratios: [Double] = []
        for melIdx in 0..<min(20, result.count) {
            let expected = expectedMel[0][melIdx]
            let actual = Double(result[melIdx][0])
            if expected > 1e-6 && actual > 1e-6 {
                ratios.append(expected / actual)
            }
        }
        let avgRatio = ratios.reduce(0, +) / Double(ratios.count)
        print("\n--- Scale Analysis ---")
        print("Expected / Actual ratios (first 20 mels):", ratios.prefix(10).map { String(format: "%.3f", $0) }.joined(separator: ", "))
        print("Average ratio: \(avgRatio)")
    }

    func testMelFilterbankDebug() async throws {
        // Check our mel filterbank against librosa
        let melFilterbank = MelFilterbank(
            nMels: 128,
            nFFT: 2048,
            sampleRate: 22050,
            fMin: 0,
            fMax: nil,
            htk: false,
            norm: .slaney
        )

        let filters = melFilterbank.filters()
        print("Swift mel filterbank shape: mels=\(filters.count), freqs=\(filters.first?.count ?? 0)")
        print("Swift filter 0 sum: \(filters[0].reduce(0, +))")
        print("Swift filter 0 max: \(filters[0].max() ?? 0)")
        print("Swift filter 0 first 50 (non-zero):", filters[0].prefix(50).filter { $0 > 0 }.map { String(format: "%.6f", $0) }.joined(separator: ", "))

        // librosa values:
        // mel_basis[0] sum: 0.09034588
        // mel_basis[0] max: 0.032365706
        print("\nlibrosa filter 0 sum: 0.09034588")
        print("librosa filter 0 max: 0.032365706")
    }

    func testFFTScaling() async throws {
        // Test FFT scaling with simple signals

        // First test STFT magnitude to compare with librosa
        let signal = try loadReferenceSignal("sine_440hz")

        let stftMag = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: true,
            padMode: .reflect
        ))

        let magnitude = await stftMag.magnitude(signal)
        let power = await stftMag.power(signal)

        print("Reference signal STFT (center=true):")
        print("  Swift magnitude[0,0]: \(magnitude[0][0])")
        print("  librosa magnitude[0,0]: 7.965587")
        print("  Magnitude ratio: \(magnitude[0][0] / 7.965587)")
        print("")
        print("  Swift power[0,0]: \(power[0][0])")
        print("  librosa power[0,0]: 63.450577")
        print("  Power ratio: \(power[0][0] / 63.450577)")

        // Test without centering
        let stftNoCenter = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        ))

        let magnitudeNoCenter = await stftNoCenter.magnitude(signal)
        let powerNoCenter = await stftNoCenter.power(signal)

        // Frame 1 should have actual signal (frame 0 starts from beginning)
        if magnitudeNoCenter[0].count > 1 && powerNoCenter[0].count > 1 {
            print("\nReference signal STFT (center=false, frame 1):")
            print("  Swift magnitude[0,1]: \(magnitudeNoCenter[0][1])")
            print("  Swift power[0,1]: \(powerNoCenter[0][1])")
        }

        // Now test with simple signals
        print("\n--- Simple signal tests ---")

        // Test 1: Unit impulse (N=8)
        var impulse8 = [Float](repeating: 0, count: 8)
        impulse8[0] = 1.0

        let stft8 = STFT(config: STFTConfig(
            nFFT: 8,
            hopLength: 8,
            winLength: 8,
            windowType: .rectangular,
            center: false,
            padMode: .constant(0)
        ))

        let power8 = await stft8.power(impulse8)
        print("Unit impulse (N=8) power:")
        print("  Swift:", power8.map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        print("  numpy: 1.0000, 1.0000, 1.0000, 1.0000, 1.0000")

        // Test 2: Constant signal
        let constant8 = [Float](repeating: 1.0, count: 8)
        let powerConst8 = await stft8.power(constant8)
        print("\nConstant signal [1,1,1,1,1,1,1,1] power:")
        print("  Swift DC:", powerConst8[0][0])
        print("  numpy DC: 64.0 (N^2)")
        print("  Ratio: \(powerConst8[0][0] / 64.0)")

        // Test 3: Cosine at frequency 1
        let t8 = (0..<8).map { Float($0) / 8.0 }
        let cos8 = t8.map { cos(2 * Float.pi * 1 * $0) }
        print("\nCosine signal:", cos8.map { String(format: "%.4f", $0) }.joined(separator: ", "))
        let powerCos8 = await stft8.power(cos8)
        print("Cosine power:")
        print("  Swift:", powerCos8.map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        print("  numpy: ~0, 16, ~0, ~0, ~0 (power at bin 1 = 16)")
        print("  Swift bin 1 / numpy bin 1:", powerCos8[1][0] / 16.0)

        // Test 4: N=2048 sine wave to match our real test case
        let sr: Float = 22050
        let nFFT = 2048
        let t2048 = (0..<nFFT).map { Float($0) / sr }
        let sine2048 = t2048.map { sin(2 * Float.pi * 440 * $0) }

        let stft2048 = STFT(config: STFTConfig(
            nFFT: nFFT,
            hopLength: nFFT,
            winLength: nFFT,
            windowType: .hann,  // Use hann window like our real case
            center: false,
            padMode: .constant(0)
        ))

        let power2048 = await stft2048.power(sine2048)
        print("\n440 Hz sine wave (N=2048, hann window) power:")
        print("  Swift bin 0 (DC):", power2048[0][0])

        // Find peak bin
        var maxPower: Float = 0
        var maxBin = 0
        for (i, p) in power2048.enumerated() {
            if p[0] > maxPower {
                maxPower = p[0]
                maxBin = i
            }
        }
        print("  Swift peak: bin \(maxBin), power \(maxPower)")
        print("  Expected peak bin: \(Int(440 * Float(nFFT) / sr))")
    }

    func testManualMelComputation() async throws {
        let signal = try loadReferenceSignal("sine_440hz")

        // Test 1: With centering (like librosa default)
        let stftCentered = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: true,
            padMode: .reflect
        ))

        let powerCentered = await stftCentered.power(signal)
        print("CENTERED (center=true):")
        print("  Power spec shape: freqs=\(powerCentered.count), frames=\(powerCentered.first?.count ?? 0)")
        print("  Power spec first 5 bins, frame 0:", powerCentered.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        print("  librosa: 63.4506, 63.5966, 63.7560, 64.2133, 64.6858")
        print("  Ratio: \(powerCentered[0][0] / 63.450577)")

        // Test 2: Without centering
        let stftNoCenter = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: false,
            padMode: .constant(0)
        ))

        let powerNoCenter = await stftNoCenter.power(signal)
        print("\nNO CENTERING (center=false):")
        print("  Power spec shape: freqs=\(powerNoCenter.count), frames=\(powerNoCenter.first?.count ?? 0)")
        print("  Power spec first 5 bins, frame 0:", powerNoCenter.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))

        // Now use the standard config
        let stft = STFT(config: STFTConfig(
            nFFT: 2048,
            hopLength: 512,
            winLength: nil,
            windowType: .hann,
            center: true,
            padMode: .reflect
        ))

        let powerSpec = await stft.power(signal)
        print("\nUsing standard config for rest of test:")
        print("Power spec shape: freqs=\(powerSpec.count), frames=\(powerSpec.first?.count ?? 0)")
        print("Power spec first 5 bins, frame 0:", powerSpec.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        // librosa: Power at bin 0, frame 0: 253.8023

        // Get mel filterbank
        let melFilterbank = MelFilterbank(
            nMels: 128,
            nFFT: 2048,
            sampleRate: 22050,
            fMin: 0,
            fMax: nil,
            htk: false,
            norm: .slaney
        )

        let filters = melFilterbank.filters()

        // Compute mel spectrogram manually (filters @ powerSpec)
        // filters: [nMels][nFreqs]
        // powerSpec: [nFreqs][nFrames]
        // result: [nMels][nFrames]
        let nMels = filters.count
        let nFreqs = powerSpec.count
        let nFrames = powerSpec[0].count

        var manualMel = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: nMels)
        for m in 0..<nMels {
            for t in 0..<nFrames {
                var sum: Float = 0
                for f in 0..<min(nFreqs, filters[m].count) {
                    sum += filters[m][f] * powerSpec[f][t]
                }
                manualMel[m][t] = sum
            }
        }

        print("\nManual mel spec (frame 0, first 5 mels):", manualMel.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))
        print("librosa mel spec (frame 0, first 5 mels): 5.7827, 6.1022, 6.3472, 6.4308, 7.3432")

        // Now use MelSpectrogram class
        let melSpec = MelSpectrogram(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128
        )

        let classMel = await melSpec.transform(signal)
        print("\nMelSpectrogram class (frame 0, first 5 mels):", classMel.prefix(5).map { String(format: "%.4f", $0[0]) }.joined(separator: ", "))

        print("\nRatio (manual/librosa):", manualMel[0][0] / 5.7827)
        print("Ratio (class/librosa):", classMel[0][0] / 5.7827)
    }
}
