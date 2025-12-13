import XCTest
@testable import SwiftRosaEffects
@testable import SwiftRosaCore

final class ICQTTests: XCTestCase {

    // MARK: - Basic ICQT Tests

    func testICQTBasic() async {
        // Create CQT for forward transform
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 36,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)

        // Generate simple test signal
        let duration: Float = 1.0
        let sampleRate: Float = 22050
        let nSamples = Int(duration * sampleRate)
        var signal = [Float](repeating: 0, count: nSamples)

        // Create sine wave at A4 (440 Hz)
        for i in 0..<nSamples {
            let t = Float(i) / sampleRate
            signal[i] = sin(2 * Float.pi * 440 * t) * 0.5
        }

        // Forward CQT
        let cqtResult = await cqt.transform(signal)

        // Inverse CQT using the CQT's config
        let icqt = ICQT(cqtConfig: cqtConfig)

        let reconstructed = await icqt.transform(cqtResult, length: nSamples)

        // Should have correct length
        XCTAssertEqual(reconstructed.count, nSamples)

        // Signal should have some energy
        let energy = reconstructed.reduce(0) { $0 + $1 * $1 }
        XCTAssertGreaterThan(energy, 0)
    }

    func testICQTPreservesLength() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)

        let targetLength = 22050  // 1 second at 22050 Hz
        var signal = [Float](repeating: 0, count: targetLength)
        for i in 0..<targetLength {
            signal[i] = Float.random(in: -0.5...0.5)
        }

        let cqtResult = await cqt.transform(signal)

        let icqt = ICQT(cqtConfig: cqtConfig)

        let reconstructed = await icqt.transform(cqtResult, length: targetLength)

        XCTAssertEqual(reconstructed.count, targetLength,
            "ICQT should produce signal of specified length")
    }

    func testICQTFromMagnitude() async {
        // Test Griffin-Lim style reconstruction from magnitude only
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)

        let nSamples = 11025  // 0.5 seconds
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / 22050
            signal[i] = sin(2 * Float.pi * 440 * t) * 0.3 + sin(2 * Float.pi * 880 * t) * 0.2
        }

        let cqtResult = await cqt.transform(signal)

        // Extract magnitude
        let magnitude = cqtResult.magnitude

        // Reconstruct from magnitude using ICQT
        let icqt = ICQT(cqtConfig: cqtConfig)

        let reconstructed = await icqt.fromMagnitude(magnitude, length: nSamples, iterations: 16)

        XCTAssertEqual(reconstructed.count, nSamples)

        // Reconstructed signal should have energy
        let energy = reconstructed.reduce(0) { $0 + $1 * $1 }
        XCTAssertGreaterThan(energy, 0)
    }

    func testICQTIterationsEffect() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)

        let nSamples = 8192
        var signal = [Float](repeating: 0, count: nSamples)
        for i in 0..<nSamples {
            let t = Float(i) / 22050
            signal[i] = sin(2 * Float.pi * 440 * t) * 0.5
        }

        let cqtResult = await cqt.transform(signal)
        let magnitude = cqtResult.magnitude

        let icqt = ICQT(cqtConfig: cqtConfig)

        // More iterations should generally improve reconstruction
        let reconstructed8 = await icqt.fromMagnitude(magnitude, length: nSamples, iterations: 8)
        let reconstructed32 = await icqt.fromMagnitude(magnitude, length: nSamples, iterations: 32)

        // Both should produce valid output
        XCTAssertEqual(reconstructed8.count, nSamples)
        XCTAssertEqual(reconstructed32.count, nSamples)
    }

    // MARK: - Configuration Tests

    func testICQTConfigFromCQT() {
        let cqtConfig = CQTConfig(
            sampleRate: 44100,
            hopLength: 256,
            nBins: 72,
            binsPerOctave: 24
        )

        let icqt = ICQT(cqtConfig: cqtConfig)

        XCTAssertEqual(icqt.config.sampleRate, 44100)
        XCTAssertEqual(icqt.config.hopLength, 256)
        XCTAssertEqual(icqt.config.nBins, 72)
        XCTAssertEqual(icqt.config.binsPerOctave, 24)
    }

    // MARK: - CQT Extension Tests

    func testCQTMakeInverse() async {
        let cqt = CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 36,
            binsPerOctave: 12
        ))

        let icqt = cqt.makeInverse()

        // Config should match
        XCTAssertEqual(icqt.config.sampleRate, cqt.config.sampleRate)
        XCTAssertEqual(icqt.config.hopLength, cqt.config.hopLength)
        XCTAssertEqual(icqt.config.nBins, cqt.config.nBins)
        XCTAssertEqual(icqt.config.binsPerOctave, cqt.config.binsPerOctave)
    }

    func testMakeInverseProducesOutput() async {
        let cqt = CQT(config: CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        ))

        let signal = (0..<11025).map { sin(2 * Float.pi * 440 * Float($0) / 22050) * 0.5 }
        let cqtResult = await cqt.transform(signal)

        let icqt = cqt.makeInverse()
        let reconstructed = await icqt.transform(cqtResult, length: 11025)

        XCTAssertEqual(reconstructed.count, 11025)
    }

    // MARK: - Edge Cases

    func testICQTEmptyInput() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let icqt = ICQT(cqtConfig: cqtConfig)

        let emptyComplex = ComplexMatrix(real: [], imag: [])
        let result = await icqt.transform(emptyComplex, length: 1000)

        // Should handle gracefully
        XCTAssertEqual(result.count, 0)
    }

    func testICQTVeryShortSignal() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)
        let icqt = ICQT(cqtConfig: cqtConfig)

        // Very short signal (just 2048 samples)
        let signal = (0..<2048).map { sin(2 * Float.pi * 440 * Float($0) / 22050) * 0.5 }
        let cqtResult = await cqt.transform(signal)
        let reconstructed = await icqt.transform(cqtResult, length: 2048)

        // Should produce output of correct length
        XCTAssertEqual(reconstructed.count, 2048)
    }

    func testICQTZeroIterations() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)

        let signal = (0..<4096).map { sin(2 * Float.pi * 440 * Float($0) / 22050) * 0.5 }
        let cqtResult = await cqt.transform(signal)
        let magnitude = cqtResult.magnitude

        let icqt = ICQT(cqtConfig: cqtConfig)

        // Zero iterations should still produce output (just one pass)
        let reconstructed = await icqt.fromMagnitude(magnitude, length: 4096, iterations: 0)

        // May not be great quality but should have correct length
        XCTAssertEqual(reconstructed.count, 4096)
    }

    func testICQTDefaultIterations() async {
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: 512,
            nBins: 24,
            binsPerOctave: 12
        )
        let cqt = CQT(config: cqtConfig)
        let icqt = ICQT(cqtConfig: cqtConfig)

        let signal = (0..<8192).map { sin(2 * Float.pi * 440 * Float($0) / 22050) * 0.5 }
        let cqtResult = await cqt.transform(signal)

        // Default iterations (32)
        let reconstructed = await icqt.transform(cqtResult, length: 8192)

        XCTAssertEqual(reconstructed.count, 8192)

        // Should have energy
        let energy = reconstructed.reduce(0) { $0 + $1 * $1 }
        XCTAssertGreaterThan(energy, 0)
    }
}
