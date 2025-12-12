import XCTest
@testable import SwiftRosaEffects
@testable import SwiftRosaCore

final class CQTDebugTest: XCTestCase {
    func testCQTValues() async throws {
        let possiblePaths = [
            "/Users/zakkeown/Code/SwiftAudioPrimitives/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json"
        ]

        var referenceData: [String: Any]?
        for path in possiblePaths {
            if let data = FileManager.default.contents(atPath: path),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                referenceData = json
                break
            }
        }
        
        guard let ref = referenceData,
              let signals = ref["test_signals"] as? [String: Any],
              let sineData = signals["sine_440hz"] as? [String: Any],
              let signalArray = sineData["data"] as? [Double],
              let cqtRef = (ref["cqt"] as? [String: Any])?["sine_440hz"] as? [String: Any],
              let hopLength = cqtRef["hop_length"] as? Int,
              let fmin = cqtRef["fmin"] as? Double,
              let nBins = cqtRef["n_bins"] as? Int,
              let binsPerOctave = cqtRef["bins_per_octave"] as? Int,
              let expectedMag = cqtRef["magnitude"] as? [[Double]] else {
            XCTFail("Failed to load reference data")
            return
        }
        
        let signal = signalArray.map { Float($0) }
        
        print("\n=== CQT Debug Output ===")
        print("Config: hopLength=\(hopLength), fmin=\(fmin), nBins=\(nBins), binsPerOctave=\(binsPerOctave)")
        print("Expected shape: \(expectedMag.count) frames x \(expectedMag[0].count) bins")
        
        let cqtConfig = CQTConfig(
            sampleRate: 22050,
            hopLength: hopLength,
            fMin: Float(fmin),
            nBins: nBins,
            binsPerOctave: binsPerOctave
        )
        
        let cqt = CQT(config: cqtConfig)
        let result = await cqt.transform(signal)
        let magnitude = result.magnitude
        
        print("Swift CQT shape: \(magnitude.count) bins x \(magnitude[0].count) frames")
        
        print("\nFirst 5 bins, first 3 frames:")
        print("Expected (reference):")
        for frameIdx in 0..<3 {
            let row = (0..<5).map { String(format: "%.6f", expectedMag[frameIdx][$0]) }.joined(separator: " ")
            print("  Frame \(frameIdx): \(row)")
        }
        
        print("Actual (Swift):")
        for frameIdx in 0..<3 {
            let row = (0..<5).map { String(format: "%.6f", magnitude[$0][frameIdx]) }.joined(separator: " ")
            print("  Frame \(frameIdx): \(row)")
        }
        
        // Find the bin with 440Hz (should be around bin 38 for fmin=32.7, 12 bins per octave)
        // 440/32.7 = 13.45... which is about 3.75 octaves = 45 bins
        let bin440 = Int(round(12.0 * log2(440.0 / fmin)))
        print("\n440Hz should be near bin \(bin440)")
        if bin440 < nBins {
            print("Bin \(bin440) magnitude (first 5 frames):")
            print("  Expected: \((0..<5).map { String(format: "%.6f", expectedMag[$0][bin440]) }.joined(separator: " "))")
            print("  Actual:   \((0..<5).map { String(format: "%.6f", magnitude[bin440][$0]) }.joined(separator: " "))")
        }
        
        print("=========================\n")
    }
}
