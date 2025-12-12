import XCTest
@testable import SwiftRosaEffects
@testable import SwiftRosaCore

final class MFCCDebugTest: XCTestCase {
    func testMFCCValues() async throws {
        // Load reference data
        let possiblePaths = [
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("ReferenceData/librosa_reference.json")
                .path,
            FileManager.default.currentDirectoryPath + "/Tests/SwiftRosaCoreTests/ReferenceData/librosa_reference.json",
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
              let mfccRef = ref["mfcc"] as? [String: Any],
              let sineRef = mfccRef["sine_440hz"] as? [String: Any],
              let expectedMFCC = sineRef["mfcc"] as? [[Double]] else {
            XCTFail("Failed to load reference data")
            return
        }
        
        let signal = signalArray.map { Float($0) }
        
        let mfcc = MFCC(
            sampleRate: 22050,
            nFFT: 2048,
            hopLength: 512,
            nMels: 128,
            nMFCC: 13
        )
        
        let result = await mfcc.transform(signal)
        
        print("\n=== MFCC Debug Output ===")
        print("Result shape: \(result.count) x \(result[0].count)")
        print("\nActual MFCC[0] (first 5 frames):")
        for i in 0..<5 {
            print("  Frame \(i): \(result[0][i])")
        }
        
        print("\nExpected MFCC[0] (first 5 frames):")
        for i in 0..<5 {
            print("  Frame \(i): \(expectedMFCC[i][0])")
        }
        
        print("\nActual MFCC[1] (first 5 frames):")
        for i in 0..<5 {
            print("  Frame \(i): \(result[1][i])")
        }
        
        print("\nExpected MFCC[1] (first 5 frames):")
        for i in 0..<5 {
            print("  Frame \(i): \(expectedMFCC[i][1])")
        }
        
        print("=========================\n")
    }
}
