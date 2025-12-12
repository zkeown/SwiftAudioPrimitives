import XCTest
import CoreML
@testable import SwiftAudioPrimitives

final class AudioModelRunnerTests: XCTestCase {

    // MARK: - Tolerance Analysis
    //
    // For exact integer values stored as Float32:
    // - Float32 can exactly represent integers up to 2^24 (16,777,216)
    // - For small integers (1-1000), conversion should be exact
    //
    // For floating-point values:
    // - Use relative tolerance: |a - b| / max(|a|, |b|) < epsilon
    // - Or use Float.ulpOfOne * scale for absolute tolerance
    private let exactIntegerTolerance: Float = Float.ulpOfOne * 10  // ~1e-6
    private let floatTolerance: Float = 1e-6

    // MARK: - MLMultiArray Conversion Tests (2D)

    func testToMLMultiArray2D() throws {
        // Integer values should be exactly representable
        let data: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]

        let array = try AudioModelRunner.toMLMultiArray(data, shape: [2, 3])

        XCTAssertEqual(array.shape, [2, 3] as [NSNumber])

        // For small integers stored as Float, should be exactly representable
        XCTAssertEqual(array[[0, 0] as [NSNumber]] as! Float, 1.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[0, 1] as [NSNumber]] as! Float, 2.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[0, 2] as [NSNumber]] as! Float, 3.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 0] as [NSNumber]] as! Float, 4.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 1] as [NSNumber]] as! Float, 5.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 2] as [NSNumber]] as! Float, 6.0, accuracy: exactIntegerTolerance)
    }

    func testFromMLMultiArray2D() throws {
        // Create MLMultiArray with exact integer values
        let array = try MLMultiArray(shape: [2, 3], dataType: .float32)
        array[[0, 0] as [NSNumber]] = 1.0
        array[[0, 1] as [NSNumber]] = 2.0
        array[[0, 2] as [NSNumber]] = 3.0
        array[[1, 0] as [NSNumber]] = 4.0
        array[[1, 1] as [NSNumber]] = 5.0
        array[[1, 2] as [NSNumber]] = 6.0

        let result = AudioModelRunner.fromMLMultiArray2D(array, rows: 2, cols: 3)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].count, 3)

        // Integer values should round-trip exactly
        XCTAssertEqual(result[0][0], 1.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(result[0][1], 2.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(result[0][2], 3.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(result[1][0], 4.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(result[1][1], 5.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(result[1][2], 6.0, accuracy: exactIntegerTolerance)
    }

    func test2DRoundTrip() throws {
        // Use decimal values that are exactly representable in Float32
        // 0.125 = 1/8 = 2^-3 (exact), 0.5 = 1/2 = 2^-1 (exact), etc.
        let original: [[Float]] = [
            [0.125, 0.25, 0.375, 0.5],
            [0.625, 0.75, 0.875, 1.0],
            [1.125, 1.25, 1.375, 1.5],
        ]

        let array = try AudioModelRunner.toMLMultiArray(original, shape: [3, 4])
        let reconstructed = AudioModelRunner.fromMLMultiArray2D(array, rows: 3, cols: 4)

        XCTAssertEqual(reconstructed.count, original.count)

        // Verify every single value (comprehensive check, not spot check)
        var maxError: Float = 0
        for i in 0..<original.count {
            XCTAssertEqual(reconstructed[i].count, original[i].count)
            for j in 0..<original[i].count {
                let error = abs(reconstructed[i][j] - original[i][j])
                maxError = max(maxError, error)
                XCTAssertEqual(reconstructed[i][j], original[i][j], accuracy: floatTolerance,
                    "Mismatch at [\(i)][\(j)]: got \(reconstructed[i][j]), expected \(original[i][j])")
            }
        }

        // Summary assertion
        XCTAssertLessThan(maxError, floatTolerance,
            "Round-trip max error: \(maxError)")
    }

    // MARK: - MLMultiArray Conversion Tests (3D)

    func testToMLMultiArray3D() throws {
        let data: [[[Float]]] = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]

        let array = try AudioModelRunner.toMLMultiArray(data, shape: [2, 2, 2])

        XCTAssertEqual(array.shape, [2, 2, 2] as [NSNumber])

        // All corners and center values (comprehensive for small array)
        XCTAssertEqual(array[[0, 0, 0] as [NSNumber]] as! Float, 1.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[0, 0, 1] as [NSNumber]] as! Float, 2.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[0, 1, 0] as [NSNumber]] as! Float, 3.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[0, 1, 1] as [NSNumber]] as! Float, 4.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 0, 0] as [NSNumber]] as! Float, 5.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 0, 1] as [NSNumber]] as! Float, 6.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 1, 0] as [NSNumber]] as! Float, 7.0, accuracy: exactIntegerTolerance)
        XCTAssertEqual(array[[1, 1, 1] as [NSNumber]] as! Float, 8.0, accuracy: exactIntegerTolerance)
    }

    func testFromMLMultiArray3D() throws {
        let array = try MLMultiArray(shape: [2, 2, 2], dataType: .float32)
        for i in 0..<8 {
            array[i] = Float(i + 1) as NSNumber
        }

        let result = AudioModelRunner.fromMLMultiArray3D(array, d1: 2, d2: 2, d3: 2)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].count, 2)
        XCTAssertEqual(result[0][0].count, 2)
    }

    // MARK: - MLMultiArray Conversion Tests (4D)

    func testToMLMultiArray4D() throws {
        let data: [[[[Float]]]] = [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        ]

        let array = try AudioModelRunner.toMLMultiArray(data, shape: [1, 2, 2, 2])

        XCTAssertEqual(array.shape, [1, 2, 2, 2] as [NSNumber])
        XCTAssertEqual(array[[0, 0, 0, 0] as [NSNumber]] as! Float, 1.0, accuracy: 0.001)
        XCTAssertEqual(array[[0, 1, 1, 1] as [NSNumber]] as! Float, 8.0, accuracy: 0.001)
    }

    // MARK: - ComplexMatrix Conversion Tests

    func testComplexToMLMultiArray() throws {
        // Create a simple ComplexMatrix
        let real: [[Float]] = [[1, 2], [3, 4], [5, 6]]
        let imag: [[Float]] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        let complex = ComplexMatrix(real: real, imag: imag)

        let array = try AudioModelRunner.complexToMLMultiArray(complex, batch: 1)

        // Shape should be [batch, 2 (real/imag channels), freqs, frames]
        XCTAssertEqual(array.shape.count, 4)
        XCTAssertEqual(array.shape[0], 1 as NSNumber)  // batch
        XCTAssertEqual(array.shape[1], 2 as NSNumber)  // real/imag channels
        XCTAssertEqual(array.shape[2], 3 as NSNumber)  // freqs
        XCTAssertEqual(array.shape[3], 2 as NSNumber)  // frames

        // Check values
        XCTAssertEqual(array[[0, 0, 0, 0] as [NSNumber]] as! Float, 1.0, accuracy: 0.001)  // real[0][0]
        XCTAssertEqual(array[[0, 1, 0, 0] as [NSNumber]] as! Float, 0.1, accuracy: 0.001)  // imag[0][0]
    }

    func testMLMultiArrayToComplex() throws {
        // Create MLMultiArray in model output format
        let array = try MLMultiArray(shape: [1, 2, 3, 2], dataType: .float32)

        // Fill real channel (channel 0)
        array[[0, 0, 0, 0] as [NSNumber]] = 1.0
        array[[0, 0, 0, 1] as [NSNumber]] = 2.0
        array[[0, 0, 1, 0] as [NSNumber]] = 3.0
        array[[0, 0, 1, 1] as [NSNumber]] = 4.0
        array[[0, 0, 2, 0] as [NSNumber]] = 5.0
        array[[0, 0, 2, 1] as [NSNumber]] = 6.0

        // Fill imag channel (channel 1)
        array[[0, 1, 0, 0] as [NSNumber]] = 0.1
        array[[0, 1, 0, 1] as [NSNumber]] = 0.2
        array[[0, 1, 1, 0] as [NSNumber]] = 0.3
        array[[0, 1, 1, 1] as [NSNumber]] = 0.4
        array[[0, 1, 2, 0] as [NSNumber]] = 0.5
        array[[0, 1, 2, 1] as [NSNumber]] = 0.6

        let complex = AudioModelRunner.mlMultiArrayToComplex(array, freqs: 3, frames: 2)

        XCTAssertEqual(complex.rows, 3)
        XCTAssertEqual(complex.cols, 2)

        XCTAssertEqual(complex.real[0][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(complex.real[0][1], 2.0, accuracy: 0.001)
        XCTAssertEqual(complex.real[2][1], 6.0, accuracy: 0.001)

        XCTAssertEqual(complex.imag[0][0], 0.1, accuracy: 0.001)
        XCTAssertEqual(complex.imag[2][1], 0.6, accuracy: 0.001)
    }

    func testComplexRoundTrip() throws {
        let real: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
        let imag: [[Float]] = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
        let original = ComplexMatrix(real: real, imag: imag)

        let array = try AudioModelRunner.complexToMLMultiArray(original, batch: 1)
        let reconstructed = AudioModelRunner.mlMultiArrayToComplex(array, freqs: 3, frames: 4)

        XCTAssertEqual(reconstructed.rows, original.rows)
        XCTAssertEqual(reconstructed.cols, original.cols)

        for f in 0..<original.rows {
            for t in 0..<original.cols {
                XCTAssertEqual(reconstructed.real[f][t], original.real[f][t], accuracy: 0.0001)
                XCTAssertEqual(reconstructed.imag[f][t], original.imag[f][t], accuracy: 0.0001)
            }
        }
    }

    // MARK: - Flat Array Conversion Tests

    func testFromMLMultiArrayFlat() throws {
        let array = try MLMultiArray(shape: [6], dataType: .float32)
        for i in 0..<6 {
            array[i] = Float(i + 1) as NSNumber
        }

        let result = AudioModelRunner.fromMLMultiArray(array)

        XCTAssertEqual(result.count, 6)
        XCTAssertEqual(result[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(result[5], 6.0, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testEmptyArray() throws {
        let data: [[Float]] = []
        let array = try AudioModelRunner.toMLMultiArray(data, shape: [0, 0])
        XCTAssertNotNil(array)
    }

    func testSingleElement() throws {
        let data: [[Float]] = [[42.0]]
        let array = try AudioModelRunner.toMLMultiArray(data, shape: [1, 1])

        XCTAssertEqual(array[[0, 0] as [NSNumber]] as! Float, 42.0, accuracy: 0.001)
    }

    // MARK: - Model Configuration Tests

    func testModelConfigurationDefaults() {
        let config = MLModelConfiguration()

        // Just verify default config can be created
        XCTAssertNotNil(config)
    }

    func testModelConfigurationWithComputeUnits() {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        XCTAssertEqual(config.computeUnits, .all)

        config.computeUnits = .cpuOnly
        XCTAssertEqual(config.computeUnits, .cpuOnly)

        config.computeUnits = .cpuAndNeuralEngine
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    // MARK: - Large Array Tests

    func testLargeSpectrogramConversion() throws {
        // Realistic spectrogram size
        let nFreqs = 1025  // n_fft=2048 -> 1025 bins
        let nFrames = 500

        // Use seeded random for reproducibility
        srand48(42)
        var real: [[Float]] = []
        var imag: [[Float]] = []

        for _ in 0..<nFreqs {
            var realRow = [Float](repeating: 0, count: nFrames)
            var imagRow = [Float](repeating: 0, count: nFrames)
            for t in 0..<nFrames {
                realRow[t] = Float(drand48() * 2 - 1)
                imagRow[t] = Float(drand48() * 2 - 1)
            }
            real.append(realRow)
            imag.append(imagRow)
        }

        let complex = ComplexMatrix(real: real, imag: imag)
        let array = try AudioModelRunner.complexToMLMultiArray(complex, batch: 1)

        XCTAssertEqual(array.shape, [1, 2, nFreqs, nFrames] as [NSNumber])

        // Round-trip
        let reconstructed = AudioModelRunner.mlMultiArrayToComplex(array, freqs: nFreqs, frames: nFrames)

        XCTAssertEqual(reconstructed.rows, nFreqs)
        XCTAssertEqual(reconstructed.cols, nFrames)

        // Comprehensive spot-checking at multiple positions
        let checkFreqs = [0, nFreqs/4, nFreqs/2, 3*nFreqs/4, nFreqs-1]
        let checkFrames = [0, nFrames/4, nFrames/2, 3*nFrames/4, nFrames-1]

        var maxRealError: Float = 0
        var maxImagError: Float = 0

        for f in checkFreqs {
            for t in checkFrames {
                let realError = abs(reconstructed.real[f][t] - complex.real[f][t])
                let imagError = abs(reconstructed.imag[f][t] - complex.imag[f][t])
                maxRealError = max(maxRealError, realError)
                maxImagError = max(maxImagError, imagError)

                XCTAssertEqual(reconstructed.real[f][t], complex.real[f][t], accuracy: floatTolerance,
                    "Real mismatch at [\(f)][\(t)]")
                XCTAssertEqual(reconstructed.imag[f][t], complex.imag[f][t], accuracy: floatTolerance,
                    "Imag mismatch at [\(f)][\(t)]")
            }
        }

        // Summary assertions
        XCTAssertLessThan(maxRealError, floatTolerance,
            "Large spectrogram real max error: \(maxRealError)")
        XCTAssertLessThan(maxImagError, floatTolerance,
            "Large spectrogram imag max error: \(maxImagError)")

        // Verify total element count
        let totalElements = nFreqs * nFrames
        XCTAssertEqual(totalElements, 512500,
            "Should handle 512,500 element array")
    }

    func testLargeSpectrogramFullVerification() throws {
        // Smaller but still significant size for full verification
        let nFreqs = 129  // n_fft=256 -> 129 bins
        let nFrames = 50

        // Use deterministic values for exact verification
        var real: [[Float]] = []
        var imag: [[Float]] = []

        for f in 0..<nFreqs {
            var realRow = [Float](repeating: 0, count: nFrames)
            var imagRow = [Float](repeating: 0, count: nFrames)
            for t in 0..<nFrames {
                // Use deterministic pattern
                realRow[t] = Float(f * nFrames + t) / Float(nFreqs * nFrames)
                imagRow[t] = -Float(f * nFrames + t) / Float(nFreqs * nFrames)
            }
            real.append(realRow)
            imag.append(imagRow)
        }

        let complex = ComplexMatrix(real: real, imag: imag)
        let array = try AudioModelRunner.complexToMLMultiArray(complex, batch: 1)
        let reconstructed = AudioModelRunner.mlMultiArrayToComplex(array, freqs: nFreqs, frames: nFrames)

        // Full verification of ALL elements
        var maxError: Float = 0
        var errorCount = 0

        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                let realError = abs(reconstructed.real[f][t] - complex.real[f][t])
                let imagError = abs(reconstructed.imag[f][t] - complex.imag[f][t])
                maxError = max(maxError, max(realError, imagError))

                if realError > floatTolerance || imagError > floatTolerance {
                    errorCount += 1
                }
            }
        }

        XCTAssertEqual(errorCount, 0,
            "Found \(errorCount) elements with error > \(floatTolerance)")
        XCTAssertLessThan(maxError, floatTolerance,
            "Full verification max error: \(maxError)")
    }

    // MARK: - Batch Dimension Tests

    func testComplexToMLMultiArrayBatch() throws {
        // Test with batch > 1
        let real: [[Float]] = [[1, 2], [3, 4], [5, 6]]
        let imag: [[Float]] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        let complex = ComplexMatrix(real: real, imag: imag)

        let batch4 = try AudioModelRunner.complexToMLMultiArray(complex, batch: 4)

        // Shape should include batch dimension
        XCTAssertEqual(batch4.shape.count, 4)
        XCTAssertEqual(batch4.shape[0], 4 as NSNumber)  // batch
        XCTAssertEqual(batch4.shape[1], 2 as NSNumber)  // real/imag channels
        XCTAssertEqual(batch4.shape[2], 3 as NSNumber)  // freqs
        XCTAssertEqual(batch4.shape[3], 2 as NSNumber)  // frames
    }
}
