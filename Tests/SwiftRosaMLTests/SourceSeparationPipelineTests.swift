import XCTest
@testable import SwiftRosaML
import SwiftRosaCore

final class SourceSeparationPipelineTests: XCTestCase {

    // MARK: - Tolerance Constants
    //
    // For complex mask application:
    // - Integer values: exact within float tolerance
    // - General: should match analytical results closely
    private let exactTolerance: Float = 1e-6
    private let maskTolerance: Float = 1e-5

    // MARK: - Configuration Tests

    func testBanquetDefaultConfig() {
        let config = SourceSeparationConfig.banquetDefault

        XCTAssertEqual(config.sampleRate, 44100)
        XCTAssertEqual(config.nFFT, 2048)
        XCTAssertEqual(config.hopLength, 512)
        XCTAssertEqual(config.windowType, .hann)
        XCTAssertEqual(config.chunkDuration, 10.0)
        XCTAssertEqual(config.chunkOverlap, 0.5)
    }

    func testCustomConfig() {
        let config = SourceSeparationConfig(
            sampleRate: 22050,
            nFFT: 1024,
            hopLength: 256,
            windowType: .hamming,
            chunkDuration: 5.0,
            chunkOverlap: 0.25
        )

        XCTAssertEqual(config.sampleRate, 22050)
        XCTAssertEqual(config.nFFT, 1024)
        XCTAssertEqual(config.hopLength, 256)
        XCTAssertEqual(config.windowType, .hamming)
        XCTAssertEqual(config.chunkDuration, 5.0)
        XCTAssertEqual(config.chunkOverlap, 0.25)
    }

    func testConfigWithNoChunking() {
        let config = SourceSeparationConfig(
            sampleRate: 44100,
            nFFT: 2048,
            hopLength: 512,
            windowType: .hann,
            chunkDuration: nil,  // Process all at once
            chunkOverlap: 0.0
        )

        XCTAssertNil(config.chunkDuration)
    }

    // MARK: - Complex Mask Application Tests

    func testComplexMaskApplication() async {
        // Test complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

        // Create test spectrogram with real-only values
        let specReal: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let specImag: [[Float]] = [[0.0, 0.0], [0.0, 0.0]]
        let spectrogram = ComplexMatrix(real: specReal, imag: specImag)

        // Create mask that doubles real values (2+0i)
        let maskReal: [[Float]] = [[2.0, 2.0], [2.0, 2.0]]
        let maskImag: [[Float]] = [[0.0, 0.0], [0.0, 0.0]]
        let mask = ComplexMatrix(real: maskReal, imag: maskImag)

        // Apply mask
        let result = applyComplexMask(mask: mask, to: spectrogram)

        // Result should be exactly doubled (integer multiplication)
        XCTAssertEqual(result.real[0][0], 2.0, accuracy: exactTolerance)
        XCTAssertEqual(result.real[0][1], 4.0, accuracy: exactTolerance)
        XCTAssertEqual(result.real[1][0], 6.0, accuracy: exactTolerance)
        XCTAssertEqual(result.real[1][1], 8.0, accuracy: exactTolerance)

        // Imaginary parts should remain zero
        for f in 0..<result.rows {
            for t in 0..<result.cols {
                XCTAssertEqual(result.imag[f][t], 0.0, accuracy: exactTolerance,
                    "Imaginary should be 0 at [\(f)][\(t)]")
            }
        }
    }

    func testComplexMaskWithImaginary() async {
        // Test: (3+4i) * (1+2i) = (3*1 - 4*2) + (3*2 + 4*1)i = (3-8) + (6+4)i = -5 + 10i
        let specReal: [[Float]] = [[3.0]]
        let specImag: [[Float]] = [[4.0]]
        let spectrogram = ComplexMatrix(real: specReal, imag: specImag)

        let maskReal: [[Float]] = [[1.0]]
        let maskImag: [[Float]] = [[2.0]]
        let mask = ComplexMatrix(real: maskReal, imag: maskImag)

        let result = applyComplexMask(mask: mask, to: spectrogram)

        // Exact integer arithmetic
        XCTAssertEqual(result.real[0][0], -5.0, accuracy: exactTolerance)
        XCTAssertEqual(result.imag[0][0], 10.0, accuracy: exactTolerance)

        // Verify magnitude: |result| = |spec| * |mask|
        // |spec| = sqrt(3^2 + 4^2) = 5
        // |mask| = sqrt(1^2 + 2^2) = sqrt(5)
        // |result| = sqrt((-5)^2 + 10^2) = sqrt(125) = 5*sqrt(5)
        let specMag: Float = sqrt(Float(3*3 + 4*4))  // 5
        let maskMag: Float = sqrt(Float(1*1 + 2*2))  // sqrt(5)
        let resultMag = sqrt(result.real[0][0] * result.real[0][0] + result.imag[0][0] * result.imag[0][0])
        XCTAssertEqual(resultMag, specMag * maskMag, accuracy: maskTolerance,
            "Magnitude should be product of input magnitudes")
    }

    func testComplexMaskWithZeroMask() async {
        // Zero mask should produce exactly zero output
        let specReal: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let specImag: [[Float]] = [[0.5, 0.6], [0.7, 0.8]]
        let spectrogram = ComplexMatrix(real: specReal, imag: specImag)

        let maskReal: [[Float]] = [[0.0, 0.0], [0.0, 0.0]]
        let maskImag: [[Float]] = [[0.0, 0.0], [0.0, 0.0]]
        let mask = ComplexMatrix(real: maskReal, imag: maskImag)

        let result = applyComplexMask(mask: mask, to: spectrogram)

        for f in 0..<result.rows {
            for t in 0..<result.cols {
                // Should be exactly zero
                XCTAssertEqual(result.real[f][t], 0.0, accuracy: exactTolerance,
                    "Real should be 0 at [\(f)][\(t)]")
                XCTAssertEqual(result.imag[f][t], 0.0, accuracy: exactTolerance,
                    "Imag should be 0 at [\(f)][\(t)]")
            }
        }
    }

    func testComplexMaskIdentity() async {
        // Mask of (1+0i) should leave input exactly unchanged
        let specReal: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let specImag: [[Float]] = [[0.125, 0.25], [0.375, 0.5]]  // Exact binary fractions
        let spectrogram = ComplexMatrix(real: specReal, imag: specImag)

        let maskReal: [[Float]] = [[1.0, 1.0], [1.0, 1.0]]
        let maskImag: [[Float]] = [[0.0, 0.0], [0.0, 0.0]]
        let mask = ComplexMatrix(real: maskReal, imag: maskImag)

        let result = applyComplexMask(mask: mask, to: spectrogram)

        var maxError: Float = 0
        for f in 0..<result.rows {
            for t in 0..<result.cols {
                let realError = abs(result.real[f][t] - spectrogram.real[f][t])
                let imagError = abs(result.imag[f][t] - spectrogram.imag[f][t])
                maxError = max(maxError, max(realError, imagError))

                XCTAssertEqual(result.real[f][t], spectrogram.real[f][t], accuracy: maskTolerance,
                    "Real mismatch at [\(f)][\(t)]")
                XCTAssertEqual(result.imag[f][t], spectrogram.imag[f][t], accuracy: maskTolerance,
                    "Imag mismatch at [\(f)][\(t)]")
            }
        }

        XCTAssertLessThan(maxError, maskTolerance,
            "Identity mask max error: \(maxError)")
    }

    func testComplexMaskPhaseRotation() async {
        // Test 90-degree phase rotation: multiply by i = (0+1i)
        // (a+bi) * (0+1i) = -b + ai
        let specReal: [[Float]] = [[3.0]]
        let specImag: [[Float]] = [[4.0]]
        let spectrogram = ComplexMatrix(real: specReal, imag: specImag)

        let maskReal: [[Float]] = [[0.0]]
        let maskImag: [[Float]] = [[1.0]]  // i
        let mask = ComplexMatrix(real: maskReal, imag: maskImag)

        let result = applyComplexMask(mask: mask, to: spectrogram)

        // (3+4i) * i = 3i + 4i^2 = 3i - 4 = -4 + 3i
        XCTAssertEqual(result.real[0][0], -4.0, accuracy: exactTolerance)
        XCTAssertEqual(result.imag[0][0], 3.0, accuracy: exactTolerance)

        // Magnitude should be preserved under phase rotation
        let inputMag = sqrt(spectrogram.real[0][0] * spectrogram.real[0][0] +
                          spectrogram.imag[0][0] * spectrogram.imag[0][0])
        let outputMag = sqrt(result.real[0][0] * result.real[0][0] +
                            result.imag[0][0] * result.imag[0][0])
        XCTAssertEqual(outputMag, inputMag, accuracy: exactTolerance,
            "Magnitude should be preserved under phase rotation")
    }

    // MARK: - Chunking Logic Tests

    func testChunkCalculation() {
        let config = SourceSeparationConfig.banquetDefault

        // 10 seconds at 44100 Hz = 441000 samples
        let chunkSamples = Int(config.chunkDuration! * config.sampleRate)
        XCTAssertEqual(chunkSamples, 441000)

        // With 50% overlap, hop is half the chunk
        let chunkHop = Int(Float(chunkSamples) * (1.0 - config.chunkOverlap))
        XCTAssertEqual(chunkHop, 220500)
    }

    func testShortAudioNoChunking() {
        // Audio shorter than chunk duration shouldn't need chunking
        let config = SourceSeparationConfig.banquetDefault
        let audioLength = 5 * 44100  // 5 seconds

        let chunkSamples = Int(config.chunkDuration! * config.sampleRate)

        // 5 seconds is less than 10 second chunk
        XCTAssertLessThan(audioLength, chunkSamples)
    }

    func testLongAudioChunking() {
        // Audio longer than chunk duration needs chunking
        let config = SourceSeparationConfig.banquetDefault
        let audioLength = 60 * 44100  // 60 seconds

        let chunkSamples = Int(config.chunkDuration! * config.sampleRate)
        let chunkHop = Int(Float(chunkSamples) * (1.0 - config.chunkOverlap))

        // Calculate expected number of chunks
        let numChunks = (audioLength - chunkSamples) / chunkHop + 1
        XCTAssertGreaterThan(numChunks, 1)
    }

    // MARK: - Helper Function (since pipeline requires actual model)

    /// Helper to apply complex mask without needing full pipeline
    func applyComplexMask(mask: ComplexMatrix, to input: ComplexMatrix) -> ComplexMatrix {
        var resultReal = [[Float]](repeating: [Float](repeating: 0, count: input.cols), count: input.rows)
        var resultImag = [[Float]](repeating: [Float](repeating: 0, count: input.cols), count: input.rows)

        for f in 0..<input.rows {
            for t in 0..<input.cols {
                let ar = input.real[f][t]
                let ai = input.imag[f][t]
                let br = mask.real[f][t]
                let bi = mask.imag[f][t]

                // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                resultReal[f][t] = ar * br - ai * bi
                resultImag[f][t] = ar * bi + ai * br
            }
        }

        return ComplexMatrix(real: resultReal, imag: resultImag)
    }

    // MARK: - Progress Callback Tests

    func testProgressCallbackValues() {
        // Verify progress calculation logic
        let totalChunks = 10

        for i in 0..<totalChunks {
            let progress = Float(i + 1) / Float(totalChunks)
            XCTAssertGreaterThan(progress, 0)
            XCTAssertLessThanOrEqual(progress, 1.0)
        }

        // Final progress should be 1.0
        let finalProgress = Float(totalChunks) / Float(totalChunks)
        XCTAssertEqual(finalProgress, 1.0, accuracy: 0.0001)
    }

    // MARK: - STFT Config Consistency Tests

    func testSTFTConfigFromSeparationConfig() {
        let sepConfig = SourceSeparationConfig.banquetDefault

        // The pipeline should create STFT configs matching separation config
        let stftConfig = STFTConfig(
            uncheckedNFFT: sepConfig.nFFT,
            hopLength: sepConfig.hopLength,
            windowType: sepConfig.windowType,
            center: true
        )

        XCTAssertEqual(stftConfig.nFFT, sepConfig.nFFT)
        XCTAssertEqual(stftConfig.hopLength, sepConfig.hopLength)
        XCTAssertEqual(stftConfig.windowType, sepConfig.windowType)
    }

    // MARK: - Spectrogram Shape Tests

    func testExpectedSpectrogramShape() async {
        // Verify expected shapes for Banquet
        let config = SourceSeparationConfig.banquetDefault
        let signalLength = 441000  // 10 seconds

        // Generate test signal
        var signal = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            signal[i] = sin(Float(i) * 0.01)
        }

        let stftConfig = STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            windowType: config.windowType,
            center: true
        )

        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // Check frequency bins: nFFT/2 + 1
        XCTAssertEqual(spectrogram.rows, config.nFFT / 2 + 1)

        // Check time frames (approximately)
        let expectedFrames = signalLength / config.hopLength + 1
        XCTAssertEqual(spectrogram.cols, expectedFrames, accuracy: 5)
    }

    // MARK: - End-to-End Pipeline Tests (using mock identity transform)

    func testEndToEndPipelineWithIdentityMask() async {
        // Simulate E2E pipeline with identity mask (output = input)
        // This tests the full STFT -> mask -> ISTFT flow

        let config = SourceSeparationConfig(
            sampleRate: 22050,
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            chunkDuration: nil,  // No chunking for this test
            chunkOverlap: 0.0
        )

        // Generate test signal
        let signalLength = 22050  // 1 second
        var signal = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            signal[i] = 0.5 * sin(Float(i) * 0.1) + 0.3 * cos(Float(i) * 0.05)
        }

        // STFT
        let stftConfig = STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            windowType: config.windowType,
            center: true
        )
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // Apply identity mask (1+0i)
        let identityReal = [[Float]](repeating: [Float](repeating: 1.0, count: spectrogram.cols), count: spectrogram.rows)
        let identityImag = [[Float]](repeating: [Float](repeating: 0.0, count: spectrogram.cols), count: spectrogram.rows)
        let identityMask = ComplexMatrix(real: identityReal, imag: identityImag)

        let maskedSpec = applyComplexMask(mask: identityMask, to: spectrogram)

        // ISTFT
        let istft = ISTFT(config: stftConfig)
        let reconstructed = await istft.transform(maskedSpec, length: signalLength)

        // Verify reconstruction accuracy
        XCTAssertEqual(reconstructed.count, signalLength)

        // Calculate error in central region (avoid edge effects)
        let margin = config.nFFT / 2
        var maxError: Float = 0
        var sumSquaredError: Float = 0
        var compareCount = 0

        for i in margin..<(signalLength - margin) {
            let error = abs(signal[i] - reconstructed[i])
            maxError = max(maxError, error)
            sumSquaredError += error * error
            compareCount += 1
        }
        let rmse = sqrt(sumSquaredError / Float(compareCount))

        // With identity mask and COLA-compliant config, should match original closely
        XCTAssertLessThan(maxError, 1e-4,
            "E2E identity pipeline max error: \(maxError)")
        XCTAssertLessThan(rmse, 1e-5,
            "E2E identity pipeline RMSE: \(rmse)")
    }

    func testEndToEndPipelineWithHalfMask() async {
        // Test with 0.5 mask (output should be half amplitude)

        let config = SourceSeparationConfig(
            sampleRate: 22050,
            nFFT: 1024,
            hopLength: 256,
            windowType: .hann,
            chunkDuration: nil,
            chunkOverlap: 0.0
        )

        // Generate test signal
        let signalLength = 22050
        var signal = [Float](repeating: 0, count: signalLength)
        for i in 0..<signalLength {
            signal[i] = sin(Float(i) * 0.1)
        }

        // STFT
        let stftConfig = STFTConfig(
            uncheckedNFFT: config.nFFT,
            hopLength: config.hopLength,
            windowType: config.windowType,
            center: true
        )
        let stft = STFT(config: stftConfig)
        let spectrogram = await stft.transform(signal)

        // Apply 0.5 mask (halves amplitude)
        let halfReal = [[Float]](repeating: [Float](repeating: 0.5, count: spectrogram.cols), count: spectrogram.rows)
        let halfImag = [[Float]](repeating: [Float](repeating: 0.0, count: spectrogram.cols), count: spectrogram.rows)
        let halfMask = ComplexMatrix(real: halfReal, imag: halfImag)

        let maskedSpec = applyComplexMask(mask: halfMask, to: spectrogram)

        // ISTFT
        let istft = ISTFT(config: stftConfig)
        let reconstructed = await istft.transform(maskedSpec, length: signalLength)

        // Verify amplitude is halved
        let margin = config.nFFT / 2
        var sumSquaredOriginal: Float = 0
        var sumSquaredOutput: Float = 0

        for i in margin..<(signalLength - margin) {
            sumSquaredOriginal += signal[i] * signal[i]
            sumSquaredOutput += reconstructed[i] * reconstructed[i]
        }

        let rmsOriginal = sqrt(sumSquaredOriginal / Float(signalLength - 2 * margin))
        let rmsOutput = sqrt(sumSquaredOutput / Float(signalLength - 2 * margin))
        let rmsRatio = rmsOutput / max(rmsOriginal, 1e-10)

        // Output RMS should be ~0.5 of original (within 10%)
        XCTAssertEqual(rmsRatio, 0.5, accuracy: 0.05,
            "0.5 mask should halve amplitude, got ratio: \(rmsRatio)")
    }

    // MARK: - Chunk Boundary Tests

    func testChunkBoundarySmoothing() async {
        // Verify that chunk overlaps produce smooth transitions
        // Simulating what happens at chunk boundaries

        let chunkSize = 1024
        let overlapSize = 512
        let hopSize = chunkSize - overlapSize

        // Create two overlapping "chunks" with different values
        var chunk1 = [Float](repeating: 1.0, count: chunkSize)
        var chunk2 = [Float](repeating: 1.0, count: chunkSize)

        // Crossfade window (linear)
        var fadeOut = [Float](repeating: 0, count: overlapSize)
        var fadeIn = [Float](repeating: 0, count: overlapSize)
        for i in 0..<overlapSize {
            fadeOut[i] = 1.0 - Float(i) / Float(overlapSize)
            fadeIn[i] = Float(i) / Float(overlapSize)
        }

        // Apply fades to overlap regions
        for i in 0..<overlapSize {
            chunk1[hopSize + i] *= fadeOut[i]
            chunk2[i] *= fadeIn[i]
        }

        // Combine at overlap
        var combined = [Float](repeating: 0, count: chunkSize + hopSize)
        for i in 0..<chunkSize {
            combined[i] += chunk1[i]
        }
        for i in 0..<chunkSize {
            combined[hopSize + i] += chunk2[i]
        }

        // Check smoothness at boundary
        let boundaryStart = hopSize
        let boundaryEnd = boundaryStart + overlapSize

        // Calculate max derivative (should be smooth, no discontinuities)
        var maxDerivative: Float = 0
        for i in boundaryStart..<(boundaryEnd - 1) {
            let derivative = abs(combined[i + 1] - combined[i])
            maxDerivative = max(maxDerivative, derivative)
        }

        // With proper overlap-add, the combined signal should be constant
        // (linear fade out + linear fade in = 1.0)
        XCTAssertLessThan(maxDerivative, 0.01,
            "Chunk boundary should be smooth, max derivative: \(maxDerivative)")

        // Verify value at midpoint of overlap
        let midpoint = boundaryStart + overlapSize / 2
        XCTAssertEqual(combined[midpoint], 1.0, accuracy: 0.01,
            "Combined value at overlap midpoint should be ~1.0")
    }
}
