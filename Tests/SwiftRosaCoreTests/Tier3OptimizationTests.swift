import XCTest
@testable import SwiftRosaCore

/// Tests for Tier 3 performance optimizations:
/// - ContiguousMatrix storage
/// - STFT contiguous output methods
/// - SpectralFeatures contiguous input methods
/// - GPU FFT implementation
final class Tier3OptimizationTests: XCTestCase {

    // MARK: - Test Helpers

    /// Generate a pure sine wave
    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let numSamples = Int(sampleRate * duration)
        return (0..<numSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }
    }

    /// Generate test signal with multiple harmonics
    private func generateTestSignal(length: Int, sampleRate: Float = 22050) -> [Float] {
        (0..<length).map { i in
            let t = Float(i) / sampleRate
            return 0.5 * sin(2 * Float.pi * 440 * t) +
                   0.3 * sin(2 * Float.pi * 880 * t) +
                   0.2 * sin(2 * Float.pi * 1760 * t)
        }
    }

    // MARK: - ContiguousMatrix Tests

    func testContiguousMatrixCreation() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let matrix = ContiguousMatrix(data: data, rows: 2, cols: 3)

        XCTAssertEqual(matrix.rows, 2)
        XCTAssertEqual(matrix.cols, 3)
        XCTAssertEqual(matrix.count, 6)
    }

    func testContiguousMatrixZeroInit() {
        let matrix = ContiguousMatrix(rows: 10, cols: 20)

        XCTAssertEqual(matrix.rows, 10)
        XCTAssertEqual(matrix.cols, 20)
        XCTAssertEqual(matrix.count, 200)

        // All values should be zero
        for i in 0..<matrix.rows {
            for j in 0..<matrix.cols {
                XCTAssertEqual(matrix[i, j], 0)
            }
        }
    }

    func testContiguousMatrixSubscript() {
        // Row-major layout: data = [row0col0, row0col1, row0col2, row1col0, row1col1, row1col2]
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        var matrix = ContiguousMatrix(data: data, rows: 2, cols: 3)

        // Test read
        XCTAssertEqual(matrix[0, 0], 1)
        XCTAssertEqual(matrix[0, 1], 2)
        XCTAssertEqual(matrix[0, 2], 3)
        XCTAssertEqual(matrix[1, 0], 4)
        XCTAssertEqual(matrix[1, 1], 5)
        XCTAssertEqual(matrix[1, 2], 6)

        // Test write
        matrix[0, 1] = 10
        XCTAssertEqual(matrix[0, 1], 10)
    }

    func testContiguousMatrixRowAccess() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let matrix = ContiguousMatrix(data: data, rows: 2, cols: 3)

        let row0 = Array(matrix.row(0))
        let row1 = Array(matrix.row(1))

        XCTAssertEqual(row0, [1, 2, 3])
        XCTAssertEqual(row1, [4, 5, 6])
    }

    func testContiguousMatrixColumnAccess() {
        // 3 rows × 2 cols, row-major
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let matrix = ContiguousMatrix(data: data, rows: 3, cols: 2)

        let col0 = matrix.column(0)
        let col1 = matrix.column(1)

        XCTAssertEqual(col0, [1, 3, 5])  // Elements at indices 0, 2, 4
        XCTAssertEqual(col1, [2, 4, 6])  // Elements at indices 1, 3, 5
    }

    func testContiguousMatrixWithColumn() {
        // 4 rows × 3 cols
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let matrix = ContiguousMatrix(data: data, rows: 4, cols: 3)

        // Column 1 should be [2, 5, 8, 11] with stride 3
        var sum: Float = 0
        matrix.withColumn(1) { ptr, stride, count in
            XCTAssertEqual(count, 4)  // 4 rows
            XCTAssertEqual(stride, 3) // stride = cols

            for i in 0..<count {
                sum += ptr[i * stride]
            }
        }

        XCTAssertEqual(sum, 2 + 5 + 8 + 11)
    }

    func testContiguousMatrixFrom2DArray() {
        let array2D: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6]
        ]

        let matrix = ContiguousMatrix(from: array2D)

        XCTAssertEqual(matrix.rows, 2)
        XCTAssertEqual(matrix.cols, 3)
        XCTAssertEqual(matrix[0, 0], 1)
        XCTAssertEqual(matrix[1, 2], 6)
    }

    func testContiguousMatrixTo2DArray() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let matrix = ContiguousMatrix(data: data, rows: 2, cols: 3)

        let array2D = matrix.to2DArray()

        XCTAssertEqual(array2D.count, 2)
        XCTAssertEqual(array2D[0], [1, 2, 3])
        XCTAssertEqual(array2D[1], [4, 5, 6])
    }

    func testContiguousMatrixRoundTrip() {
        let original: [[Float]] = [
            [1.5, 2.5, 3.5, 4.5],
            [5.5, 6.5, 7.5, 8.5],
            [9.5, 10.5, 11.5, 12.5]
        ]

        let matrix = ContiguousMatrix(from: original)
        let restored = matrix.to2DArray()

        XCTAssertEqual(restored, original)
    }

    func testContiguousMatrixSquared() {
        let data: [Float] = [2, 3, 4]
        let matrix = ContiguousMatrix(data: data, rows: 1, cols: 3)

        let squared = matrix.squared

        XCTAssertEqual(squared[0, 0], 4, accuracy: 1e-5)
        XCTAssertEqual(squared[0, 1], 9, accuracy: 1e-5)
        XCTAssertEqual(squared[0, 2], 16, accuracy: 1e-5)
    }

    func testContiguousMatrixSqrt() {
        let data: [Float] = [4, 9, 16]
        let matrix = ContiguousMatrix(data: data, rows: 1, cols: 3)

        let sqrtMatrix = matrix.sqrt

        XCTAssertEqual(sqrtMatrix[0, 0], 2, accuracy: 1e-5)
        XCTAssertEqual(sqrtMatrix[0, 1], 3, accuracy: 1e-5)
        XCTAssertEqual(sqrtMatrix[0, 2], 4, accuracy: 1e-5)
    }

    func testContiguousMatrixLog() {
        let data: [Float] = [exp(1), exp(2), exp(3)]
        let matrix = ContiguousMatrix(data: data, rows: 1, cols: 3)

        let logMatrix = matrix.log()

        XCTAssertEqual(logMatrix[0, 0], 1, accuracy: 1e-5)
        XCTAssertEqual(logMatrix[0, 1], 2, accuracy: 1e-5)
        XCTAssertEqual(logMatrix[0, 2], 3, accuracy: 1e-5)
    }

    func testContiguousMatrixLogWithFloor() {
        // Test that floor prevents -inf
        let data: [Float] = [0, 0.001, 1]
        let matrix = ContiguousMatrix(data: data, rows: 1, cols: 3)

        let logMatrix = matrix.log(floor: 1e-5)

        // First value should be log(1e-5), not -inf
        XCTAssertFalse(logMatrix[0, 0].isInfinite)
        XCTAssertEqual(logMatrix[0, 0], log(Float(1e-5)), accuracy: 1e-3)
    }

    // MARK: - STFT Contiguous Output Tests

    func testMagnitudeContiguousShape() async {
        let signal = generateTestSignal(length: 8192)
        let config = STFTConfig(uncheckedNFFT: 1024, hopLength: 256)
        let stft = STFT(config: config)

        let mag = await stft.magnitudeContiguous(signal)

        // nFreqs = nFFT/2 + 1 = 513
        XCTAssertEqual(mag.rows, 513)
        XCTAssertGreaterThan(mag.cols, 0)
    }

    func testMagnitudeContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 4096)
        let config = STFTConfig(uncheckedNFFT: 512, hopLength: 128)
        let stft = STFT(config: config)

        // Compute both versions
        let regular = await stft.magnitude(signal)
        let contiguous = await stft.magnitudeContiguous(signal)

        // Should have same dimensions
        XCTAssertEqual(contiguous.rows, regular.count)
        XCTAssertEqual(contiguous.cols, regular.first?.count ?? 0)

        // Values should match
        for f in 0..<regular.count {
            for t in 0..<regular[f].count {
                XCTAssertEqual(
                    contiguous[f, t], regular[f][t],
                    accuracy: 1e-5,
                    "Mismatch at [\(f)][\(t)]: contiguous=\(contiguous[f, t]), regular=\(regular[f][t])"
                )
            }
        }
    }

    func testPowerContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 4096)
        let config = STFTConfig(uncheckedNFFT: 512, hopLength: 128)
        let stft = STFT(config: config)

        let regular = await stft.power(signal)
        let contiguous = await stft.powerContiguous(signal)

        XCTAssertEqual(contiguous.rows, regular.count)
        XCTAssertEqual(contiguous.cols, regular.first?.count ?? 0)

        for f in 0..<min(10, regular.count) {
            for t in 0..<min(10, regular[f].count) {
                XCTAssertEqual(
                    contiguous[f, t], regular[f][t],
                    accuracy: 1e-5,
                    "Power mismatch at [\(f)][\(t)]"
                )
            }
        }
    }

    func testTransformContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 4096)
        let config = STFTConfig(uncheckedNFFT: 512, hopLength: 128)
        let stft = STFT(config: config)

        let regular = await stft.transform(signal)
        let contiguous = await stft.transformContiguous(signal)

        XCTAssertEqual(contiguous.rows, regular.rows)
        XCTAssertEqual(contiguous.cols, regular.cols)

        // Check a sample of values
        for f in stride(from: 0, to: regular.rows, by: 50) {
            for t in stride(from: 0, to: regular.cols, by: 10) {
                // ContiguousComplexMatrix stores data as flat arrays
                let contiguousVal = contiguous[f, t]  // Returns (real: Float, imag: Float)
                XCTAssertEqual(
                    contiguousVal.real, regular.real[f][t],
                    accuracy: 1e-5,
                    "Real mismatch at [\(f)][\(t)]"
                )
                XCTAssertEqual(
                    contiguousVal.imag, regular.imag[f][t],
                    accuracy: 1e-5,
                    "Imag mismatch at [\(f)][\(t)]"
                )
            }
        }
    }

    func testContiguousEmptySignal() async {
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 512))

        let mag = await stft.magnitudeContiguous([])

        XCTAssertEqual(mag.rows, 0)
        XCTAssertEqual(mag.cols, 0)
    }

    // MARK: - SpectralFeatures Contiguous Tests

    func testCentroidContiguousMatchesRegular() async {
        let signal = generateSineWave(frequency: 1000, sampleRate: 22050, duration: 1.0)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.centroid(signal)

        // Get contiguous version
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.centroidContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        for i in 0..<regular.count {
            XCTAssertEqual(
                contiguous[i], regular[i],
                accuracy: 1e-3,
                "Centroid mismatch at frame \(i): contiguous=\(contiguous[i]), regular=\(regular[i])"
            )
        }
    }

    func testBandwidthContiguousMatchesRegular() async {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.bandwidth(signal)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.bandwidthContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        for i in 0..<regular.count {
            XCTAssertEqual(
                contiguous[i], regular[i],
                accuracy: 1e-3,
                "Bandwidth mismatch at frame \(i)"
            )
        }
    }

    func testRolloffContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 22050)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.rolloff(signal)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.rolloffContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        for i in 0..<regular.count {
            XCTAssertEqual(
                contiguous[i], regular[i],
                accuracy: 1e-3,
                "Rolloff mismatch at frame \(i)"
            )
        }
    }

    func testFlatnessContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 22050)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.flatness(signal)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.flatnessContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        for i in 0..<regular.count {
            XCTAssertEqual(
                contiguous[i], regular[i],
                accuracy: 1e-4,
                "Flatness mismatch at frame \(i)"
            )
        }
    }

    func testFluxContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 22050)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.flux(signal)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.fluxContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        // Flux at frame 0 should be 0 for both
        XCTAssertEqual(contiguous[0], 0)

        for i in 1..<regular.count {
            XCTAssertEqual(
                contiguous[i], regular[i],
                accuracy: 1e-3,
                "Flux mismatch at frame \(i)"
            )
        }
    }

    func testContrastContiguousMatchesRegular() async {
        let signal = generateTestSignal(length: 22050)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.contrast(signal)

        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let mag = await stft.magnitudeContiguous(signal)
        let contiguous = spectralFeatures.contrastContiguous(magnitudeSpec: mag)

        XCTAssertEqual(regular.count, contiguous.count)

        for band in 0..<regular.count {
            XCTAssertEqual(regular[band].count, contiguous[band].count)

            for t in 0..<regular[band].count {
                XCTAssertEqual(
                    contiguous[band][t], regular[band][t],
                    accuracy: 1e-3,
                    "Contrast mismatch at band \(band), frame \(t)"
                )
            }
        }
    }

    func testExtractAllContiguousMatchesRegular() async {
        let signal = generateSineWave(frequency: 440, sampleRate: 22050, duration: 1.0)
        let spectralFeatures = SpectralFeatures()

        let regular = await spectralFeatures.extractAll(signal)
        let contiguous = await spectralFeatures.extractAllContiguous(signal)

        // Check all feature arrays have matching lengths
        XCTAssertEqual(regular.centroid.count, contiguous.centroid.count)
        XCTAssertEqual(regular.bandwidth.count, contiguous.bandwidth.count)
        XCTAssertEqual(regular.rolloff.count, contiguous.rolloff.count)
        XCTAssertEqual(regular.flatness.count, contiguous.flatness.count)
        XCTAssertEqual(regular.flux.count, contiguous.flux.count)
        XCTAssertEqual(regular.contrast.count, contiguous.contrast.count)

        // Spot check a few values
        if !regular.centroid.isEmpty {
            XCTAssertEqual(
                contiguous.centroid[0], regular.centroid[0],
                accuracy: 1e-3
            )
        }
    }

    // MARK: - GPU FFT Tests

    func testBatchFFTBasic() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        // Simple test: FFT of DC signal
        let n = 8
        let real: [Float] = [Float](repeating: 1, count: n)
        let imag: [Float] = [Float](repeating: 0, count: n)

        let (resultReal, resultImag) = try await metalEngine.batchFFT(
            real: real,
            imag: imag,
            n: n,
            batchSize: 1
        )

        XCTAssertEqual(resultReal.count, n)
        XCTAssertEqual(resultImag.count, n)

        // DC component should be sum of all values = n
        XCTAssertEqual(resultReal[0], Float(n), accuracy: 1e-3)
        XCTAssertEqual(resultImag[0], 0, accuracy: 1e-3)
    }

    func testBatchFFTSineWave() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let n = 64
        let k = 4  // Put energy at bin 4

        // Generate sine at bin k: cos(2*pi*k*n/N) + j*sin(2*pi*k*n/N)
        var real = [Float](repeating: 0, count: n)
        for i in 0..<n {
            real[i] = cos(2 * Float.pi * Float(k) * Float(i) / Float(n))
        }
        let imag = [Float](repeating: 0, count: n)

        let (resultReal, resultImag) = try await metalEngine.batchFFT(
            real: real,
            imag: imag,
            n: n,
            batchSize: 1
        )

        // Energy should be concentrated at bin k
        let magK = sqrt(resultReal[k] * resultReal[k] + resultImag[k] * resultImag[k])
        let magOther = sqrt(resultReal[1] * resultReal[1] + resultImag[1] * resultImag[1])

        XCTAssertGreaterThan(magK, magOther * 10,
            "Energy at bin \(k) should dominate")
    }

    func testBatchFFTMultipleBatches() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let n = 16
        let batchSize = 4

        // Create different signals for each batch
        var real = [Float](repeating: 0, count: n * batchSize)
        var imag = [Float](repeating: 0, count: n * batchSize)

        // Batch 0: DC
        for i in 0..<n {
            real[i] = 1
        }

        // Batch 1: Fundamental (bin 1)
        for i in 0..<n {
            real[n + i] = cos(2 * Float.pi * Float(i) / Float(n))
        }

        // Batch 2: Second harmonic (bin 2)
        for i in 0..<n {
            real[2 * n + i] = cos(2 * Float.pi * 2 * Float(i) / Float(n))
        }

        // Batch 3: Third harmonic (bin 3)
        for i in 0..<n {
            real[3 * n + i] = cos(2 * Float.pi * 3 * Float(i) / Float(n))
        }

        let (resultReal, resultImag) = try await metalEngine.batchFFT(
            real: real,
            imag: imag,
            n: n,
            batchSize: batchSize
        )

        // Check each batch's peak
        for batch in 0..<batchSize {
            var maxBin = 0
            var maxMag: Float = 0

            for bin in 0..<n {
                let idx = batch * n + bin
                let mag = sqrt(resultReal[idx] * resultReal[idx] + resultImag[idx] * resultImag[idx])
                if mag > maxMag {
                    maxMag = mag
                    maxBin = bin
                }
            }

            // Batch 0 should peak at bin 0, others at their respective bins
            let expectedBin = batch
            XCTAssertEqual(maxBin, expectedBin,
                "Batch \(batch) should have peak at bin \(expectedBin), got bin \(maxBin)")
        }
    }

    func testBatchFFTMatchesCPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let n = 256

        // Random signal
        var real = [Float](repeating: 0, count: n)
        for i in 0..<n {
            real[i] = Float.random(in: -1...1)
        }
        let imag = [Float](repeating: 0, count: n)

        // GPU FFT
        let (gpuReal, gpuImag) = try await metalEngine.batchFFT(
            real: real,
            imag: imag,
            n: n,
            batchSize: 1
        )

        // CPU FFT via STFT (transform single frame)
        let stft = STFT(config: STFTConfig(uncheckedNFFT: n, hopLength: n, windowType: .rectangular, center: false))
        let cpuResult = await stft.transform(real)

        // Compare first nFreqs bins (GPU has full n, CPU has n/2+1)
        let nFreqs = n / 2 + 1
        for bin in 0..<nFreqs {
            let gpuMag = sqrt(gpuReal[bin] * gpuReal[bin] + gpuImag[bin] * gpuImag[bin])
            let cpuMag = sqrt(cpuResult.real[bin][0] * cpuResult.real[bin][0] +
                             cpuResult.imag[bin][0] * cpuResult.imag[bin][0])

            // Allow some tolerance due to windowing differences
            XCTAssertEqual(gpuMag, cpuMag, accuracy: max(cpuMag * 0.1, 1e-3),
                "Magnitude mismatch at bin \(bin): GPU=\(gpuMag), CPU=\(cpuMag)")
        }
    }

    func testSTFTMagnitudeGPU() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let nFFT = 256
        let nFrames = 10

        // Create windowed frames (just random data for shape test)
        var frames = [Float](repeating: 0, count: nFrames * nFFT)
        for i in 0..<frames.count {
            frames[i] = Float.random(in: -1...1)
        }

        let result = try await metalEngine.stftMagnitude(
            frames: frames,
            nFFT: nFFT,
            nFrames: nFrames
        )

        // Check shape
        let nFreqs = nFFT / 2 + 1
        XCTAssertEqual(result.rows, nFreqs)
        XCTAssertEqual(result.cols, nFrames)

        // All magnitudes should be non-negative
        for f in 0..<result.rows {
            for t in 0..<result.cols {
                XCTAssertGreaterThanOrEqual(result[f, t], 0)
            }
        }
    }

    // MARK: - Edge Cases

    func testContiguousMatrixEmptyConversion() {
        let empty: [[Float]] = []
        let matrix = ContiguousMatrix(from: empty)

        XCTAssertEqual(matrix.rows, 0)
        XCTAssertEqual(matrix.cols, 0)
        XCTAssertEqual(matrix.count, 0)
    }

    func testSpectralFeaturesContiguousEmptyInput() {
        let spectralFeatures = SpectralFeatures()
        let emptyMag = ContiguousMatrix(rows: 0, cols: 0)

        let centroid = spectralFeatures.centroidContiguous(magnitudeSpec: emptyMag)
        let bandwidth = spectralFeatures.bandwidthContiguous(magnitudeSpec: emptyMag)
        let rolloff = spectralFeatures.rolloffContiguous(magnitudeSpec: emptyMag)
        let flatness = spectralFeatures.flatnessContiguous(magnitudeSpec: emptyMag)
        let flux = spectralFeatures.fluxContiguous(magnitudeSpec: emptyMag)
        let contrast = spectralFeatures.contrastContiguous(magnitudeSpec: emptyMag)

        XCTAssertEqual(centroid.count, 0)
        XCTAssertEqual(bandwidth.count, 0)
        XCTAssertEqual(rolloff.count, 0)
        XCTAssertEqual(flatness.count, 0)
        XCTAssertEqual(flux.count, 0)
        XCTAssertEqual(contrast.count, 0)
    }
}
