import XCTest
@testable import SwiftRosaCore

/// Benchmark tests comparing Tier 3 optimized (contiguous) implementations
/// against original array-of-arrays implementations.
final class Tier3BenchmarkTests: XCTestCase {

    // MARK: - Test Configuration

    static let sampleRate: Float = 22050

    // Pre-computed test signals
    private static let signal1s = generateTestSignalStatic(length: 22050)
    private static let signal10s = generateTestSignalStatic(length: 220500)

    private static func generateTestSignalStatic(length: Int) -> [Float] {
        (0..<length).map { i in
            let t = Float(i) / sampleRate
            return 0.5 * sin(2 * Float.pi * 440 * t) +
                   0.3 * sin(2 * Float.pi * 880 * t) +
                   0.2 * sin(2 * Float.pi * 1760 * t)
        }
    }

    // MARK: - STFT Benchmarks

    func testSTFTMagnitudeRegular_1s() async {
        let signal = Self.signal1s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))

        // Warmup
        _ = await stft.magnitude(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = await stft.magnitude(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("STFT magnitude (regular, 1s): \(String(format: "%.2f", avgMs)) ms")
    }

    func testSTFTMagnitudeContiguous_1s() async {
        let signal = Self.signal1s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))

        // Warmup
        _ = await stft.magnitudeContiguous(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = await stft.magnitudeContiguous(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("STFT magnitude (contiguous, 1s): \(String(format: "%.2f", avgMs)) ms")
    }

    func testSTFTMagnitudeRegular_10s() async {
        let signal = Self.signal10s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))

        // Warmup
        _ = await stft.magnitude(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 5

        for _ in 0..<iterations {
            _ = await stft.magnitude(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("STFT magnitude (regular, 10s): \(String(format: "%.2f", avgMs)) ms")
    }

    func testSTFTMagnitudeContiguous_10s() async {
        let signal = Self.signal10s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))

        // Warmup
        _ = await stft.magnitudeContiguous(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 5

        for _ in 0..<iterations {
            _ = await stft.magnitudeContiguous(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("STFT magnitude (contiguous, 10s): \(String(format: "%.2f", avgMs)) ms")
    }

    // MARK: - SpectralFeatures Benchmarks

    func testSpectralFeaturesExtractAll_Regular() async {
        let signal = Self.signal10s
        let spectralFeatures = SpectralFeatures()

        // Warmup
        _ = await spectralFeatures.extractAll(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 5

        for _ in 0..<iterations {
            _ = await spectralFeatures.extractAll(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("SpectralFeatures extractAll (regular, 10s): \(String(format: "%.2f", avgMs)) ms")
    }

    func testSpectralFeaturesExtractAll_Contiguous() async {
        let signal = Self.signal10s
        let spectralFeatures = SpectralFeatures()

        // Warmup
        _ = await spectralFeatures.extractAllContiguous(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 5

        for _ in 0..<iterations {
            _ = await spectralFeatures.extractAllContiguous(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("SpectralFeatures extractAll (contiguous, 10s): \(String(format: "%.2f", avgMs)) ms")
    }

    // MARK: - Individual Feature Benchmarks

    func testCentroidBenchmark_Regular() async {
        let signal = Self.signal10s
        let spectralFeatures = SpectralFeatures()

        // Warmup
        _ = await spectralFeatures.centroid(signal)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = await spectralFeatures.centroid(signal)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("Spectral centroid (regular, 10s): \(String(format: "%.2f", avgMs)) ms")
    }

    func testCentroidBenchmark_Contiguous() async {
        let signal = Self.signal10s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let spectralFeatures = SpectralFeatures()

        // Compute magnitude once (this is the optimized path)
        let mag = await stft.magnitudeContiguous(signal)

        // Warmup
        _ = spectralFeatures.centroidContiguous(magnitudeSpec: mag)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = spectralFeatures.centroidContiguous(magnitudeSpec: mag)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("Spectral centroid (contiguous, 10s, from cached mag): \(String(format: "%.2f", avgMs)) ms")
    }

    // MARK: - GPU FFT Benchmarks

    func testGPUFFTBenchmark() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let n = 2048
        let batchSize = 100

        var real = [Float](repeating: 0, count: n * batchSize)
        var imag = [Float](repeating: 0, count: n * batchSize)

        // Fill with test data
        for i in 0..<real.count {
            real[i] = Float.random(in: -1...1)
        }

        // Warmup
        _ = try await metalEngine.batchFFT(real: real, imag: imag, n: n, batchSize: batchSize)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = try await metalEngine.batchFFT(real: real, imag: imag, n: n, batchSize: batchSize)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("GPU batch FFT (\(batchSize) × \(n)): \(String(format: "%.2f", avgMs)) ms")
    }

    func testSTFTMagnitudeGPUBenchmark() async throws {
        guard MetalEngine.isAvailable else {
            throw XCTSkip("Metal not available")
        }

        let metalEngine = try MetalEngine()

        let nFFT = 2048
        let nFrames = 430  // ~10s of audio at hop=512

        // Create windowed frames
        var frames = [Float](repeating: 0, count: nFrames * nFFT)
        for i in 0..<frames.count {
            frames[i] = Float.random(in: -1...1)
        }

        // Warmup
        _ = try await metalEngine.stftMagnitude(frames: frames, nFFT: nFFT, nFrames: nFrames)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = try await metalEngine.stftMagnitude(frames: frames, nFFT: nFFT, nFrames: nFrames)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("GPU STFT magnitude (\(nFrames) frames × \(nFFT) FFT): \(String(format: "%.2f", avgMs)) ms")
    }

    // MARK: - Memory Layout Comparison

    func testContiguousMatrixConversionOverhead() {
        let rows = 1025  // Typical nFreqs
        let cols = 430   // Typical 10s frames

        // Create array-of-arrays
        var arrayOfArrays = [[Float]]()
        arrayOfArrays.reserveCapacity(rows)
        for r in 0..<rows {
            arrayOfArrays.append((0..<cols).map { Float($0 + r) })
        }

        // Measure conversion to contiguous
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 100

        for _ in 0..<iterations {
            let _ = ContiguousMatrix(from: arrayOfArrays)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("[[Float]] -> ContiguousMatrix (\(rows)×\(cols)): \(String(format: "%.2f", avgMs)) ms")
    }

    func testContiguousMatrixTo2DArrayOverhead() {
        let rows = 1025
        let cols = 430

        var data = [Float](repeating: 0, count: rows * cols)
        for i in 0..<data.count {
            data[i] = Float(i)
        }

        let matrix = ContiguousMatrix(data: data, rows: rows, cols: cols)

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 100

        for _ in 0..<iterations {
            let _ = matrix.to2DArray()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000

        print("ContiguousMatrix -> [[Float]] (\(rows)×\(cols)): \(String(format: "%.2f", avgMs)) ms")
    }

    // MARK: - Comparative Summary Test

    func testPrintPerformanceSummary() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("TIER 3 PERFORMANCE SUMMARY")
        print(String(repeating: "=", count: 60))

        let signal = Self.signal10s
        let stft = STFT(config: STFTConfig(uncheckedNFFT: 2048, hopLength: 512))
        let spectralFeatures = SpectralFeatures()

        // STFT comparison
        var startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 { _ = await stft.magnitude(signal) }
        let regularSTFT = (CFAbsoluteTimeGetCurrent() - startTime) / 3 * 1000

        startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 { _ = await stft.magnitudeContiguous(signal) }
        let contiguousSTFT = (CFAbsoluteTimeGetCurrent() - startTime) / 3 * 1000

        print("\nSTFT Magnitude (10s audio):")
        print("  Regular:    \(String(format: "%6.2f", regularSTFT)) ms")
        print("  Contiguous: \(String(format: "%6.2f", contiguousSTFT)) ms")
        print("  Speedup:    \(String(format: "%5.2f", regularSTFT / contiguousSTFT))x")

        // SpectralFeatures comparison
        startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 { _ = await spectralFeatures.extractAll(signal) }
        let regularFeatures = (CFAbsoluteTimeGetCurrent() - startTime) / 3 * 1000

        startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 { _ = await spectralFeatures.extractAllContiguous(signal) }
        let contiguousFeatures = (CFAbsoluteTimeGetCurrent() - startTime) / 3 * 1000

        print("\nSpectral Features (10s audio):")
        print("  Regular:    \(String(format: "%6.2f", regularFeatures)) ms")
        print("  Contiguous: \(String(format: "%6.2f", contiguousFeatures)) ms")
        print("  Speedup:    \(String(format: "%5.2f", regularFeatures / contiguousFeatures))x")

        // GPU FFT (if available)
        if MetalEngine.isAvailable, let metalEngine = try? MetalEngine() {
            let n = 2048
            let batchSize = 430  // Same as nFrames
            var real = [Float](repeating: 0, count: n * batchSize)
            var imag = [Float](repeating: 0, count: n * batchSize)
            for i in 0..<real.count { real[i] = Float.random(in: -1...1) }

            startTime = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                _ = try await metalEngine.batchFFT(real: real, imag: imag, n: n, batchSize: batchSize)
            }
            let gpuFFT = (CFAbsoluteTimeGetCurrent() - startTime) / 3 * 1000

            print("\nGPU Batch FFT (\(batchSize) × \(n)):")
            print("  GPU:        \(String(format: "%6.2f", gpuFFT)) ms")
        }

        print("\n" + String(repeating: "=", count: 60))
    }
}
