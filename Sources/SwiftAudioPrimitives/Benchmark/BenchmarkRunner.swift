import Foundation

/// Configuration for benchmark execution.
public struct BenchmarkConfiguration: Sendable {
    /// Number of warm-up iterations (results discarded).
    public var warmupIterations: Int
    /// Number of measurement iterations.
    public var measureIterations: Int
    /// Whether to remove outliers from results.
    public var removeOutliers: Bool
    /// Whether to capture memory metrics.
    public var captureMemory: Bool
    /// Whether to force garbage collection between iterations.
    public var gcBetweenIterations: Bool

    /// Create a benchmark configuration.
    ///
    /// - Parameters:
    ///   - warmupIterations: Warm-up iterations (default: 3).
    ///   - measureIterations: Measurement iterations (default: 10).
    ///   - removeOutliers: Remove statistical outliers (default: true).
    ///   - captureMemory: Track memory usage (default: true).
    ///   - gcBetweenIterations: Force GC between iterations (default: false).
    public init(
        warmupIterations: Int = 3,
        measureIterations: Int = 10,
        removeOutliers: Bool = true,
        captureMemory: Bool = true,
        gcBetweenIterations: Bool = false
    ) {
        self.warmupIterations = warmupIterations
        self.measureIterations = measureIterations
        self.removeOutliers = removeOutliers
        self.captureMemory = captureMemory
        self.gcBetweenIterations = gcBetweenIterations
    }

    /// Quick configuration for fast feedback (fewer iterations).
    public static let quick = BenchmarkConfiguration(
        warmupIterations: 1,
        measureIterations: 5,
        removeOutliers: true,
        captureMemory: true,
        gcBetweenIterations: false
    )

    /// Thorough configuration for accurate measurements.
    public static let thorough = BenchmarkConfiguration(
        warmupIterations: 5,
        measureIterations: 20,
        removeOutliers: true,
        captureMemory: true,
        gcBetweenIterations: true
    )

    /// Memory-focused configuration for allocation analysis.
    public static let memoryFocused = BenchmarkConfiguration(
        warmupIterations: 2,
        measureIterations: 5,
        removeOutliers: false,
        captureMemory: true,
        gcBetweenIterations: true
    )
}

/// Complete benchmark result including timing, memory, and throughput metrics.
public struct BenchmarkResult: Codable, Sendable {
    /// Name of the benchmark.
    public let name: String
    /// Parameters used for this benchmark run.
    public let parameters: [String: String]
    /// Timestamp when the benchmark was run.
    public let timestamp: Date

    /// Wall clock time statistics.
    public let wallTimeStats: BenchmarkStatistics
    /// CPU time statistics.
    public let cpuTimeStats: BenchmarkStatistics

    /// Memory delta (if memory tracking was enabled).
    public let memoryDelta: MemoryDelta?
    /// Peak memory usage in bytes.
    public let peakMemory: UInt64?

    /// Audio throughput metrics (if applicable).
    public let throughput: ThroughputMetrics?

    /// Raw wall time measurements for each iteration.
    public let rawWallTimes: [Double]
    /// Raw CPU time measurements for each iteration.
    public let rawCPUTimes: [Double]

    /// Format peak memory as a readable string.
    public var formattedPeakMemory: String {
        if let peak = peakMemory {
            return ByteCountFormatter.string(fromByteCount: Int64(peak), countStyle: .memory)
        }
        return "N/A"
    }

    /// Summary string for quick display.
    public var summary: String {
        var parts = [
            "Mean: \(wallTimeStats.formattedMean)",
            "StdDev: \(wallTimeStats.formattedStdDev)",
        ]

        if let throughput = throughput {
            parts.append("RTF: \(throughput.formattedRTF)")
        }

        if let peak = peakMemory {
            parts.append("Peak: \(ByteCountFormatter.string(fromByteCount: Int64(peak), countStyle: .memory))")
        }

        return parts.joined(separator: " | ")
    }
}

/// Main benchmark runner for executing and collecting benchmark results.
///
/// ## Example Usage
///
/// ```swift
/// let runner = BenchmarkRunner(config: .thorough)
///
/// let result = try await runner.run(
///     name: "STFT_10s",
///     parameters: ["nFFT": "2048"],
///     samplesProcessed: 220500,
///     sampleRate: 22050
/// ) {
///     _ = await stft.transform(signal)
/// }
///
/// print(result.summary)
/// ```
public actor BenchmarkRunner {
    private let config: BenchmarkConfiguration
    private var results: [BenchmarkResult] = []

    /// Create a benchmark runner with the given configuration.
    ///
    /// - Parameter config: Benchmark configuration (default: standard settings).
    public init(config: BenchmarkConfiguration = BenchmarkConfiguration()) {
        self.config = config
    }

    /// Run a benchmark and collect metrics.
    ///
    /// - Parameters:
    ///   - name: Name of the benchmark.
    ///   - parameters: Optional parameters for categorization.
    ///   - samplesProcessed: Number of audio samples processed (for throughput calculation).
    ///   - sampleRate: Audio sample rate in Hz.
    ///   - framesProcessed: Number of STFT frames processed (optional).
    ///   - setup: Optional setup closure run before each iteration.
    ///   - teardown: Optional teardown closure run after each iteration.
    ///   - block: The operation to benchmark.
    /// - Returns: Complete benchmark result with statistics.
    @discardableResult
    public func run<T>(
        name: String,
        parameters: [String: String] = [:],
        samplesProcessed: Int? = nil,
        sampleRate: Float = 22050,
        framesProcessed: Int? = nil,
        setup: (() async throws -> Void)? = nil,
        teardown: (() async throws -> Void)? = nil,
        block: () async throws -> T
    ) async throws -> BenchmarkResult {
        // Warm-up iterations (discarded)
        for _ in 0..<config.warmupIterations {
            try await setup?()
            _ = try await block()
            try await teardown?()
        }

        // Measurement iterations
        var wallTimes: [Double] = []
        var cpuTimes: [Double] = []
        wallTimes.reserveCapacity(config.measureIterations)
        cpuTimes.reserveCapacity(config.measureIterations)

        // Memory tracking
        var memoryTracker: MemoryTracker?
        if config.captureMemory {
            memoryTracker = MemoryTracker()
            await memoryTracker?.begin()
        }

        for _ in 0..<config.measureIterations {
            if config.gcBetweenIterations {
                // Force ARC collection
                autoreleasepool { }
            }

            try await setup?()

            let timer = PrecisionTimer.start()
            _ = try await block()
            let measurement = timer.stop()

            wallTimes.append(measurement.wallTime)
            cpuTimes.append(measurement.cpuTime)

            try await teardown?()

            // Sample memory periodically
            await memoryTracker?.sample()
        }

        let memoryDelta = await memoryTracker?.end()

        // Calculate throughput if samples provided
        let throughput: ThroughputMetrics?
        if let samples = samplesProcessed {
            let avgWallTime = wallTimes.reduce(0, +) / Double(wallTimes.count)
            throughput = ThroughputMetrics(
                samplesProcessed: samples,
                sampleRate: sampleRate,
                processingTime: avgWallTime,
                framesProcessed: framesProcessed
            )
        } else {
            throughput = nil
        }

        let result = BenchmarkResult(
            name: name,
            parameters: parameters,
            timestamp: Date(),
            wallTimeStats: BenchmarkStatistics(samples: wallTimes, removeOutliers: config.removeOutliers),
            cpuTimeStats: BenchmarkStatistics(samples: cpuTimes, removeOutliers: config.removeOutliers),
            memoryDelta: memoryDelta,
            peakMemory: memoryDelta?.peakFootprint,
            throughput: throughput,
            rawWallTimes: wallTimes,
            rawCPUTimes: cpuTimes
        )

        results.append(result)
        return result
    }

    /// Run a synchronous benchmark.
    ///
    /// Convenience method for benchmarking non-async operations.
    @discardableResult
    public func runSync<T>(
        name: String,
        parameters: [String: String] = [:],
        samplesProcessed: Int? = nil,
        sampleRate: Float = 22050,
        block: () throws -> T
    ) async throws -> BenchmarkResult {
        try await run(
            name: name,
            parameters: parameters,
            samplesProcessed: samplesProcessed,
            sampleRate: sampleRate
        ) {
            try block()
        }
    }

    /// Get all collected benchmark results.
    public func allResults() -> [BenchmarkResult] {
        results
    }

    /// Clear all collected results.
    public func clearResults() {
        results.removeAll()
    }

    /// Get the most recent result for a given benchmark name.
    public func latestResult(for name: String) -> BenchmarkResult? {
        results.last { $0.name == name }
    }
}

/// Convenience methods for common benchmark scenarios.
extension BenchmarkRunner {
    /// Run a series of benchmarks with different signal lengths.
    ///
    /// - Parameters:
    ///   - name: Base name for the benchmarks.
    ///   - signalLengths: Array of signal lengths to test.
    ///   - sampleRate: Sample rate in Hz.
    ///   - block: Closure that takes a signal and performs the operation.
    /// - Returns: Array of benchmark results.
    public func runWithSignalLengths(
        name: String,
        signalLengths: [Int],
        sampleRate: Float = 22050,
        signalGenerator: (Int) -> [Float],
        block: ([Float]) async throws -> Void
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for length in signalLengths {
            let signal = signalGenerator(length)
            let duration = Double(length) / Double(sampleRate)
            let durationStr = duration >= 60
                ? String(format: "%.0fmin", duration / 60)
                : String(format: "%.0fs", duration)

            let result = try await run(
                name: "\(name)_\(durationStr)",
                parameters: ["signalLength": "\(length)", "duration": durationStr],
                samplesProcessed: length,
                sampleRate: sampleRate
            ) {
                try await block(signal)
            }

            results.append(result)
        }

        return results
    }
}
