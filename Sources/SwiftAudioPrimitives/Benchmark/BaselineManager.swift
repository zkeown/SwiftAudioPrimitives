import Foundation

/// A stored baseline for a benchmark.
public struct BenchmarkBaseline: Codable, Sendable {
    /// Name of the benchmark.
    public let name: String
    /// Mean execution time in seconds.
    public let mean: Double
    /// Standard deviation in seconds.
    public let standardDeviation: Double
    /// Parameters used for this baseline.
    public let parameters: [String: String]
    /// When the baseline was captured.
    public let capturedAt: Date
    /// Git commit hash (if available).
    public let gitCommit: String?
    /// Peak memory usage in bytes.
    public let peakMemory: UInt64?
    /// Real-time factor (if applicable).
    public let realTimeFactor: Double?

    /// Create a baseline from a benchmark result.
    public init(from result: BenchmarkResult, gitCommit: String? = nil) {
        self.name = result.name
        self.mean = result.wallTimeStats.mean
        self.standardDeviation = result.wallTimeStats.standardDeviation
        self.parameters = result.parameters
        self.capturedAt = result.timestamp
        self.gitCommit = gitCommit
        self.peakMemory = result.peakMemory
        self.realTimeFactor = result.throughput?.realTimeFactor
    }
}

/// Container for all baselines.
public struct BaselineCollection: Codable, Sendable {
    /// Version of the baseline format.
    public var version: Int = 1
    /// When the collection was last updated.
    public var lastUpdated: Date
    /// Map of benchmark name to baseline.
    public var baselines: [String: BenchmarkBaseline]

    public init() {
        self.lastUpdated = Date()
        self.baselines = [:]
    }
}

/// Manages saving and loading of benchmark baselines.
///
/// Baselines are used to detect performance regressions by comparing
/// current benchmark results against previously captured values.
///
/// ## Example Usage
///
/// ```swift
/// let manager = BaselineManager(url: baselineURL)
/// try await manager.loadBaselines()
///
/// // After running benchmarks
/// for result in runner.allResults() {
///     await manager.setBaseline(for: result)
/// }
///
/// try await manager.saveBaselines()
/// ```
public actor BaselineManager {
    /// URL where baselines are stored.
    public let baselineURL: URL
    /// Current baseline collection.
    private var collection: BaselineCollection

    /// Create a baseline manager.
    ///
    /// - Parameter baselineURL: URL to the baseline JSON file.
    public init(baselineURL: URL) {
        self.baselineURL = baselineURL
        self.collection = BaselineCollection()
    }

    /// Create a baseline manager with a default path.
    ///
    /// Uses `.benchmarks/baselines.json` in the current directory.
    public init() {
        let defaultURL = URL(fileURLWithPath: ".benchmarks/baselines.json")
        self.baselineURL = defaultURL
        self.collection = BaselineCollection()
    }

    /// Load baselines from disk.
    ///
    /// If the file doesn't exist, starts with an empty collection.
    public func loadBaselines() throws {
        guard FileManager.default.fileExists(atPath: baselineURL.path) else {
            collection = BaselineCollection()
            return
        }

        let data = try Data(contentsOf: baselineURL)
        collection = try JSONDecoder().decode(BaselineCollection.self, from: data)
    }

    /// Save baselines to disk.
    ///
    /// Creates the parent directory if it doesn't exist.
    public func saveBaselines() throws {
        // Create directory if needed
        let directory = baselineURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        collection.lastUpdated = Date()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(collection)
        try data.write(to: baselineURL)
    }

    /// Set or update a baseline from a benchmark result.
    ///
    /// - Parameters:
    ///   - result: The benchmark result to use as baseline.
    ///   - gitCommit: Optional git commit hash.
    public func setBaseline(for result: BenchmarkResult, gitCommit: String? = nil) {
        let commit = gitCommit ?? getCurrentGitCommit()
        let baseline = BenchmarkBaseline(from: result, gitCommit: commit)
        collection.baselines[result.name] = baseline
    }

    /// Get the baseline for a benchmark.
    ///
    /// - Parameter name: Benchmark name.
    /// - Returns: The baseline if it exists.
    public func getBaseline(for name: String) -> BenchmarkBaseline? {
        collection.baselines[name]
    }

    /// Get all stored baselines.
    public func allBaselines() -> [String: BenchmarkBaseline] {
        collection.baselines
    }

    /// Remove a baseline.
    ///
    /// - Parameter name: Benchmark name to remove.
    public func removeBaseline(for name: String) {
        collection.baselines.removeValue(forKey: name)
    }

    /// Clear all baselines.
    public func clearAllBaselines() {
        collection.baselines.removeAll()
    }

    /// Compare a result against its baseline.
    ///
    /// - Parameter result: The benchmark result to compare.
    /// - Returns: Comparison result if a baseline exists, nil otherwise.
    public func compare(_ result: BenchmarkResult) -> BenchmarkComparison? {
        guard let baseline = collection.baselines[result.name] else {
            return nil
        }

        // Reconstruct baseline statistics for comparison
        let baselineStats = BenchmarkStatistics(
            samples: [baseline.mean],
            removeOutliers: false
        )

        return BenchmarkComparison(
            baseline: BenchmarkStatistics(
                samples: Array(repeating: baseline.mean, count: 10),
                removeOutliers: false
            ),
            current: result.wallTimeStats
        )
    }

    /// Get the current git commit hash.
    private func getCurrentGitCommit() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["rev-parse", "--short", "HEAD"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }
}

/// Detects performance regressions by comparing results to baselines.
public struct RegressionDetector: Sendable {
    /// Threshold for detecting regressions (e.g., 0.10 = 10% slower).
    public let regressionThreshold: Double
    /// Threshold for detecting improvements (e.g., 0.10 = 10% faster).
    public let improvementThreshold: Double

    /// Create a regression detector.
    ///
    /// - Parameters:
    ///   - regressionThreshold: Minimum percent increase to flag as regression (default: 10%).
    ///   - improvementThreshold: Minimum percent decrease to flag as improvement (default: 10%).
    public init(regressionThreshold: Double = 0.10, improvementThreshold: Double = 0.10) {
        self.regressionThreshold = regressionThreshold
        self.improvementThreshold = improvementThreshold
    }

    /// Analyze results against baselines.
    ///
    /// - Parameters:
    ///   - results: Benchmark results to analyze.
    ///   - baselines: Baseline values to compare against.
    /// - Returns: Analysis report.
    public func analyze(
        results: [BenchmarkResult],
        baselines: [String: BenchmarkBaseline]
    ) -> RegressionReport {
        var regressions: [RegressionItem] = []
        var improvements: [RegressionItem] = []
        var unchanged: [String] = []
        var noBaseline: [String] = []

        for result in results {
            guard let baseline = baselines[result.name] else {
                noBaseline.append(result.name)
                continue
            }

            let percentChange = (result.wallTimeStats.mean - baseline.mean) / baseline.mean

            if percentChange > regressionThreshold {
                regressions.append(RegressionItem(
                    name: result.name,
                    baselineMean: baseline.mean,
                    currentMean: result.wallTimeStats.mean,
                    percentChange: percentChange * 100
                ))
            } else if percentChange < -improvementThreshold {
                improvements.append(RegressionItem(
                    name: result.name,
                    baselineMean: baseline.mean,
                    currentMean: result.wallTimeStats.mean,
                    percentChange: percentChange * 100
                ))
            } else {
                unchanged.append(result.name)
            }
        }

        return RegressionReport(
            regressions: regressions,
            improvements: improvements,
            unchanged: unchanged,
            noBaseline: noBaseline,
            timestamp: Date()
        )
    }
}

/// An item in the regression report.
public struct RegressionItem: Sendable {
    public let name: String
    public let baselineMean: Double
    public let currentMean: Double
    public let percentChange: Double

    public var formattedChange: String {
        String(format: "%+.1f%%", percentChange)
    }
}

/// Report of regression analysis.
public struct RegressionReport: Sendable {
    /// Benchmarks that regressed.
    public let regressions: [RegressionItem]
    /// Benchmarks that improved.
    public let improvements: [RegressionItem]
    /// Benchmarks with no significant change.
    public let unchanged: [String]
    /// Benchmarks without baselines.
    public let noBaseline: [String]
    /// When the analysis was performed.
    public let timestamp: Date

    /// Whether any regressions were detected.
    public var hasRegressions: Bool {
        !regressions.isEmpty
    }

    /// Summary string.
    public var summary: String {
        var parts: [String] = []

        if !regressions.isEmpty {
            parts.append("\(regressions.count) regression(s)")
        }
        if !improvements.isEmpty {
            parts.append("\(improvements.count) improvement(s)")
        }
        if !unchanged.isEmpty {
            parts.append("\(unchanged.count) unchanged")
        }
        if !noBaseline.isEmpty {
            parts.append("\(noBaseline.count) without baseline")
        }

        return parts.joined(separator: ", ")
    }
}
