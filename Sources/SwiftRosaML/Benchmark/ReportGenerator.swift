import SwiftRosaCore
import Foundation

/// Generates reports from benchmark results.
public struct ReportGenerator: Sendable {
    public init() {}

    /// Generate a Markdown report from benchmark results.
    ///
    /// - Parameters:
    ///   - results: Benchmark results to include.
    ///   - baselines: Optional baselines for comparison.
    ///   - title: Report title.
    /// - Returns: Markdown-formatted report string.
    public func generateMarkdown(
        results: [BenchmarkResult],
        baselines: [String: BenchmarkBaseline]? = nil,
        title: String = "SwiftAudioPrimitives Benchmark Report"
    ) -> String {
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime]

        var md = """
        # \(title)

        Generated: \(dateFormatter.string(from: Date()))

        ## Summary

        | Benchmark | Mean | Std Dev | P95 | Throughput | Memory | vs Baseline |
        |-----------|------|---------|-----|------------|--------|-------------|

        """

        let detector = RegressionDetector()

        for result in results {
            let throughputStr = result.throughput?.formattedRTF ?? "N/A"
            let memoryStr = result.formattedPeakMemory

            var baselineStr = "N/A"
            if let baselines = baselines, let baseline = baselines[result.name] {
                let percentChange = (result.wallTimeStats.mean - baseline.mean) / baseline.mean * 100
                if percentChange > 10 {
                    baselineStr = String(format: "**+%.1f%%**", percentChange)
                } else if percentChange < -10 {
                    baselineStr = String(format: "-%.1f%%", abs(percentChange))
                } else {
                    baselineStr = "~"
                }
            }

            md += """
            | \(result.name) | \(result.wallTimeStats.formattedMean) | \(result.wallTimeStats.formattedStdDev) | \(result.wallTimeStats.formattedP95) | \(throughputStr) | \(memoryStr) | \(baselineStr) |

            """
        }

        // Add detailed sections
        md += """

        ## Detailed Results

        """

        for result in results {
            md += """

        ### \(result.name)

        **Parameters:** \(result.parameters.isEmpty ? "None" : result.parameters.map { "\($0.key)=\($0.value)" }.joined(separator: ", "))

        **Timing:**
        - Mean: \(result.wallTimeStats.formattedMean)
        - Median: \(formatDuration(result.wallTimeStats.median))
        - Std Dev: \(result.wallTimeStats.formattedStdDev)
        - Min: \(formatDuration(result.wallTimeStats.min))
        - Max: \(formatDuration(result.wallTimeStats.max))
        - P95: \(result.wallTimeStats.formattedP95)
        - P99: \(formatDuration(result.wallTimeStats.p99))
        - CV: \(String(format: "%.2f%%", result.wallTimeStats.coefficientOfVariation * 100))

        """

            if let throughput = result.throughput {
                md += """
        **Throughput:**
        - Samples/sec: \(throughput.formattedThroughput)
        - Real-time factor: \(throughput.formattedRTF)
        - Audio duration: \(String(format: "%.2fs", throughput.audioDuration))

        """
            }

            if let delta = result.memoryDelta {
                md += """
        **Memory:**
        - Peak: \(delta.formattedPeak)
        - Delta: \(delta.formattedAllocated)

        """
            }
        }

        return md
    }

    /// Generate a JSON report from benchmark results.
    ///
    /// - Parameters:
    ///   - results: Benchmark results to include.
    ///   - baselines: Optional baselines for comparison.
    /// - Returns: JSON-formatted report data.
    public func generateJSON(
        results: [BenchmarkResult],
        baselines: [String: BenchmarkBaseline]? = nil
    ) throws -> Data {
        let report = JSONReport(
            version: 1,
            timestamp: Date(),
            results: results,
            baselines: baselines
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        return try encoder.encode(report)
    }

    /// Generate a CSV report from benchmark results.
    ///
    /// - Parameter results: Benchmark results to include.
    /// - Returns: CSV-formatted report string.
    public func generateCSV(results: [BenchmarkResult]) -> String {
        var csv = "name,mean_ms,stddev_ms,p95_ms,peak_memory_bytes,rtf,samples_processed\n"

        for result in results {
            let meanMs = result.wallTimeStats.mean * 1000
            let stddevMs = result.wallTimeStats.standardDeviation * 1000
            let p95Ms = result.wallTimeStats.p95 * 1000
            let peakMemory = result.peakMemory ?? 0
            let rtf = result.throughput?.realTimeFactor ?? 0
            let samples = result.throughput?.samplesProcessed ?? 0

            csv += "\(result.name),\(meanMs),\(stddevMs),\(p95Ms),\(peakMemory),\(rtf),\(samples)\n"
        }

        return csv
    }

    /// Export results to a file.
    ///
    /// - Parameters:
    ///   - results: Benchmark results to export.
    ///   - url: Destination URL.
    ///   - format: Output format.
    public func export(
        results: [BenchmarkResult],
        to url: URL,
        format: ExportFormat,
        baselines: [String: BenchmarkBaseline]? = nil
    ) throws {
        let data: Data

        switch format {
        case .json:
            data = try generateJSON(results: results, baselines: baselines)
        case .csv:
            data = generateCSV(results: results).data(using: .utf8)!
        case .markdown:
            data = generateMarkdown(results: results, baselines: baselines).data(using: .utf8)!
        }

        try data.write(to: url)
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 0.001 {
            return String(format: "%.2f μs", seconds * 1_000_000)
        } else if seconds < 1.0 {
            return String(format: "%.2f ms", seconds * 1000)
        } else {
            return String(format: "%.3f s", seconds)
        }
    }
}

/// Export format for benchmark reports.
public enum ExportFormat: String, Sendable {
    case json
    case csv
    case markdown
}

/// JSON report structure.
private struct JSONReport: Codable {
    let version: Int
    let timestamp: Date
    let results: [BenchmarkResult]
    let baselines: [String: BenchmarkBaseline]?
}

/// Console output helpers for benchmark results.
public struct BenchmarkPrinter: Sendable {
    public init() {}

    /// Print a summary table to the console.
    public func printSummary(_ results: [BenchmarkResult]) {
        print("\n" + String(repeating: "=", count: 80))
        print("BENCHMARK RESULTS")
        print(String(repeating: "=", count: 80))

        let nameWidth = max(results.map { $0.name.count }.max() ?? 20, 20)

        print(String(format: "%-\(nameWidth)s  %12s  %12s  %12s  %10s",
                     "Benchmark", "Mean", "StdDev", "P95", "RTF"))
        print(String(repeating: "-", count: 80))

        for result in results {
            let rtfStr = result.throughput.map { String(format: "%.2fx", $0.realTimeFactor) } ?? "N/A"

            print(String(format: "%-\(nameWidth)s  %12s  %12s  %12s  %10s",
                         result.name,
                         result.wallTimeStats.formattedMean,
                         result.wallTimeStats.formattedStdDev,
                         result.wallTimeStats.formattedP95,
                         rtfStr))
        }

        print(String(repeating: "=", count: 80) + "\n")
    }

    /// Print a single result with details.
    public func printDetailed(_ result: BenchmarkResult) {
        print("\n--- \(result.name) ---")
        print("  Timing:")
        print("    Mean:   \(result.wallTimeStats.formattedMean)")
        print("    StdDev: \(result.wallTimeStats.formattedStdDev)")
        print("    P95:    \(result.wallTimeStats.formattedP95)")

        if let throughput = result.throughput {
            print("  Throughput:")
            print("    RTF:    \(throughput.formattedRTF)")
            print("    Rate:   \(throughput.formattedThroughput)")
        }

        if let delta = result.memoryDelta {
            print("  Memory:")
            print("    Peak:   \(delta.formattedPeak)")
            print("    Delta:  \(delta.formattedAllocated)")
        }
    }

    /// Print a regression report.
    public func printRegressionReport(_ report: RegressionReport) {
        print("\n" + String(repeating: "=", count: 60))
        print("REGRESSION ANALYSIS")
        print(String(repeating: "=", count: 60))

        if !report.regressions.isEmpty {
            print("\n⚠️  REGRESSIONS:")
            for item in report.regressions {
                print("  • \(item.name): \(item.formattedChange)")
            }
        }

        if !report.improvements.isEmpty {
            print("\n✅ IMPROVEMENTS:")
            for item in report.improvements {
                print("  • \(item.name): \(item.formattedChange)")
            }
        }

        if !report.unchanged.isEmpty {
            print("\n📊 Unchanged: \(report.unchanged.count) benchmarks")
        }

        if !report.noBaseline.isEmpty {
            print("\n❓ No baseline: \(report.noBaseline.joined(separator: ", "))")
        }

        print("\nSummary: \(report.summary)")
        print(String(repeating: "=", count: 60) + "\n")
    }
}
