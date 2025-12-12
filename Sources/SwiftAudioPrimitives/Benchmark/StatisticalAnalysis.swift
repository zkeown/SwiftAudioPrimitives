import Accelerate
import Foundation

/// Statistical summary of benchmark measurements.
public struct BenchmarkStatistics: Codable, Sendable {
    /// Number of samples in the analysis.
    public let sampleCount: Int
    /// Arithmetic mean of the samples.
    public let mean: Double
    /// Median (50th percentile) of the samples.
    public let median: Double
    /// Minimum value.
    public let min: Double
    /// Maximum value.
    public let max: Double
    /// Standard deviation of the samples.
    public let standardDeviation: Double
    /// Variance of the samples.
    public let variance: Double
    /// 50th percentile (same as median).
    public let p50: Double
    /// 90th percentile.
    public let p90: Double
    /// 95th percentile.
    public let p95: Double
    /// 99th percentile.
    public let p99: Double

    /// Coefficient of variation (CV = stdDev / mean).
    ///
    /// Lower values indicate more consistent measurements.
    /// - < 0.1: Highly consistent
    /// - 0.1 - 0.3: Moderately consistent
    /// - > 0.3: High variability
    public var coefficientOfVariation: Double {
        guard mean > 0 else { return 0 }
        return standardDeviation / mean
    }

    /// 95% confidence interval for the mean.
    ///
    /// Returns the range within which the true mean is expected to fall
    /// with 95% confidence.
    public var confidenceInterval95: (lower: Double, upper: Double) {
        let z = 1.96 // Z-score for 95% confidence
        let margin = z * standardDeviation / sqrt(Double(sampleCount))
        return (mean - margin, mean + margin)
    }

    /// Create statistics from an array of sample values.
    ///
    /// - Parameters:
    ///   - samples: The raw measurement values.
    ///   - removeOutliers: Whether to remove outliers using the IQR method (default: true).
    public init(samples: [Double], removeOutliers: Bool = true) {
        var data = samples
        if removeOutliers && samples.count >= 4 {
            data = Self.removeOutliersIQR(samples)
        }

        guard !data.isEmpty else {
            self.sampleCount = 0
            self.mean = 0
            self.median = 0
            self.min = 0
            self.max = 0
            self.standardDeviation = 0
            self.variance = 0
            self.p50 = 0
            self.p90 = 0
            self.p95 = 0
            self.p99 = 0
            return
        }

        self.sampleCount = data.count

        // Calculate mean using vDSP
        var meanValue: Double = 0
        vDSP_meanvD(data, 1, &meanValue, vDSP_Length(data.count))
        self.mean = meanValue

        // Sort for percentiles
        let sorted = data.sorted()

        // Min and max
        self.min = sorted.first!
        self.max = sorted.last!

        // Percentiles
        self.p50 = Self.percentile(sorted, p: 0.50)
        self.median = self.p50
        self.p90 = Self.percentile(sorted, p: 0.90)
        self.p95 = Self.percentile(sorted, p: 0.95)
        self.p99 = Self.percentile(sorted, p: 0.99)

        // Variance and standard deviation
        if data.count > 1 {
            var sumSquaredDiff: Double = 0
            for value in data {
                let diff = value - meanValue
                sumSquaredDiff += diff * diff
            }
            self.variance = sumSquaredDiff / Double(data.count - 1)
            self.standardDeviation = sqrt(self.variance)
        } else {
            self.variance = 0
            self.standardDeviation = 0
        }
    }

    /// Remove outliers using the Interquartile Range (IQR) method.
    ///
    /// Removes values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    private static func removeOutliersIQR(_ samples: [Double]) -> [Double] {
        let sorted = samples.sorted()
        let q1Index = sorted.count / 4
        let q3Index = 3 * sorted.count / 4

        let q1 = sorted[q1Index]
        let q3 = sorted[q3Index]
        let iqr = q3 - q1

        let lowerBound = q1 - 1.5 * iqr
        let upperBound = q3 + 1.5 * iqr

        return sorted.filter { $0 >= lowerBound && $0 <= upperBound }
    }

    /// Calculate the p-th percentile of sorted data.
    private static func percentile(_ sorted: [Double], p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = p * Double(sorted.count - 1)
        let lower = Int(floor(index))
        let upper = Int(ceil(index))

        if lower == upper || upper >= sorted.count {
            return sorted[Swift.min(lower, sorted.count - 1)]
        }

        let fraction = index - Double(lower)
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower])
    }

    /// Format the mean as a time string.
    public var formattedMean: String {
        formatDuration(mean)
    }

    /// Format the standard deviation as a time string.
    public var formattedStdDev: String {
        formatDuration(standardDeviation)
    }

    /// Format the P95 as a time string.
    public var formattedP95: String {
        formatDuration(p95)
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

/// Comparison result between two benchmark runs.
public struct BenchmarkComparison: Sendable {
    /// The baseline statistics.
    public let baseline: BenchmarkStatistics
    /// The current statistics.
    public let current: BenchmarkStatistics
    /// Percent change in mean (positive = slower, negative = faster).
    public let percentChange: Double
    /// Whether the change is statistically significant.
    public let isSignificant: Bool
    /// Classification of the result.
    public let result: ComparisonResult

    public enum ComparisonResult: Sendable {
        case regression(percentage: Double)
        case improvement(percentage: Double)
        case noChange
    }

    /// Create a comparison between baseline and current statistics.
    ///
    /// - Parameters:
    ///   - baseline: The baseline (previous) statistics.
    ///   - current: The current statistics.
    ///   - regressionThreshold: Minimum percent increase to count as regression (default: 10%).
    ///   - improvementThreshold: Minimum percent decrease to count as improvement (default: 10%).
    public init(
        baseline: BenchmarkStatistics,
        current: BenchmarkStatistics,
        regressionThreshold: Double = 0.10,
        improvementThreshold: Double = 0.10
    ) {
        self.baseline = baseline
        self.current = current

        // Calculate percent change
        if baseline.mean > 0 {
            self.percentChange = (current.mean - baseline.mean) / baseline.mean
        } else {
            self.percentChange = 0
        }

        // Check statistical significance using Welch's t-test approximation
        let pooledStdErr = sqrt(
            (baseline.variance / Double(baseline.sampleCount)) +
            (current.variance / Double(current.sampleCount))
        )
        let tStatistic: Double
        if pooledStdErr > 0 {
            tStatistic = abs(current.mean - baseline.mean) / pooledStdErr
        } else {
            tStatistic = 0
        }

        // t > 2.0 roughly corresponds to p < 0.05 for reasonable sample sizes
        self.isSignificant = tStatistic >= 2.0

        // Classify result
        if isSignificant && percentChange > regressionThreshold {
            self.result = .regression(percentage: percentChange * 100)
        } else if isSignificant && percentChange < -improvementThreshold {
            self.result = .improvement(percentage: abs(percentChange) * 100)
        } else {
            self.result = .noChange
        }
    }

    /// Format the result as a string.
    public var formattedResult: String {
        switch result {
        case .regression(let pct):
            return String(format: "+%.1f%% (regression)", pct)
        case .improvement(let pct):
            return String(format: "-%.1f%% (improvement)", pct)
        case .noChange:
            return "~0% (no significant change)"
        }
    }
}
