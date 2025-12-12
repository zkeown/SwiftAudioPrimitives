import SwiftRosaCore
import Foundation

/// Audio processing throughput metrics.
///
/// Measures how efficiently audio data is processed, including
/// real-time factor (RTF) which indicates whether processing
/// is faster or slower than real-time playback.
public struct ThroughputMetrics: Codable, Sendable {
    /// Number of audio samples processed.
    public let samplesProcessed: Int
    /// Sample rate of the audio in Hz.
    public let sampleRate: Float
    /// Processing time in seconds.
    public let processingTime: Double
    /// Number of STFT frames processed (optional).
    public let framesProcessed: Int?

    /// Create throughput metrics for an audio processing operation.
    ///
    /// - Parameters:
    ///   - samplesProcessed: Number of audio samples processed.
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - processingTime: Time taken to process in seconds.
    ///   - framesProcessed: Optional number of STFT frames processed.
    public init(
        samplesProcessed: Int,
        sampleRate: Float = 22050,
        processingTime: Double,
        framesProcessed: Int? = nil
    ) {
        self.samplesProcessed = samplesProcessed
        self.sampleRate = sampleRate
        self.processingTime = processingTime
        self.framesProcessed = framesProcessed
    }

    /// Samples processed per second.
    public var samplesPerSecond: Double {
        guard processingTime > 0 else { return 0 }
        return Double(samplesProcessed) / processingTime
    }

    /// Real-time factor (RTF).
    ///
    /// - > 1.0: Processing is faster than real-time (good for real-time applications)
    /// - = 1.0: Processing matches real-time
    /// - < 1.0: Processing is slower than real-time
    public var realTimeFactor: Double {
        guard processingTime > 0 else { return 0 }
        let audioDuration = Double(samplesProcessed) / Double(sampleRate)
        return audioDuration / processingTime
    }

    /// Frames processed per second (for STFT-based operations).
    public var framesPerSecond: Double? {
        guard let frames = framesProcessed, processingTime > 0 else { return nil }
        return Double(frames) / processingTime
    }

    /// Duration of the processed audio in seconds.
    public var audioDuration: Double {
        Double(samplesProcessed) / Double(sampleRate)
    }

    /// Format the real-time factor as a readable string.
    public var formattedRTF: String {
        String(format: "%.2fx real-time", realTimeFactor)
    }

    /// Format the throughput as samples per second.
    public var formattedThroughput: String {
        if samplesPerSecond >= 1_000_000 {
            return String(format: "%.2f M samples/s", samplesPerSecond / 1_000_000)
        } else if samplesPerSecond >= 1000 {
            return String(format: "%.2f K samples/s", samplesPerSecond / 1000)
        } else {
            return String(format: "%.0f samples/s", samplesPerSecond)
        }
    }
}

/// Streaming latency metrics for real-time audio processing.
public struct StreamingLatencyMetrics: Codable, Sendable {
    /// Per-frame processing latencies in seconds.
    public let frameTimes: [Double]
    /// Sample rate in Hz.
    public let sampleRate: Float
    /// Hop length in samples.
    public let hopLength: Int

    /// Create streaming latency metrics.
    ///
    /// - Parameters:
    ///   - frameTimes: Array of per-frame processing times in seconds.
    ///   - sampleRate: Audio sample rate in Hz.
    ///   - hopLength: STFT hop length in samples.
    public init(frameTimes: [Double], sampleRate: Float, hopLength: Int) {
        self.frameTimes = frameTimes
        self.sampleRate = sampleRate
        self.hopLength = hopLength
    }

    /// Mean frame processing time in seconds.
    public var meanLatency: Double {
        guard !frameTimes.isEmpty else { return 0 }
        return frameTimes.reduce(0, +) / Double(frameTimes.count)
    }

    /// Maximum frame processing time in seconds.
    public var maxLatency: Double {
        frameTimes.max() ?? 0
    }

    /// Minimum frame processing time in seconds.
    public var minLatency: Double {
        frameTimes.min() ?? 0
    }

    /// 95th percentile latency in seconds.
    public var p95Latency: Double {
        guard !frameTimes.isEmpty else { return 0 }
        let sorted = frameTimes.sorted()
        let index = Int(Double(sorted.count) * 0.95)
        return sorted[min(index, sorted.count - 1)]
    }

    /// 99th percentile latency in seconds.
    public var p99Latency: Double {
        guard !frameTimes.isEmpty else { return 0 }
        let sorted = frameTimes.sorted()
        let index = Int(Double(sorted.count) * 0.99)
        return sorted[min(index, sorted.count - 1)]
    }

    /// Time budget per frame to maintain real-time processing.
    ///
    /// This is the maximum time allowed per frame to keep up with
    /// incoming audio data.
    public var frameBudget: Double {
        Double(hopLength) / Double(sampleRate)
    }

    /// Whether processing can keep up with real-time audio.
    ///
    /// Returns true if the 95th percentile latency is within the frame budget.
    public var isRealTimeCapable: Bool {
        p95Latency <= frameBudget
    }

    /// Headroom as a percentage of frame budget.
    ///
    /// Positive values indicate spare capacity, negative indicates overrun.
    public var headroomPercent: Double {
        guard frameBudget > 0 else { return 0 }
        return ((frameBudget - meanLatency) / frameBudget) * 100
    }

    /// Format mean latency as a readable string.
    public var formattedMeanLatency: String {
        formatLatency(meanLatency)
    }

    /// Format P95 latency as a readable string.
    public var formattedP95Latency: String {
        formatLatency(p95Latency)
    }

    private func formatLatency(_ seconds: Double) -> String {
        if seconds < 0.001 {
            return String(format: "%.1f μs", seconds * 1_000_000)
        } else {
            return String(format: "%.2f ms", seconds * 1000)
        }
    }
}
