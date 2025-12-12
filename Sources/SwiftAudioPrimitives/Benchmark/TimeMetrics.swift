import Foundation

/// High-precision time measurement for benchmark operations.
public struct TimeMeasurement: Codable, Sendable {
    /// Wall clock time in seconds.
    public let wallTime: Double
    /// Total CPU time in seconds (user + system).
    public let cpuTime: Double
    /// User CPU time in seconds.
    public let userTime: Double
    /// System CPU time in seconds.
    public let systemTime: Double

    /// CPU efficiency ratio (cpuTime / wallTime).
    ///
    /// Values > 1.0 indicate parallel execution across multiple cores.
    public var cpuEfficiency: Double {
        guard wallTime > 0 else { return 0 }
        return cpuTime / wallTime
    }

    /// Format wall time as a human-readable string.
    public var formattedWallTime: String {
        formatDuration(wallTime)
    }

    /// Format CPU time as a human-readable string.
    public var formattedCPUTime: String {
        formatDuration(cpuTime)
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

/// High-precision timer using mach_absolute_time for nanosecond accuracy.
///
/// ## Example Usage
///
/// ```swift
/// let timer = PrecisionTimer.start()
/// // Perform operation
/// let measurement = timer.stop()
/// print("Wall time: \(measurement.formattedWallTime)")
/// ```
public struct PrecisionTimer: Sendable {
    private let startWall: UInt64
    private let startUserMicros: UInt64
    private let startSystemMicros: UInt64
    private static let timebaseInfo: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    private init(startWall: UInt64, startUserMicros: UInt64, startSystemMicros: UInt64) {
        self.startWall = startWall
        self.startUserMicros = startUserMicros
        self.startSystemMicros = startSystemMicros
    }

    /// Start a new precision timer.
    ///
    /// - Returns: A timer that can be stopped to get the elapsed time.
    public static func start() -> PrecisionTimer {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)

        let userMicros = UInt64(usage.ru_utime.tv_sec) * 1_000_000 + UInt64(usage.ru_utime.tv_usec)
        let systemMicros = UInt64(usage.ru_stime.tv_sec) * 1_000_000 + UInt64(usage.ru_stime.tv_usec)

        return PrecisionTimer(
            startWall: mach_absolute_time(),
            startUserMicros: userMicros,
            startSystemMicros: systemMicros
        )
    }

    /// Stop the timer and return the elapsed time measurement.
    ///
    /// - Returns: The time measurement including wall clock and CPU times.
    public func stop() -> TimeMeasurement {
        let endWall = mach_absolute_time()

        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)

        let endUserMicros = UInt64(usage.ru_utime.tv_sec) * 1_000_000 + UInt64(usage.ru_utime.tv_usec)
        let endSystemMicros = UInt64(usage.ru_stime.tv_sec) * 1_000_000 + UInt64(usage.ru_stime.tv_usec)

        // Convert wall time from mach units to seconds
        let elapsed = endWall - startWall
        let nanos = elapsed * UInt64(Self.timebaseInfo.numer) / UInt64(Self.timebaseInfo.denom)
        let wallTime = Double(nanos) / 1_000_000_000.0

        // Convert CPU times from microseconds to seconds
        let userTime = Double(endUserMicros - startUserMicros) / 1_000_000.0
        let systemTime = Double(endSystemMicros - startSystemMicros) / 1_000_000.0

        return TimeMeasurement(
            wallTime: wallTime,
            cpuTime: userTime + systemTime,
            userTime: userTime,
            systemTime: systemTime
        )
    }
}

/// Simple wall clock timer for quick measurements.
public struct WallTimer: Sendable {
    private let startTime: UInt64
    private static let timebaseInfo: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    private init(startTime: UInt64) {
        self.startTime = startTime
    }

    /// Start a new wall clock timer.
    public static func start() -> WallTimer {
        WallTimer(startTime: mach_absolute_time())
    }

    /// Get the elapsed time in seconds.
    public func elapsed() -> Double {
        let now = mach_absolute_time()
        let elapsed = now - startTime
        let nanos = elapsed * UInt64(Self.timebaseInfo.numer) / UInt64(Self.timebaseInfo.denom)
        return Double(nanos) / 1_000_000_000.0
    }

    /// Get the elapsed time in milliseconds.
    public func elapsedMilliseconds() -> Double {
        elapsed() * 1000.0
    }

    /// Get the elapsed time in microseconds.
    public func elapsedMicroseconds() -> Double {
        elapsed() * 1_000_000.0
    }
}
