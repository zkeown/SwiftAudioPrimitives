import Foundation

/// A snapshot of the current memory state captured via mach APIs.
public struct MemorySnapshot: Codable, Sendable {
    /// Physical memory footprint in bytes.
    public let physicalFootprint: UInt64
    /// Internal memory (app's own allocations) in bytes.
    public let internalMemory: UInt64
    /// External memory (shared, mapped files) in bytes.
    public let externalMemory: UInt64
    /// Compressed memory pages in bytes.
    public let compressedMemory: UInt64
    /// Timestamp when the snapshot was captured.
    public let timestamp: Date

    /// Capture the current memory state.
    ///
    /// Uses `task_info` with `TASK_VM_INFO` to get detailed memory statistics.
    ///
    /// - Returns: A snapshot of the current memory state.
    public static func capture() -> MemorySnapshot {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ptr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ptr, &count)
            }
        }

        if result == KERN_SUCCESS {
            return MemorySnapshot(
                physicalFootprint: info.phys_footprint,
                internalMemory: info.internal,
                externalMemory: info.external,
                compressedMemory: info.compressed,
                timestamp: Date()
            )
        }

        // Fallback if task_info fails
        return MemorySnapshot(
            physicalFootprint: 0,
            internalMemory: 0,
            externalMemory: 0,
            compressedMemory: 0,
            timestamp: Date()
        )
    }

    /// Format the physical footprint as a human-readable string.
    public var formattedFootprint: String {
        ByteCountFormatter.string(fromByteCount: Int64(physicalFootprint), countStyle: .memory)
    }
}

/// The difference in memory between two snapshots.
public struct MemoryDelta: Codable, Sendable {
    /// Peak memory footprint observed during tracking.
    public let peakFootprint: UInt64
    /// Net change in allocated bytes (can be negative if memory was freed).
    public let allocatedBytes: Int64
    /// Memory snapshot before the operation.
    public let before: MemorySnapshot
    /// Memory snapshot after the operation.
    public let after: MemorySnapshot

    /// Format the peak footprint as a human-readable string.
    public var formattedPeak: String {
        ByteCountFormatter.string(fromByteCount: Int64(peakFootprint), countStyle: .memory)
    }

    /// Format the allocated bytes as a human-readable string.
    public var formattedAllocated: String {
        let prefix = allocatedBytes >= 0 ? "+" : ""
        return prefix + ByteCountFormatter.string(fromByteCount: allocatedBytes, countStyle: .memory)
    }
}

/// Actor for tracking memory usage during operations.
///
/// Use this to measure peak memory and memory deltas during benchmark runs.
///
/// ## Example Usage
///
/// ```swift
/// let tracker = MemoryTracker()
/// await tracker.begin()
///
/// // Perform memory-intensive operation
/// let result = await someExpensiveOperation()
///
/// // Optionally sample during long operations
/// await tracker.sample()
///
/// let delta = await tracker.end()
/// print("Peak memory: \(delta.formattedPeak)")
/// ```
public actor MemoryTracker {
    private var baseline: MemorySnapshot?
    private var peakFootprint: UInt64 = 0
    private var samples: [MemorySnapshot] = []

    public init() {}

    /// Begin tracking memory usage.
    ///
    /// Captures the baseline memory snapshot and resets peak tracking.
    public func begin() {
        let snapshot = MemorySnapshot.capture()
        baseline = snapshot
        peakFootprint = snapshot.physicalFootprint
        samples = []
    }

    /// Sample the current memory state.
    ///
    /// Call this periodically during long operations to track peak usage.
    public func sample() {
        let current = MemorySnapshot.capture()
        samples.append(current)
        peakFootprint = max(peakFootprint, current.physicalFootprint)
    }

    /// End tracking and return the memory delta.
    ///
    /// - Returns: The difference between the baseline and current memory state,
    ///            including the peak memory observed during tracking.
    public func end() -> MemoryDelta {
        let after = MemorySnapshot.capture()
        peakFootprint = max(peakFootprint, after.physicalFootprint)

        guard let before = baseline else {
            return MemoryDelta(
                peakFootprint: peakFootprint,
                allocatedBytes: 0,
                before: after,
                after: after
            )
        }

        return MemoryDelta(
            peakFootprint: peakFootprint,
            allocatedBytes: Int64(after.physicalFootprint) - Int64(before.physicalFootprint),
            before: before,
            after: after
        )
    }

    /// Get all samples collected during tracking.
    public func getSamples() -> [MemorySnapshot] {
        samples
    }

    /// Reset the tracker to its initial state.
    public func reset() {
        baseline = nil
        peakFootprint = 0
        samples = []
    }
}

/// Memory pressure levels for testing behavior under constrained memory.
public enum MemoryPressureLevel: Sendable {
    /// Warning level (~50% system memory used).
    case warning
    /// Critical level (~80% system memory used).
    case critical
}
