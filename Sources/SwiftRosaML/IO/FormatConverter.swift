import SwiftRosaCore
import Accelerate
import Foundation

/// PCM audio formats for conversion.
public enum PCMFormat: Sendable {
    case float32
    case int16
    case int24
    case int32
}

/// Audio format conversion utilities.
///
/// Provides methods for converting between PCM formats and mixing audio channels.
///
/// ## Example Usage
///
/// ```swift
/// // Mix stereo to mono
/// let mono = FormatConverter.mixToMono([leftChannel, rightChannel])
///
/// // Convert float samples to 16-bit PCM data
/// let pcmData = FormatConverter.convert(samples, to: .int16)
///
/// // Convert 16-bit PCM data back to float
/// let floatSamples = FormatConverter.toFloat(pcmData, from: .int16)
/// ```
public struct FormatConverter: Sendable {
    private init() {}

    // MARK: - Channel Mixing

    /// Mix multi-channel audio to mono.
    ///
    /// Takes the average of all channels at each sample position.
    ///
    /// - Parameter channels: Array of channel data.
    /// - Returns: Mono signal.
    public static func mixToMono(_ channels: [[Float]]) -> [Float] {
        guard !channels.isEmpty else { return [] }
        guard channels.count > 1 else { return channels[0] }

        let channelCount = channels.count
        let frameCount = channels[0].count

        // Verify all channels have same length
        for channel in channels {
            guard channel.count == frameCount else {
                // Return first channel if lengths don't match
                return channels[0]
            }
        }

        var mono = [Float](repeating: 0, count: frameCount)

        // Sum all channels
        for channel in channels {
            vDSP_vadd(mono, 1, channel, 1, &mono, 1, vDSP_Length(frameCount))
        }

        // Divide by channel count
        var scale = 1.0 / Float(channelCount)
        vDSP_vsmul(mono, 1, &scale, &mono, 1, vDSP_Length(frameCount))

        return mono
    }

    /// Mix stereo to mono with custom left/right balance.
    ///
    /// - Parameters:
    ///   - left: Left channel samples.
    ///   - right: Right channel samples.
    ///   - leftWeight: Weight for left channel (default: 0.5).
    ///   - rightWeight: Weight for right channel (default: 0.5).
    /// - Returns: Mono signal.
    public static func mixStereoToMono(
        left: [Float],
        right: [Float],
        leftWeight: Float = 0.5,
        rightWeight: Float = 0.5
    ) -> [Float] {
        let frameCount = min(left.count, right.count)
        guard frameCount > 0 else { return [] }

        var mono = [Float](repeating: 0, count: frameCount)

        // left * leftWeight
        var lw = leftWeight
        var leftScaled = [Float](repeating: 0, count: frameCount)
        vDSP_vsmul(left, 1, &lw, &leftScaled, 1, vDSP_Length(frameCount))

        // right * rightWeight
        var rw = rightWeight
        var rightScaled = [Float](repeating: 0, count: frameCount)
        vDSP_vsmul(right, 1, &rw, &rightScaled, 1, vDSP_Length(frameCount))

        // Sum
        vDSP_vadd(leftScaled, 1, rightScaled, 1, &mono, 1, vDSP_Length(frameCount))

        return mono
    }

    /// Split mono to stereo (duplicate to both channels).
    ///
    /// - Parameter mono: Mono signal.
    /// - Returns: Tuple of (left, right) channels.
    public static func monoToStereo(_ mono: [Float]) -> (left: [Float], right: [Float]) {
        return (mono, mono)
    }

    // MARK: - Format Conversion

    /// Convert Float samples to PCM data.
    ///
    /// - Parameters:
    ///   - samples: Float audio samples (expected range: -1.0 to 1.0).
    ///   - format: Target PCM format.
    /// - Returns: Raw PCM data.
    public static func convert(_ samples: [Float], to format: PCMFormat) -> Data {
        switch format {
        case .float32:
            return samples.withUnsafeBytes { Data($0) }

        case .int16:
            return convertToInt16(samples)

        case .int24:
            return convertToInt24(samples)

        case .int32:
            return convertToInt32(samples)
        }
    }

    /// Convert PCM data to Float samples.
    ///
    /// - Parameters:
    ///   - data: Raw PCM data.
    ///   - format: Source PCM format.
    /// - Returns: Float audio samples (range: -1.0 to 1.0).
    public static func toFloat(_ data: Data, from format: PCMFormat) -> [Float] {
        switch format {
        case .float32:
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }

        case .int16:
            return convertFromInt16(data)

        case .int24:
            return convertFromInt24(data)

        case .int32:
            return convertFromInt32(data)
        }
    }

    // MARK: - Int16 Conversion

    private static func convertToInt16(_ samples: [Float]) -> Data {
        let count = samples.count

        // Clamp samples to [-1, 1]
        var clampedSamples = [Float](repeating: 0, count: count)
        var minVal: Float = -1.0
        var maxVal: Float = 1.0
        vDSP_vclip(samples, 1, &minVal, &maxVal, &clampedSamples, 1, vDSP_Length(count))

        // Scale to Int16 range
        var scale = Float(Int16.max)
        var scaledSamples = [Float](repeating: 0, count: count)
        vDSP_vsmul(clampedSamples, 1, &scale, &scaledSamples, 1, vDSP_Length(count))

        // Convert to Int16
        var int16Samples = [Int16](repeating: 0, count: count)
        vDSP_vfix16(scaledSamples, 1, &int16Samples, 1, vDSP_Length(count))

        return int16Samples.withUnsafeBytes { Data($0) }
    }

    private static func convertFromInt16(_ data: Data) -> [Float] {
        let int16Samples = data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Int16.self))
        }

        let count = int16Samples.count
        var floatSamples = [Float](repeating: 0, count: count)

        // Convert Int16 to Float
        vDSP_vflt16(int16Samples, 1, &floatSamples, 1, vDSP_Length(count))

        // Scale to [-1, 1] range
        var scale = 1.0 / Float(Int16.max)
        vDSP_vsmul(floatSamples, 1, &scale, &floatSamples, 1, vDSP_Length(count))

        return floatSamples
    }

    // MARK: - Int24 Conversion

    private static func convertToInt24(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 3)

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int24Value = Int32(clamped * Float(0x7FFFFF))

            // Little-endian: write low, mid, high bytes
            data.append(UInt8(truncatingIfNeeded: int24Value & 0xFF))
            data.append(UInt8(truncatingIfNeeded: (int24Value >> 8) & 0xFF))
            data.append(UInt8(truncatingIfNeeded: (int24Value >> 16) & 0xFF))
        }

        return data
    }

    private static func convertFromInt24(_ data: Data) -> [Float] {
        let sampleCount = data.count / 3
        var floatSamples = [Float](repeating: 0, count: sampleCount)

        for i in 0..<sampleCount {
            let offset = i * 3
            let low = Int32(data[offset])
            let mid = Int32(data[offset + 1])
            let high = Int32(data[offset + 2])

            // Reconstruct 24-bit value (sign extend from 24 to 32 bits)
            var int24Value = low | (mid << 8) | (high << 16)
            if (int24Value & 0x800000) != 0 {
                int24Value |= Int32(bitPattern: 0xFF000000)  // Sign extend
            }

            floatSamples[i] = Float(int24Value) / Float(0x7FFFFF)
        }

        return floatSamples
    }

    // MARK: - Int32 Conversion

    private static func convertToInt32(_ samples: [Float]) -> Data {
        let count = samples.count

        // Clamp samples to [-1, 1]
        var clampedSamples = [Float](repeating: 0, count: count)
        var minVal: Float = -1.0
        var maxVal: Float = 1.0
        vDSP_vclip(samples, 1, &minVal, &maxVal, &clampedSamples, 1, vDSP_Length(count))

        // Scale to Int32 range
        var scale = Float(Int32.max)
        var scaledSamples = [Float](repeating: 0, count: count)
        vDSP_vsmul(clampedSamples, 1, &scale, &scaledSamples, 1, vDSP_Length(count))

        // Convert to Int32
        var int32Samples = [Int32](repeating: 0, count: count)
        vDSP_vfix32(scaledSamples, 1, &int32Samples, 1, vDSP_Length(count))

        return int32Samples.withUnsafeBytes { Data($0) }
    }

    private static func convertFromInt32(_ data: Data) -> [Float] {
        let int32Samples = data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Int32.self))
        }

        let count = int32Samples.count
        var floatSamples = [Float](repeating: 0, count: count)

        // Convert Int32 to Float
        vDSP_vflt32(int32Samples, 1, &floatSamples, 1, vDSP_Length(count))

        // Scale to [-1, 1] range
        var scale = 1.0 / Float(Int32.max)
        vDSP_vsmul(floatSamples, 1, &scale, &floatSamples, 1, vDSP_Length(count))

        return floatSamples
    }

    // MARK: - Interleaving

    /// Interleave multi-channel audio.
    ///
    /// Uses vDSP strided operations for optimal performance.
    ///
    /// - Parameter channels: Array of channel data.
    /// - Returns: Interleaved samples [L0, R0, L1, R1, ...].
    public static func interleave(_ channels: [[Float]]) -> [Float] {
        guard !channels.isEmpty else { return [] }
        guard channels.count > 1 else { return channels[0] }

        let channelCount = channels.count
        let frameCount = channels[0].count
        var interleaved = [Float](repeating: 0, count: channelCount * frameCount)

        // Use vDSP strided copy for each channel
        // Copy channel[c] to every channelCount-th position starting at offset c
        for (c, channel) in channels.enumerated() {
            channel.withUnsafeBufferPointer { srcPtr in
                interleaved.withUnsafeMutableBufferPointer { dstPtr in
                    // vDSP_mmov copies with strides
                    // Source stride: 1 (contiguous channel data)
                    // Dest stride: channelCount (interleaved positions)
                    cblas_scopy(
                        Int32(frameCount),
                        srcPtr.baseAddress!,
                        1,                                    // Source stride
                        dstPtr.baseAddress! + c,
                        Int32(channelCount)                   // Dest stride
                    )
                }
            }
        }

        return interleaved
    }

    /// Deinterleave audio to separate channels.
    ///
    /// Uses vDSP strided operations for optimal performance.
    ///
    /// - Parameters:
    ///   - samples: Interleaved samples.
    ///   - channelCount: Number of channels.
    /// - Returns: Array of channel data.
    public static func deinterleave(_ samples: [Float], channelCount: Int) -> [[Float]] {
        guard channelCount > 0 else { return [] }
        guard !samples.isEmpty else { return Array(repeating: [], count: channelCount) }

        let frameCount = samples.count / channelCount
        var channels = [[Float]]()
        channels.reserveCapacity(channelCount)

        // Use cblas_scopy with strides for each channel
        for c in 0..<channelCount {
            var channelData = [Float](repeating: 0, count: frameCount)

            samples.withUnsafeBufferPointer { srcPtr in
                channelData.withUnsafeMutableBufferPointer { dstPtr in
                    // Source stride: channelCount (skip other channels)
                    // Dest stride: 1 (contiguous output)
                    cblas_scopy(
                        Int32(frameCount),
                        srcPtr.baseAddress! + c,
                        Int32(channelCount),                   // Source stride
                        dstPtr.baseAddress!,
                        1                                       // Dest stride
                    )
                }
            }

            channels.append(channelData)
        }

        return channels
    }

    // MARK: - Normalization

    /// Normalize audio to peak amplitude of 1.0.
    ///
    /// - Parameter samples: Audio samples.
    /// - Returns: Normalized samples.
    public static func normalize(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }

        var maxVal: Float = 0
        vDSP_maxmgv(samples, 1, &maxVal, vDSP_Length(samples.count))

        guard maxVal > 0 else { return samples }

        var scale = 1.0 / maxVal
        var normalized = [Float](repeating: 0, count: samples.count)
        vDSP_vsmul(samples, 1, &scale, &normalized, 1, vDSP_Length(samples.count))

        return normalized
    }

    /// Normalize audio to target RMS level.
    ///
    /// - Parameters:
    ///   - samples: Audio samples.
    ///   - targetRMS: Target RMS level (default: 0.1).
    /// - Returns: Normalized samples.
    public static func normalizeRMS(_ samples: [Float], targetRMS: Float = 0.1) -> [Float] {
        guard !samples.isEmpty else { return [] }

        // Compute current RMS
        var sumSquares: Float = 0
        vDSP_svesq(samples, 1, &sumSquares, vDSP_Length(samples.count))
        let rms = sqrt(sumSquares / Float(samples.count))

        guard rms > 0 else { return samples }

        var scale = targetRMS / rms
        var normalized = [Float](repeating: 0, count: samples.count)
        vDSP_vsmul(samples, 1, &scale, &normalized, 1, vDSP_Length(samples.count))

        return normalized
    }
}
