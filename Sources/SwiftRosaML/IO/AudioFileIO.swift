import SwiftRosaCore
import SwiftRosaEffects
import AVFoundation
import Accelerate
import Foundation

/// Errors that can occur during audio file operations.
public enum AudioIOError: Error, Sendable {
    case fileNotFound(URL)
    case invalidFormat(String)
    case readFailed(String)
    case writeFailed(String)
    case conversionFailed(String)
    case unsupportedFormat(String)
}

/// Supported audio file formats for writing.
public enum AudioFileFormat: Sendable {
    case wav
    case aiff
    case m4a(bitRate: Int = 128000)
    case caf
}

/// Audio file reading and writing utilities.
///
/// Provides simple APIs for reading audio files into Float arrays and writing
/// Float arrays to various audio formats. Supports automatic sample rate conversion
/// and channel mixing.
///
/// ## Example Usage
///
/// ```swift
/// // Read audio file
/// let (samples, sampleRate) = try await AudioFileIO.read(url: audioURL)
///
/// // Read and resample to 16kHz
/// let (samples, _) = try await AudioFileIO.read(url: audioURL, targetSampleRate: 16000)
///
/// // Write audio file
/// try await AudioFileIO.write(samples: samples, sampleRate: 22050, to: outputURL)
/// ```
public struct AudioFileIO: Sendable {
    private init() {}

    // MARK: - Reading

    /// Read audio file to mono Float array.
    ///
    /// Automatically mixes multi-channel audio to mono.
    ///
    /// - Parameters:
    ///   - url: URL of the audio file to read.
    ///   - targetSampleRate: Optional target sample rate for resampling.
    /// - Returns: Tuple of audio samples and sample rate.
    /// - Throws: `AudioIOError` if reading fails.
    public static func read(
        url: URL,
        targetSampleRate: Float? = nil
    ) async throws -> (samples: [Float], sampleRate: Float) {
        let (channels, sampleRate) = try await readMultiChannel(url: url, targetSampleRate: targetSampleRate)

        // Mix to mono if multi-channel
        if channels.count > 1 {
            let mono = FormatConverter.mixToMono(channels)
            return (mono, sampleRate)
        } else if let first = channels.first {
            return (first, sampleRate)
        } else {
            return ([], sampleRate)
        }
    }

    /// Read audio file preserving all channels.
    ///
    /// - Parameters:
    ///   - url: URL of the audio file to read.
    ///   - targetSampleRate: Optional target sample rate for resampling.
    /// - Returns: Tuple of channel arrays and sample rate.
    /// - Throws: `AudioIOError` if reading fails.
    public static func readMultiChannel(
        url: URL,
        targetSampleRate: Float? = nil
    ) async throws -> (samples: [[Float]], sampleRate: Float) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioIOError.fileNotFound(url)
        }

        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: url)
        } catch {
            throw AudioIOError.readFailed("Failed to open audio file: \(error.localizedDescription)")
        }

        let fileFormat = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        let channelCount = Int(fileFormat.channelCount)
        let fileSampleRate = Float(fileFormat.sampleRate)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: frameCount) else {
            throw AudioIOError.readFailed("Failed to create audio buffer")
        }

        do {
            try audioFile.read(into: buffer)
        } catch {
            throw AudioIOError.readFailed("Failed to read audio data: \(error.localizedDescription)")
        }

        // Extract channel data
        var channels = [[Float]]()
        channels.reserveCapacity(channelCount)

        guard let floatChannelData = buffer.floatChannelData else {
            throw AudioIOError.readFailed("Failed to access float channel data")
        }

        for channel in 0..<channelCount {
            let channelData = floatChannelData[channel]
            let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(buffer.frameLength)))
            channels.append(samples)
        }

        // Resample if needed
        if let target = targetSampleRate, target != fileSampleRate {
            var resampledChannels = [[Float]]()
            resampledChannels.reserveCapacity(channelCount)

            for channel in channels {
                let resampled = await Resample.resample(
                    channel,
                    fromRate: fileSampleRate,
                    toRate: target,
                    quality: .high
                )
                resampledChannels.append(resampled)
            }

            return (resampledChannels, target)
        }

        return (channels, fileSampleRate)
    }

    // MARK: - Writing

    /// Write mono audio to file.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to write.
    ///   - sampleRate: Sample rate of the audio.
    ///   - url: Destination URL.
    ///   - format: Output format (default: WAV).
    /// - Throws: `AudioIOError` if writing fails.
    public static func write(
        samples: [Float],
        sampleRate: Float,
        to url: URL,
        format: AudioFileFormat = .wav
    ) async throws {
        try await writeMultiChannel(
            samples: [samples],
            sampleRate: sampleRate,
            to: url,
            format: format
        )
    }

    /// Write multi-channel audio to file.
    ///
    /// - Parameters:
    ///   - samples: Array of channel data (each inner array is one channel).
    ///   - sampleRate: Sample rate of the audio.
    ///   - url: Destination URL.
    ///   - format: Output format (default: WAV).
    /// - Throws: `AudioIOError` if writing fails.
    public static func writeMultiChannel(
        samples: [[Float]],
        sampleRate: Float,
        to url: URL,
        format: AudioFileFormat = .wav
    ) async throws {
        guard !samples.isEmpty, let firstChannel = samples.first, !firstChannel.isEmpty else {
            throw AudioIOError.writeFailed("No audio data to write")
        }

        let channelCount = samples.count
        let frameCount = firstChannel.count

        // Verify all channels have same length
        for channel in samples {
            guard channel.count == frameCount else {
                throw AudioIOError.writeFailed("Channel lengths must match")
            }
        }

        // Create audio format
        let (audioFormat, settings) = try createAudioFormat(
            sampleRate: sampleRate,
            channelCount: channelCount,
            format: format
        )

        // Create buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
            throw AudioIOError.writeFailed("Failed to create audio buffer")
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)

        // Copy data to buffer
        guard let floatChannelData = buffer.floatChannelData else {
            throw AudioIOError.writeFailed("Failed to access buffer channel data")
        }

        for (channelIndex, channelSamples) in samples.enumerated() {
            let channelPtr = floatChannelData[channelIndex]
            for (sampleIndex, sample) in channelSamples.enumerated() {
                channelPtr[sampleIndex] = sample
            }
        }

        // Write to file
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(
                forWriting: url,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
        } catch {
            throw AudioIOError.writeFailed("Failed to create output file: \(error.localizedDescription)")
        }

        do {
            try audioFile.write(from: buffer)
        } catch {
            throw AudioIOError.writeFailed("Failed to write audio data: \(error.localizedDescription)")
        }
    }

    // MARK: - Format Helpers

    private static func createAudioFormat(
        sampleRate: Float,
        channelCount: Int,
        format: AudioFileFormat
    ) throws -> (AVAudioFormat, [String: Any]) {
        var settings: [String: Any] = [
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: channelCount
        ]

        switch format {
        case .wav:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 16
            settings[AVLinearPCMIsFloatKey] = false
            settings[AVLinearPCMIsBigEndianKey] = false

        case .aiff:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 16
            settings[AVLinearPCMIsFloatKey] = false
            settings[AVLinearPCMIsBigEndianKey] = true

        case .m4a(let bitRate):
            settings[AVFormatIDKey] = kAudioFormatMPEG4AAC
            settings[AVEncoderBitRateKey] = bitRate

        case .caf:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 32
            settings[AVLinearPCMIsFloatKey] = true
            settings[AVLinearPCMIsBigEndianKey] = false
        }

        guard let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channelCount),
            interleaved: false
        ) else {
            throw AudioIOError.invalidFormat("Failed to create audio format")
        }

        return (audioFormat, settings)
    }

    // MARK: - Utilities

    /// Get audio file information without reading full data.
    ///
    /// - Parameter url: URL of the audio file.
    /// - Returns: Audio file metadata.
    /// - Throws: `AudioIOError` if file cannot be opened.
    public static func getInfo(url: URL) throws -> AudioFileInfo {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioIOError.fileNotFound(url)
        }

        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: url)
        } catch {
            throw AudioIOError.readFailed("Failed to open audio file: \(error.localizedDescription)")
        }

        let format = audioFile.processingFormat
        let duration = Double(audioFile.length) / format.sampleRate

        return AudioFileInfo(
            sampleRate: Float(format.sampleRate),
            channelCount: Int(format.channelCount),
            frameCount: Int(audioFile.length),
            duration: Float(duration)
        )
    }
}

/// Information about an audio file.
public struct AudioFileInfo: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Number of audio channels.
    public let channelCount: Int

    /// Total number of sample frames.
    public let frameCount: Int

    /// Duration in seconds.
    public let duration: Float
}
