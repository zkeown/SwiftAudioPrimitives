import SwiftRosaCore
import AVFoundation
import Foundation

/// Errors that can occur during audio capture.
public enum AudioCaptureError: Error, Sendable {
    case engineStartFailed(String)
    case formatError(String)
    case notRecording
    case alreadyRecording
    case audioSessionError(String)
}

/// Real-time audio capture using AVAudioEngine.
///
/// Captures audio from the default input device (microphone) and delivers
/// samples via callback or accumulation.
///
/// ## Example Usage
///
/// ```swift
/// let capture = AudioEngineCapture(sampleRate: 16000, bufferSize: 1024)
///
/// // Start capture with callback
/// try await capture.start { samples in
///     // Process each buffer of samples
///     print("Received \(samples.count) samples")
/// }
///
/// // ... later ...
/// let allSamples = await capture.stop()
/// ```
///
/// - Note: On iOS, this requires microphone permissions. Ensure your app's
///   Info.plist includes `NSMicrophoneUsageDescription`.
public actor AudioEngineCapture {
    /// Current capture state.
    public enum State: Sendable {
        case idle
        case recording
        case paused
    }

    /// Current state of the capture.
    public private(set) var state: State = .idle

    /// Sample rate for capture.
    public let sampleRate: Double

    /// Buffer size in samples.
    public let bufferSize: Int

    private var engine: AVAudioEngine?
    private var accumulatedSamples: [Float] = []
    private var userCallback: (@Sendable ([Float]) -> Void)?

    /// Statistics about the capture session.
    public private(set) var stats: CaptureStats = CaptureStats()

    /// Create an audio capture instance.
    ///
    /// - Parameters:
    ///   - sampleRate: Desired sample rate in Hz (default: 44100).
    ///   - bufferSize: Samples per callback buffer (default: 1024).
    public init(sampleRate: Double = 44100, bufferSize: Int = 1024) {
        self.sampleRate = sampleRate
        self.bufferSize = bufferSize
    }

    /// Start audio capture.
    ///
    /// - Parameter onBuffer: Optional callback for each buffer of samples.
    ///   If nil, samples are accumulated and returned on stop.
    /// - Throws: `AudioCaptureError` if capture cannot start.
    public func start(
        onBuffer: (@Sendable ([Float]) -> Void)? = nil
    ) throws {
        guard state == .idle else {
            throw AudioCaptureError.alreadyRecording
        }

        self.userCallback = onBuffer
        self.accumulatedSamples = []
        self.stats = CaptureStats()

        let engine = AVAudioEngine()
        self.engine = engine

        let inputNode = engine.inputNode

        // Get the input format
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard inputFormat.sampleRate > 0 else {
            throw AudioCaptureError.formatError("Invalid input format")
        }

        // Create target format if resampling needed
        let targetFormat: AVAudioFormat
        if inputFormat.sampleRate != sampleRate {
            guard let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            ) else {
                throw AudioCaptureError.formatError("Failed to create target format")
            }
            targetFormat = format
        } else {
            guard let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputFormat.sampleRate,
                channels: 1,
                interleaved: false
            ) else {
                throw AudioCaptureError.formatError("Failed to create format")
            }
            targetFormat = format
        }

        // Install tap on input node
        inputNode.installTap(
            onBus: 0,
            bufferSize: AVAudioFrameCount(bufferSize),
            format: inputFormat
        ) { [weak self] buffer, _ in
            Task { [weak self] in
                await self?.processBuffer(buffer, targetFormat: targetFormat)
            }
        }

        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            self.engine = nil
            throw AudioCaptureError.engineStartFailed(error.localizedDescription)
        }

        state = .recording
        stats.startTime = Date()
    }

    /// Process incoming audio buffer.
    private func processBuffer(_ buffer: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        guard state == .recording else { return }

        // Convert to mono float if needed
        let samples = extractMonoSamples(from: buffer)

        // Accumulate samples
        accumulatedSamples.append(contentsOf: samples)
        stats.totalSamplesReceived += samples.count
        stats.buffersReceived += 1

        // Call user callback
        if let callback = userCallback {
            callback(samples)
        }
    }

    /// Extract mono float samples from audio buffer.
    private func extractMonoSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let floatChannelData = buffer.floatChannelData else { return [] }

        let frameLength = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)

        if channelCount == 1 {
            // Already mono
            return Array(UnsafeBufferPointer(start: floatChannelData[0], count: frameLength))
        } else {
            // Mix to mono
            var mono = [Float](repeating: 0, count: frameLength)
            for channel in 0..<channelCount {
                let channelPtr = floatChannelData[channel]
                for i in 0..<frameLength {
                    mono[i] += channelPtr[i]
                }
            }

            // Average
            let scale = 1.0 / Float(channelCount)
            for i in 0..<frameLength {
                mono[i] *= scale
            }

            return mono
        }
    }

    /// Pause audio capture.
    ///
    /// Capture can be resumed with `resume()`.
    public func pause() {
        guard state == .recording else { return }
        engine?.pause()
        state = .paused
    }

    /// Resume paused audio capture.
    ///
    /// - Throws: `AudioCaptureError` if not paused.
    public func resume() throws {
        guard state == .paused else {
            throw AudioCaptureError.notRecording
        }

        do {
            try engine?.start()
            state = .recording
        } catch {
            throw AudioCaptureError.engineStartFailed(error.localizedDescription)
        }
    }

    /// Stop audio capture and return all accumulated samples.
    ///
    /// - Returns: All samples captured during the session.
    public func stop() -> [Float] {
        guard state != .idle else { return [] }

        // Remove tap and stop engine
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil

        state = .idle
        stats.endTime = Date()

        let samples = accumulatedSamples
        accumulatedSamples = []
        userCallback = nil

        return samples
    }

    /// Get current accumulated samples without stopping.
    ///
    /// - Returns: Copy of currently accumulated samples.
    public func getCurrentSamples() -> [Float] {
        return accumulatedSamples
    }

    /// Clear accumulated samples without stopping capture.
    public func clearAccumulatedSamples() {
        accumulatedSamples = []
    }

    /// Check if audio input is available.
    ///
    /// - Returns: True if audio input is available.
    public static func isInputAvailable() -> Bool {
        let engine = AVAudioEngine()
        let inputFormat = engine.inputNode.outputFormat(forBus: 0)
        return inputFormat.sampleRate > 0
    }
}

/// Statistics about an audio capture session.
public struct CaptureStats: Sendable {
    /// Total samples received.
    public var totalSamplesReceived: Int = 0

    /// Number of buffers received.
    public var buffersReceived: Int = 0

    /// Capture start time.
    public var startTime: Date?

    /// Capture end time.
    public var endTime: Date?

    /// Duration of capture in seconds.
    public var duration: TimeInterval? {
        guard let start = startTime else { return nil }
        let end = endTime ?? Date()
        return end.timeIntervalSince(start)
    }
}

// MARK: - iOS Audio Session Configuration

#if os(iOS)
import AVFAudio

extension AudioEngineCapture {
    /// Configure audio session for recording on iOS.
    ///
    /// Call this before starting capture on iOS devices.
    ///
    /// - Throws: `AudioCaptureError` if configuration fails.
    public static func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()

        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            throw AudioCaptureError.audioSessionError(error.localizedDescription)
        }
    }

    /// Request microphone permission on iOS.
    ///
    /// - Returns: True if permission was granted.
    public static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}
#endif
