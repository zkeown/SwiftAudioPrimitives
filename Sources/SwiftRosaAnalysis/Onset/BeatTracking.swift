import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for beat tracking.
public struct BeatTrackerConfig: Sendable {
    /// Sample rate in Hz.
    public let sampleRate: Float

    /// Hop length between frames.
    public let hopLength: Int

    /// Starting BPM estimate.
    public let startBPM: Float

    /// BPM search range.
    public let bpmRange: (min: Float, max: Float)

    /// Tightness of tempo constraint (higher = stricter).
    public let tightness: Float

    /// Create beat tracker configuration.
    ///
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (default: 22050).
    ///   - hopLength: Hop length (default: 512).
    ///   - startBPM: Initial BPM estimate (default: 120).
    ///   - bpmRange: BPM search range (default: 60-240).
    ///   - tightness: Tempo tightness (default: 100).
    public init(
        sampleRate: Float = 22050,
        hopLength: Int = 512,
        startBPM: Float = 120.0,
        bpmRange: (min: Float, max: Float) = (60.0, 240.0),
        tightness: Float = 100.0
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.startBPM = startBPM
        self.bpmRange = bpmRange
        self.tightness = tightness
    }

    /// Frames per second.
    public var framesPerSecond: Float {
        sampleRate / Float(hopLength)
    }

    /// Convert BPM to frames per beat.
    public func bpmToFrames(_ bpm: Float) -> Float {
        60.0 * framesPerSecond / bpm
    }

    /// Convert frames per beat to BPM.
    public func framesToBPM(_ frames: Float) -> Float {
        60.0 * framesPerSecond / frames
    }
}

/// Simple beat tracker.
///
/// Estimates tempo and tracks beat positions in audio signals.
/// Uses onset detection and autocorrelation-based tempo estimation.
///
/// ## Example Usage
///
/// ```swift
/// let tracker = BeatTracker()
///
/// // Estimate tempo
/// let bpm = await tracker.estimateTempo(audioSignal)
///
/// // Track beats
/// let beatTimes = await tracker.track(audioSignal)
/// ```
public struct BeatTracker: Sendable {
    /// Configuration.
    public let config: BeatTrackerConfig

    private let onsetDetector: OnsetDetector

    /// Create a beat tracker.
    ///
    /// - Parameter config: Configuration.
    public init(config: BeatTrackerConfig = BeatTrackerConfig()) {
        self.config = config
        // Use librosa-compatible onset strength for best accuracy
        // This uses log-compressed mel spectrogram which produces normalized values
        // suitable for the CV-based rhythm detection checks in estimateTempo
        self.onsetDetector = OnsetDetector(config: OnsetConfig(
            sampleRate: config.sampleRate,
            hopLength: config.hopLength,
            nFFT: 2048,
            method: .librosa,
            peakThreshold: 0.3,
            preMax: 3,
            postMax: 3,
            preAvg: 3,
            postAvg: 3,
            delta: 0.05,
            wait: 10
        ))
    }

    /// Estimate tempo in BPM.
    ///
    /// Uses librosa-compatible tempogram algorithm:
    /// 1. Compute onset strength envelope
    /// 2. Compute local autocorrelation (tempogram) for each frame
    /// 3. Aggregate tempogram across time with mean
    /// 4. Apply log-normal prior and select best tempo
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Estimated tempo in beats per minute, or 0 if no clear rhythm detected.
    public func estimateTempo(_ signal: [Float]) async -> Float {
        // Compute onset strength
        let onsetEnvelope = await onsetDetector.onsetStrength(signal)
        guard onsetEnvelope.count > 10 else { return 0 }

        // Check if onset envelope has sufficient energy
        var maxOnset: Float = 0
        vDSP_maxv(onsetEnvelope, 1, &maxOnset, vDSP_Length(onsetEnvelope.count))
        let minOnsetStrength: Float = 0.1
        if maxOnset < minOnsetStrength {
            return 0
        }

        // Check for periodic content - coefficient of variation
        var mean: Float = 0
        vDSP_meanv(onsetEnvelope, 1, &mean, vDSP_Length(onsetEnvelope.count))
        var variance: Float = 0
        for val in onsetEnvelope {
            let diff = val - mean
            variance += diff * diff
        }
        variance /= Float(onsetEnvelope.count)
        let cv = mean > 0.001 ? sqrt(variance) / mean : 0
        if cv < 0.25 {
            return 0
        }

        // Tempo range in lags
        let minLag = max(1, Int(config.bpmToFrames(config.bpmRange.max)))
        let maxLag = Int(config.bpmToFrames(config.bpmRange.min))

        // Compute tempogram (per-frame local autocorrelation) matching librosa
        // librosa uses win_length=384 by default
        let winLength = 384
        let tempogram = computeTempogram(onsetEnvelope, windowLength: winLength, maxLag: maxLag)

        guard !tempogram.isEmpty else { return 0 }

        // Aggregate tempogram with mean (librosa default)
        var aggregatedTG = [Float](repeating: 0, count: maxLag)
        let nFrames = tempogram.count
        for lag in 0..<maxLag {
            var sum: Float = 0
            for frame in 0..<nFrames {
                if lag < tempogram[frame].count {
                    sum += tempogram[frame][lag]
                }
            }
            aggregatedTG[lag] = sum / Float(nFrames)
        }

        // Compute BPM values for each lag
        var bpms = [Float](repeating: 0, count: maxLag)
        for lag in 1..<maxLag {
            bpms[lag] = config.framesToBPM(Float(lag))
        }

        // Apply log-normal tempo prior (exact librosa formula)
        // logprior = -0.5 * ((log2(bpm) - log2(start_bpm)) / std_bpm)^2
        let logStartBPM = log2(config.startBPM)
        let stdBPM: Float = 1.0  // librosa default

        var logPrior = [Float](repeating: -Float.infinity, count: maxLag)
        for lag in minLag..<maxLag {
            let logBPM = log2(bpms[lag])
            let diff = (logBPM - logStartBPM) / stdBPM
            logPrior[lag] = -0.5 * diff * diff
        }

        // Find best tempo: argmax(log1p(1e6 * tg) + logprior)
        var bestLag = 0
        var bestScore: Float = -Float.infinity

        for lag in minLag..<maxLag {
            let logTG = log1p(1e6 * max(0, aggregatedTG[lag]))
            let score = logTG + logPrior[lag]
            if score > bestScore {
                bestScore = score
                bestLag = lag
            }
        }

        // Reject weak estimates where autocorrelation peak is below noise threshold
        if bestLag == 0 || aggregatedTG[bestLag] < 0.01 {
            return 0
        }

        return bpms[bestLag]
    }

    /// Compute tempogram (local autocorrelation at each frame)
    /// Matches librosa.feature.tempogram with center=True (default)
    private func computeTempogram(_ onsetEnvelope: [Float], windowLength: Int, maxLag: Int) -> [[Float]] {
        let n = onsetEnvelope.count
        guard n > 0 else { return [] }

        // center=True: pad onset envelope with win_length//2 on each side
        let padLength = windowLength / 2
        var paddedEnvelope = [Float](repeating: 0, count: n + 2 * padLength)
        // Edge padding (reflect mode) - use first/last values
        for i in 0..<padLength {
            paddedEnvelope[i] = onsetEnvelope[0]
        }
        for i in 0..<n {
            paddedEnvelope[padLength + i] = onsetEnvelope[i]
        }
        for i in 0..<padLength {
            paddedEnvelope[padLength + n + i] = onsetEnvelope[n - 1]
        }

        // Pre-compute Hann window with fftbins=True (asymmetric, periodic)
        // librosa uses scipy.signal.get_window(window, win_length, fftbins=True)
        var hannWindow = [Float](repeating: 0, count: windowLength)
        for i in 0..<windowLength {
            // Periodic Hann (fftbins=True): divide by N instead of N-1
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(windowLength)))
        }

        // Number of output frames = n (original length, after trimming back)
        var tempogram = [[Float]]()
        tempogram.reserveCapacity(n)

        // For each frame, compute local autocorrelation
        // Frame extraction: frame i starts at position i in padded envelope
        for frameIdx in 0..<n {
            let frameStart = frameIdx

            // Extract frame and apply Hann window
            var windowedFrame = [Float](repeating: 0, count: windowLength)
            for i in 0..<windowLength {
                windowedFrame[i] = paddedEnvelope[frameStart + i] * hannWindow[i]
            }

            // Compute autocorrelation for this windowed frame
            let autocorr = computeLocalAutocorrelation(windowedFrame, maxLag: min(maxLag, windowLength))
            tempogram.append(autocorr)
        }

        return tempogram
    }

    /// Compute normalized autocorrelation for a single frame using FFT (matches librosa.autocorrelate)
    private func computeLocalAutocorrelation(_ frame: [Float], maxLag: Int) -> [Float] {
        let n = frame.count
        guard n > 1 else { return [Float](repeating: 0, count: maxLag) }

        // FFT-based autocorrelation (Wiener-Khinchin theorem)
        // Pad to 2*n-1 (or next power of 2 for efficiency)
        let fftLength = 1 << Int(ceil(log2(Double(2 * n - 1))))

        var paddedFrame = [Float](repeating: 0, count: fftLength)
        for i in 0..<n {
            paddedFrame[i] = frame[i]
        }

        // Forward FFT
        var realPart = paddedFrame
        var imagPart = [Float](repeating: 0, count: fftLength)

        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: maxLag)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_fft_zip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
            }
        }

        // Power spectrum: |FFT|^2
        var power = [Float](repeating: 0, count: fftLength)
        for i in 0..<fftLength {
            power[i] = realPart[i] * realPart[i] + imagPart[i] * imagPart[i]
        }

        // Inverse FFT of power spectrum
        realPart = power
        imagPart = [Float](repeating: 0, count: fftLength)
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                vDSP_fft_zip(fftSetup, &split, 1, log2n, FFTDirection(FFT_INVERSE))
            }
        }

        // Scale by FFT length
        var scale = 1.0 / Float(fftLength)
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(fftLength))

        // Extract autocorrelation values up to maxLag
        var autocorr = [Float](repeating: 0, count: maxLag)
        let limitLag = min(maxLag, n)
        for i in 0..<limitLag {
            autocorr[i] = realPart[i]
        }

        // L-infinity normalization (librosa default: divide by max absolute value)
        var maxVal: Float = 0
        vDSP_maxmgv(autocorr, 1, &maxVal, vDSP_Length(maxLag))
        if maxVal > 0 {
            var invMax = 1.0 / maxVal
            vDSP_vsmul(autocorr, 1, &invMax, &autocorr, 1, vDSP_Length(maxLag))
        }

        return autocorr
    }

    /// Track beats in audio.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Array of beat times in seconds.
    public func track(_ signal: [Float]) async -> [Float] {
        // Get onset envelope
        let onsetEnvelope = await onsetDetector.onsetStrength(signal)
        guard !onsetEnvelope.isEmpty else { return [] }

        // Estimate tempo
        let tempo = await estimateTempo(signal)
        let periodFrames = config.bpmToFrames(tempo)

        // Dynamic programming to find optimal beat sequence
        let beats = trackBeats(onsetEnvelope, periodFrames: periodFrames)

        // Convert to times
        return beats.map { Float($0) / config.framesPerSecond }
    }

    /// Get tempo in BPM and beat times together.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Tuple of (tempo in BPM, beat times in seconds).
    public func analyze(_ signal: [Float]) async -> (tempo: Float, beats: [Float]) {
        let tempo = await estimateTempo(signal)
        let beats = await track(signal)
        return (tempo, beats)
    }

    // MARK: - Private Implementation

    private func trackBeats(_ onsetEnvelope: [Float], periodFrames: Float) -> [Int] {
        let n = onsetEnvelope.count
        guard n > 0 else { return [] }

        // Normalize onset envelope
        var maxVal: Float = 0
        vDSP_maxv(onsetEnvelope, 1, &maxVal, vDSP_Length(n))
        var normalizedEnvelope = onsetEnvelope
        if maxVal > 0 {
            var scale = 1.0 / maxVal
            vDSP_vsmul(normalizedEnvelope, 1, &scale, &normalizedEnvelope, 1, vDSP_Length(n))
        }

        // Simple peak-picking with period constraint
        var beats = [Int]()
        let period = Int(periodFrames)
        let tolerance = period / 4

        // Find initial strong beat
        var bestStart = 0
        var bestStartValue: Float = 0
        for i in 0..<min(period * 2, n) {
            if normalizedEnvelope[i] > bestStartValue {
                bestStartValue = normalizedEnvelope[i]
                bestStart = i
            }
        }

        beats.append(bestStart)
        var lastBeat = bestStart

        // Track subsequent beats
        while lastBeat + period < n {
            let searchStart = lastBeat + period - tolerance
            let searchEnd = min(lastBeat + period + tolerance, n)

            // Find peak in search window
            var bestPos = lastBeat + period
            var bestValue: Float = 0

            for i in searchStart..<searchEnd {
                if normalizedEnvelope[i] > bestValue {
                    bestValue = normalizedEnvelope[i]
                    bestPos = i
                }
            }

            // Accept if above threshold or use predicted position
            if bestValue > 0.1 {
                beats.append(bestPos)
                lastBeat = bestPos
            } else {
                // Use predicted position
                let predicted = lastBeat + period
                if predicted < n {
                    beats.append(predicted)
                    lastBeat = predicted
                } else {
                    break
                }
            }
        }

        return beats
    }
}

// MARK: - Presets

extension BeatTracker {
    /// Standard configuration for general music.
    public static var standard: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 512,
            startBPM: 120.0,
            bpmRange: (60.0, 240.0),
            tightness: 100.0
        ))
    }

    /// Configuration for electronic/dance music (higher tempo range).
    public static var electronic: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 256,
            startBPM: 130.0,
            bpmRange: (100.0, 180.0),
            tightness: 150.0
        ))
    }

    /// Configuration for classical/slower music.
    public static var classical: BeatTracker {
        BeatTracker(config: BeatTrackerConfig(
            sampleRate: 22050,
            hopLength: 512,
            startBPM: 80.0,
            bpmRange: (40.0, 160.0),
            tightness: 80.0
        ))
    }
}

// MARK: - Utility Methods

extension BeatTracker {
    /// Compute inter-beat intervals from beat times.
    ///
    /// - Parameter beatTimes: Beat times in seconds.
    /// - Returns: Intervals between consecutive beats.
    public static func interBeatIntervals(_ beatTimes: [Float]) -> [Float] {
        guard beatTimes.count > 1 else { return [] }

        var intervals = [Float]()
        intervals.reserveCapacity(beatTimes.count - 1)

        for i in 1..<beatTimes.count {
            intervals.append(beatTimes[i] - beatTimes[i - 1])
        }

        return intervals
    }

    /// Convert beat times to a click track.
    ///
    /// - Parameters:
    ///   - beatTimes: Beat times in seconds.
    ///   - sampleRate: Output sample rate.
    ///   - duration: Total duration in seconds.
    ///   - clickFrequency: Click tone frequency (default: 1000 Hz).
    ///   - clickDuration: Click duration in seconds (default: 0.02).
    /// - Returns: Click track audio signal.
    public static func generateClickTrack(
        beatTimes: [Float],
        sampleRate: Float,
        duration: Float,
        clickFrequency: Float = 1000.0,
        clickDuration: Float = 0.02
    ) -> [Float] {
        let totalSamples = Int(duration * sampleRate)
        var clickTrack = [Float](repeating: 0, count: totalSamples)

        let clickSamples = Int(clickDuration * sampleRate)

        for beatTime in beatTimes {
            let startSample = Int(beatTime * sampleRate)

            for i in 0..<clickSamples {
                let sampleIndex = startSample + i
                guard sampleIndex < totalSamples else { break }

                let phase = 2.0 * Float.pi * clickFrequency * Float(i) / sampleRate
                let envelope = 1.0 - Float(i) / Float(clickSamples)  // Decay
                clickTrack[sampleIndex] = sin(phase) * envelope * 0.5
            }
        }

        return clickTrack
    }
}
