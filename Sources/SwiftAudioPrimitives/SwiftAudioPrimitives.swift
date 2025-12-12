// SwiftAudioPrimitives
// Librosa-compatible audio DSP primitives for iOS

/// SwiftAudioPrimitives provides audio DSP primitives for iOS, designed for
/// compatibility with librosa and Python audio ML pipelines.
///
/// ## Quick Start
///
/// ```swift
/// import SwiftAudioPrimitives
///
/// // Compute STFT
/// let stft = STFT(config: STFTConfig(nFFT: 2048, hopLength: 512))
/// let spectrogram = await stft.transform(audioSignal)
///
/// // Compute mel spectrogram
/// let melSpec = MelSpectrogram(sampleRate: 22050, nMels: 128)
/// let melFeatures = await melSpec.transform(audioSignal)
///
/// // Convert to dB
/// let melDb = Convert.powerToDb(melFeatures)
///
/// // Reconstruct from spectrogram (ISTFT)
/// let istft = stft.inverse
/// let reconstructed = await istft.transform(spectrogram, length: audioSignal.count)
/// ```
///
/// ## Features
///
/// - **STFT/ISTFT**: Short-time Fourier transform with configurable windows
/// - **Mel Spectrograms**: Mel filterbank and mel spectrogram computation
/// - **Window Functions**: Hann, Hamming, Blackman, Bartlett, rectangular
/// - **dB Conversion**: Power/amplitude to dB and back
///
/// ## Compatibility
///
/// All functions are designed to produce outputs compatible with librosa,
/// making it easy to port Python audio ML models to iOS.

// All public types are automatically available through normal import
