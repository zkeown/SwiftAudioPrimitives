# SwiftAudioPrimitives

Librosa-compatible audio DSP primitives for iOS, enabling ML audio applications like music source separation on iPhone/iPad.

## Features

- **STFT/ISTFT** - Short-time Fourier transform with configurable windows
- **Mel Spectrograms** - Mel filterbank and mel spectrogram computation
- **Window Functions** - Hann, Hamming, Blackman, Bartlett, rectangular
- **dB Conversion** - Power/amplitude to dB and back
- **Accelerate-powered** - Uses vDSP for optimized FFT operations

## Requirements

- iOS 16.0+ / macOS 13.0+
- Swift 5.9+

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/zakkeown/SwiftAudioPrimitives.git", from: "0.1.0")
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL.

## Quick Start

```swift
import SwiftAudioPrimitives

// Load your audio signal as [Float]
let audioSignal: [Float] = ...

// Compute STFT
let stft = STFT(config: STFTConfig(nFFT: 2048, hopLength: 512))
let spectrogram = await stft.transform(audioSignal)

// Get magnitude spectrogram
let magnitude = spectrogram.magnitude

// Compute mel spectrogram
let melSpec = MelSpectrogram(sampleRate: 22050, nMels: 128)
let melFeatures = await melSpec.transform(audioSignal)

// Convert to dB scale
let melDb = Convert.powerToDb(melFeatures)

// Reconstruct audio from spectrogram (ISTFT)
let istft = stft.inverse
let reconstructed = await istft.transform(spectrogram, length: audioSignal.count)
```

## API Reference

### STFT

```swift
let config = STFTConfig(
    nFFT: 2048,           // FFT size
    hopLength: 512,       // Hop between frames
    winLength: nil,       // Window length (default: nFFT)
    windowType: .hann,    // Window function
    center: true,         // Pad signal for centered frames
    padMode: .reflect     // Padding mode
)

let stft = STFT(config: config)
let spectrogram = await stft.transform(signal)  // ComplexMatrix
let magnitude = await stft.magnitude(signal)     // [[Float]]
let power = await stft.power(signal)             // [[Float]]
```

### ISTFT

```swift
let istft = ISTFT(config: config)
let reconstructed = await istft.transform(spectrogram, length: originalLength)

// Or use the convenience property:
let reconstructed = await stft.inverse.transform(spectrogram)
```

### Mel Spectrogram

```swift
let melSpec = MelSpectrogram(
    sampleRate: 22050,
    nFFT: 2048,
    hopLength: 512,
    nMels: 128,
    fMin: 0,
    fMax: 8000,
    htk: false,          // Use Slaney formula (librosa default)
    norm: .slaney        // Slaney normalization
)

let mel = await melSpec.transform(signal)
```

### dB Conversion

```swift
// Power to dB
let db = Convert.powerToDb(powerSpec, ref: 1.0, amin: 1e-10, topDb: 80.0)

// Amplitude to dB
let db = Convert.amplitudeToDb(magSpec, ref: 1.0, amin: 1e-5, topDb: 80.0)

// dB to power/amplitude
let power = Convert.dbToPower(db, ref: 1.0)
let amplitude = Convert.dbToAmplitude(db, ref: 1.0)
```

### Window Functions

```swift
let window = Windows.generate(.hann, length: 2048, periodic: true)

// Available types: .hann, .hamming, .blackman, .bartlett, .rectangular
```

## Librosa Compatibility

This library is designed to produce outputs compatible with [librosa](https://librosa.org/). Key compatibility notes:

- Default parameters match librosa defaults
- Mel filterbank uses Slaney formula by default (not HTK)
- STFT uses centered frames with reflect padding by default
- All outputs use Float32 precision

## Performance

- Uses Apple's Accelerate framework (vDSP) for FFT operations
- Async/await for non-blocking computation
- Sendable types for safe concurrent use

## Use Case: Music Source Separation

This library was designed to enable running ML audio models like [Banquet/query-bandit](https://github.com/Banquet/query-bandit) on iOS:

```swift
// 1. Load audio
let audio = loadAudioFile(url)

// 2. Compute STFT
let stft = STFT(config: STFTConfig(nFFT: 4096, hopLength: 1024))
let spectrogram = await stft.transform(audio)

// 3. Run Core ML model (converts spectrogram to separated sources)
let model = try MLModel(contentsOf: modelURL)
let separated = try await model.prediction(input: spectrogram)

// 4. Reconstruct audio with ISTFT
let vocals = await stft.inverse.transform(separated.vocals, length: audio.count)
let drums = await stft.inverse.transform(separated.drums, length: audio.count)
```

## License

MIT
