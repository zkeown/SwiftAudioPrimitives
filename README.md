# SwiftRosa

[![Swift 5.9](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![iOS 16+](https://img.shields.io/badge/iOS-16+-blue.svg)](https://developer.apple.com/ios/)
[![macOS 13+](https://img.shields.io/badge/macOS-13+-blue.svg)](https://developer.apple.com/macos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/zkeown/SwiftRosa/actions/workflows/validation.yml/badge.svg)](https://github.com/zkeown/SwiftRosa/actions)

**Modular Swift/Metal port of [librosa](https://librosa.org/) for iOS and macOS.**

SwiftRosa brings production-grade audio DSP to Apple platforms with GPU acceleration via Metal, enabling ML audio applications like music source separation on iPhone and iPad.

## Package Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SwiftRosa                            │  Umbrella
├─────────────────────────────────────────────────────────────┤
│                       SwiftRosaML                           │  Tier 3
│            CoreML integration, source separation            │
├─────────────────────────────────────────────────────────────┤
│                    SwiftRosaStreaming                       │  Tier 2.5
│                   Real-time processing                      │
├───────────────────┬───────────────────┬─────────────────────┤
│ SwiftRosaAnalysis │  SwiftRosaEffects │    SwiftRosaNN      │  Tier 2
│  Onset, Pitch,    │  CQT, MFCC,       │  LSTM, BiLSTM       │
│  Rhythm, VAD      │  Chromagram, DTW  │  Metal-accelerated  │
├───────────────────┴───────────────────┴─────────────────────┤
│                      SwiftRosaCore                          │  Tier 1
│     STFT, ISTFT, Mel spectrograms, Windows, Metal engine    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/zkeown/SwiftRosa.git", from: "1.0.0")
]
```

**Import options:**

```swift
// Import everything
import SwiftRosa

// Or import individual packages for smaller binaries
import SwiftRosaCore      // Just DSP primitives
import SwiftRosaEffects   // Add transforms like CQT, MFCC
import SwiftRosaML        // Add CoreML integration
```

## Quick Start

```swift
import SwiftRosa

// Load audio as [Float] (use SwiftRosaML's AudioFileIO for file loading)
let audio: [Float] = ...

// Compute STFT
let stft = STFT(config: STFTConfig(nFFT: 2048, hopLength: 512))
let spectrogram = await stft.transform(audio)

// Compute mel spectrogram
let melSpec = MelSpectrogram(sampleRate: 22050, nMels: 128)
let melFeatures = await melSpec.transform(audio)

// Convert to dB scale
let melDb = Convert.powerToDb(melFeatures)

// Reconstruct audio with ISTFT
let reconstructed = await stft.inverse.transform(spectrogram, length: audio.count)
```

## Packages

### SwiftRosaCore

Foundation DSP primitives with Metal GPU acceleration.

| Component | Description |
|-----------|-------------|
| `STFT` / `ISTFT` | Short-time Fourier transform with configurable windows |
| `MelSpectrogram` | Mel filterbank and mel spectrogram computation |
| `MelFilterbank` | Standalone mel filter generation |
| `Windows` | Hann, Hamming, Blackman, Bartlett, rectangular |
| `Convert` | Power/amplitude to dB and back |
| `MetalEngine` | GPU acceleration for large transforms |

### SwiftRosaAnalysis

Audio analysis algorithms.

| Component | Description |
|-----------|-------------|
| `OnsetDetector` | Onset strength and peak picking |
| `PitchDetector` | Fundamental frequency estimation |
| `VAD` | Voice activity detection (energy, spectral, combined) |
| Rhythm | Beat tracking and tempo estimation |

### SwiftRosaEffects

Advanced transforms and effects.

| Component | Description |
|-----------|-------------|
| `CQT` / `VQT` | Constant-Q and Variable-Q transforms |
| `MFCC` | Mel-frequency cepstral coefficients |
| `Chromagram` | Pitch class profiles |
| `Tonnetz` | Tonal centroid features |
| `Resample` | High-quality resampling |
| `HPSS` | Harmonic-percussive source separation |
| `GriffinLim` | Phase reconstruction |
| `DTW` | Dynamic time warping |

### SwiftRosaStreaming

Real-time audio processing with ring buffer architecture.

| Component | Description |
|-----------|-------------|
| `StreamingSTFT` | Frame-by-frame STFT |
| `StreamingISTFT` | Overlap-add reconstruction |
| `StreamingMel` | Real-time mel spectrograms |
| `StreamingMFCC` | Real-time MFCC extraction |
| `StreamingPitchDetector` | Continuous pitch tracking |

### SwiftRosaML

Machine learning integration.

| Component | Description |
|-----------|-------------|
| `SourceSeparationPipeline` | End-to-end source separation |
| `AudioFileIO` | Audio file reading/writing |
| `AudioEngineCapture` | Live audio capture |
| `SpecAugment` | Spectrogram augmentation |
| `BenchmarkRunner` | Performance profiling |

### SwiftRosaNN

Neural network primitives with Metal acceleration.

| Component | Description |
|-----------|-------------|
| `LSTMLayer` | LSTM recurrent layer |
| `BiLSTMLayer` | Bidirectional LSTM |
| `LSTMEngine` | Metal GPU acceleration |
| `WeightLoader` | Model weight utilities |

## Performance

SwiftRosa uses Apple's Accelerate framework (vDSP) for CPU operations and Metal for GPU acceleration.

**When GPU acceleration helps:**
- Large FFT sizes (>4096)
- Batch processing many frames
- LSTM inference with large hidden sizes

**CPU is often faster for:**
- Small transforms
- Single-frame processing
- Low-latency streaming

Most components accept a `useGPU` parameter:

```swift
let melSpec = MelSpectrogram(sampleRate: 22050, nMels: 128, useGPU: true)
let istft = ISTFT(config: config, useGPU: true)
```

## Librosa Compatibility

SwiftRosa is designed to produce outputs compatible with [librosa](https://librosa.org/):

- Default parameters match librosa defaults
- Mel filterbank uses Slaney formula (not HTK) by default
- STFT uses centered frames with reflect padding
- Reference validation tests ensure numerical compatibility

## Use Case: Music Source Separation

```swift
import SwiftRosa

// 1. Load audio
let audio = try await AudioFileIO.load(url: audioURL)

// 2. Compute spectrogram
let stft = STFT(config: STFTConfig(nFFT: 4096, hopLength: 1024))
let spectrogram = await stft.transform(audio.samples)

// 3. Run source separation model
let pipeline = try SourceSeparationPipeline(modelURL: modelURL)
let separated = try await pipeline.separate(spectrogram)

// 4. Reconstruct separated sources
let vocals = await stft.inverse.transform(separated.vocals, length: audio.samples.count)
let drums = await stft.inverse.transform(separated.drums, length: audio.samples.count)

// 5. Save outputs
try await AudioFileIO.save(vocals, to: vocalsURL, sampleRate: audio.sampleRate)
```

## Documentation

Full API documentation is available via DocC. Build locally with:

```bash
swift package generate-documentation --target SwiftRosa
```

Or browse individual package documentation:

```bash
swift package generate-documentation --target SwiftRosaCore
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

## License

MIT - See [LICENSE](LICENSE) for details.
