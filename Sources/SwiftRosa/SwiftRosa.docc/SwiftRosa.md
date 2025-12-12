# ``SwiftRosa``

A modular Swift/Metal port of librosa for iOS and macOS.

## Overview

SwiftRosa brings production-grade audio DSP to Apple platforms, enabling ML audio applications like music source separation on iPhone and iPad. It provides librosa-compatible implementations of common audio transforms with GPU acceleration via Metal.

### Package Architecture

SwiftRosa is organized into specialized packages that can be imported individually or together:

| Package | Description |
|---------|-------------|
| **SwiftRosaCore** | Foundation DSP: STFT, Mel, Windows, Metal engine |
| **SwiftRosaAnalysis** | Onset, pitch, rhythm, VAD |
| **SwiftRosaEffects** | CQT, MFCC, chromagram, resample, HPSS |
| **SwiftRosaStreaming** | Real-time processing |
| **SwiftRosaML** | CoreML integration, file I/O |
| **SwiftRosaNN** | LSTM layers with Metal |

### Quick Start

```swift
import SwiftRosa

// Compute STFT
let stft = STFT(config: STFTConfig(nFFT: 2048, hopLength: 512))
let spectrogram = await stft.transform(audioSignal)

// Compute mel spectrogram
let melSpec = MelSpectrogram(sampleRate: 22050, nMels: 128)
let mel = await melSpec.transform(audioSignal)

// Reconstruct audio
let reconstructed = await stft.inverse.transform(spectrogram, length: audioSignal.count)
```

### Librosa Compatibility

SwiftRosa is designed to produce outputs compatible with Python's librosa:
- Default parameters match librosa defaults
- Mel filterbank uses Slaney normalization
- STFT uses centered frames with reflect padding
- Reference validation tests ensure numerical accuracy

## Topics

### Core Packages

- ``SwiftRosaCore``
- ``SwiftRosaAnalysis``
- ``SwiftRosaEffects``
- ``SwiftRosaStreaming``
- ``SwiftRosaML``
- ``SwiftRosaNN``
