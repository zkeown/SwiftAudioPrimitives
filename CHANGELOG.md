# Changelog

All notable changes to SwiftRosa will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-12

### Added

- **GRU Support**: Complete GRU (Gated Recurrent Unit) implementation for SwiftRosaNN
  - `GRUCell` for single-timestep processing
  - `GRULayer` for full sequence processing
  - `BiGRULayer` for bidirectional GRU
  - `GRUConfig`, `GRUState`, `GRUWeights` configuration and state types
  - Metal-accelerated GRU kernels (`gru_state_update`, `gru_add_bias_ih`, `gru_add_bias_hh`)
  - ~25% less memory than equivalent LSTM (3 gates vs 4, no cell state)
  - PyTorch-compatible weight format and gate ordering (reset, update, new)
- `WeightLoader.loadGRUWeights()` for loading GRU model weights from binary files
- Comprehensive GRU test suite with PyTorch reference validation
- Reference data generation script `scripts/generate_gru_reference.py`

### Changed

- `LSTMEngine` now supports both LSTM and GRU cell operations via `gruCellForward()`
- `NNShaders.metal` includes GRU-specific compute kernels
- Improved buffer management consistency in GRU operations

## [1.0.0] - 2025-12-12

### Added

- Complete restructure as SwiftRosa modular package ecosystem
- **SwiftRosaCore**: Foundation DSP primitives
  - STFT/ISTFT with configurable windows
  - Mel spectrograms and filterbanks
  - Window functions (Hann, Hamming, Blackman, Bartlett, rectangular)
  - Power/amplitude dB conversion
  - Metal GPU acceleration engine
- **SwiftRosaAnalysis**: Audio analysis algorithms
  - Onset detection with peak picking
  - Pitch detection (fundamental frequency estimation)
  - Voice activity detection (energy, spectral, combined methods)
  - Rhythm analysis and beat tracking
- **SwiftRosaEffects**: Advanced transforms
  - CQT (Constant-Q Transform) and VQT (Variable-Q Transform)
  - MFCC (Mel-frequency cepstral coefficients)
  - Chromagram (pitch class profiles)
  - Tonnetz (tonal centroid features)
  - High-quality resampling
  - HPSS (Harmonic-percussive source separation)
  - Griffin-Lim phase reconstruction
  - DTW (Dynamic time warping)
- **SwiftRosaStreaming**: Real-time processing
  - StreamingSTFT and StreamingISTFT
  - StreamingMel and StreamingMFCC
  - StreamingPitchDetector
  - Ring buffer architecture for low-latency processing
- **SwiftRosaML**: Machine learning integration
  - SourceSeparationPipeline for end-to-end source separation
  - AudioFileIO for audio file reading/writing
  - AudioEngineCapture for live audio
  - SpecAugment for data augmentation
  - BenchmarkRunner for performance profiling
- **SwiftRosaNN**: Neural network primitives
  - LSTM and BiLSTM layers
  - Metal GPU acceleration for LSTM inference
  - Weight loading utilities
- **SwiftRosa**: Umbrella package re-exporting all modules
- DocC documentation for all packages
- CI/CD with librosa reference validation tests
- GitHub Actions workflow for automated testing

### Changed

- Renamed from SwiftAudioPrimitives to SwiftRosa
- Repository moved to github.com/zkeown/SwiftRosa
- Restructured from single package to 6-package modular architecture

### Migration from SwiftAudioPrimitives

If upgrading from the original SwiftAudioPrimitives:

1. Update your `Package.swift` dependency URL:
   ```swift
   // Old
   .package(url: "https://github.com/zakkeown/SwiftAudioPrimitives.git", from: "0.1.0")

   // New
   .package(url: "https://github.com/zkeown/SwiftRosa.git", from: "1.0.0")
   ```

2. Update import statements:
   ```swift
   // Old
   import SwiftAudioPrimitives

   // New (umbrella - imports everything)
   import SwiftRosa

   // Or (individual packages for smaller binaries)
   import SwiftRosaCore
   import SwiftRosaEffects
   ```

3. API remains compatible - existing code using STFT, MelSpectrogram, etc. should work without changes.
