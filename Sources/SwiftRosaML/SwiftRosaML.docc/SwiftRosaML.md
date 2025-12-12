# ``SwiftRosaML``

Machine learning integration with CoreML, audio file I/O, and data augmentation.

## Overview

SwiftRosaML bridges SwiftRosa's audio processing with Apple's CoreML framework, enabling end-to-end audio ML pipelines on iOS and macOS. It includes utilities for audio file handling, live capture, and training data augmentation.

### Key Features

- **Source Separation**: End-to-end pipeline for models like Demucs, Spleeter
- **Audio I/O**: Read and write audio files in various formats
- **Live Capture**: Record from microphone with AudioEngine
- **Data Augmentation**: SpecAugment and audio augmentation for training
- **Benchmarking**: Profile and optimize your audio ML pipelines

### Use Case: Music Source Separation

```swift
// Load audio
let audio = try await AudioFileIO.load(url: songURL)

// Run separation
let pipeline = try SourceSeparationPipeline(modelURL: modelURL)
let separated = try await pipeline.separate(audio.samples)

// Save stems
try await AudioFileIO.save(separated.vocals, to: vocalsURL, sampleRate: audio.sampleRate)
```

## Topics

### CoreML Integration

- ``SourceSeparationPipeline``
- ``AudioModelRunner``
- ``ModelConverter``

### Audio I/O

- ``AudioFileIO``
- ``FormatConverter``
- ``AudioEngineCapture``

### Data Augmentation

- ``SpecAugment``
- ``SpecAugmentConfig``
- ``AudioAugmentation``
- ``FeatureNormalization``

### Benchmarking

- ``BenchmarkRunner``
- ``BenchmarkConfig``
- ``TimeMetrics``
- ``MemoryMetrics``
- ``ThroughputMetrics``
