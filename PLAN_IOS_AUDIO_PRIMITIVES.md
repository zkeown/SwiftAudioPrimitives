# SwiftAudioPrimitives - iOS Audio DSP Library Plan

A modern Swift library providing librosa-compatible audio DSP primitives for iOS, enabling ML audio applications like Banquet/query-bandit to run on iPhone/iPad.

## Goal

Enable the Banquet/query-bandit music source separation model (~25M params) to run on iOS devices with:
- Real-time or near-real-time performance
- Core ML integration for neural network inference
- Complete STFT→Model→ISTFT pipeline

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SwiftAudioPrimitives                      │
├─────────────────────────────────────────────────────────────┤
│  Public API (async/await, Sendable types)                   │
│  ├── STFT / ISTFT                                           │
│  ├── MelSpectrogram                                         │
│  ├── Windows (hann, hamming, blackman, etc.)                │
│  └── Utilities (dB conversion, resampling)                  │
├─────────────────────────────────────────────────────────────┤
│  Core Engine                                                │
│  ├── Accelerate/vDSP FFT (primary)                          │
│  ├── BNNS for batched operations (optional)                 │
│  └── Metal compute shaders (overlap-add, large batches)     │
├─────────────────────────────────────────────────────────────┤
│  Integration Layer                                          │
│  ├── Core ML model loading/inference                        │
│  ├── AVFoundation audio I/O                                 │
│  └── AudioToolbox format conversion                         │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Core DSP Primitives (Foundation)

### 1.1 Window Functions
**Files:** `Sources/SwiftAudioPrimitives/Windows.swift`

```swift
public enum WindowType: String, CaseIterable, Sendable {
    case hann, hamming, blackman, bartlett, rectangular
}

public struct Windows: Sendable {
    public static func generate(_ type: WindowType, length: Int, periodic: Bool = true) -> [Float]
}
```

**Implementation:** Use vDSP where available, fallback to manual computation.

### 1.2 STFT (Short-Time Fourier Transform)
**Files:** `Sources/SwiftAudioPrimitives/STFT.swift`

```swift
public struct STFTConfig: Sendable {
    public let nFFT: Int           // FFT size (default: 2048)
    public let hopLength: Int      // Hop between frames (default: nFFT/4)
    public let windowType: WindowType
    public let center: Bool        // Pad signal for centered frames
    public let padMode: PadMode    // reflect, constant, etc.
}

public struct STFT: Sendable {
    public init(config: STFTConfig)

    /// Returns complex spectrogram of shape (nFreqs, nFrames)
    public func transform(_ signal: [Float]) async -> ComplexMatrix

    /// Magnitude spectrogram
    public func magnitude(_ signal: [Float]) async -> [[Float]]
}
```

**Implementation Strategy:**
1. Frame signal into overlapping windows (strided view or copy)
2. Apply window function
3. Use `vDSP_DFT_zrop_CreateSetup` for real-to-complex FFT
4. Return only positive frequencies (nFFT/2 + 1)

### 1.3 ISTFT (Inverse STFT) - CRITICAL
**Files:** `Sources/SwiftAudioPrimitives/ISTFT.swift`

```swift
public struct ISTFT: Sendable {
    public init(config: STFTConfig)

    /// Reconstruct signal from complex spectrogram
    public func transform(_ spectrogram: ComplexMatrix, length: Int?) async -> [Float]
}
```

**Implementation Strategy:**
1. Inverse FFT each frame via `vDSP_DFT_zop` (complex-to-real)
2. Apply synthesis window
3. Overlap-add reconstruction (this is the hard part)
4. Normalize by squared window sum

**Overlap-Add Options:**
- **Option A:** vDSP_vadd in a loop (simple, may be slow for long signals)
- **Option B:** Metal compute shader (parallel scatter-add with atomics)
- **Option C:** Pre-allocated buffer with index arithmetic

### 1.4 Complex Number Support
**Files:** `Sources/SwiftAudioPrimitives/Complex.swift`

```swift
public struct ComplexMatrix: Sendable {
    public let real: [[Float]]
    public let imag: [[Float]]

    public var magnitude: [[Float]] { get }
    public var phase: [[Float]] { get }

    public static func fromPolar(magnitude: [[Float]], phase: [[Float]]) -> ComplexMatrix
}
```

Use `DSPSplitComplex` internally for Accelerate compatibility.

## Phase 2: Mel-Scale Features

### 2.1 Mel Filterbank
**Files:** `Sources/SwiftAudioPrimitives/Mel.swift`

```swift
public struct MelFilterbank: Sendable {
    public let nMels: Int
    public let nFFT: Int
    public let sampleRate: Float
    public let fMin: Float
    public let fMax: Float
    public let htk: Bool           // HTK formula vs Slaney
    public let norm: MelNorm?      // "slaney" or nil

    /// Returns filterbank matrix of shape (nMels, nFreqs)
    public func filters() -> [[Float]]
}

public struct MelSpectrogram: Sendable {
    public init(sampleRate: Float, nFFT: Int, hopLength: Int, nMels: Int, ...)

    public func transform(_ signal: [Float]) async -> [[Float]]
}
```

### 2.2 dB Conversion
**Files:** `Sources/SwiftAudioPrimitives/Convert.swift`

```swift
public struct Convert {
    public static func powerToDb(_ S: [[Float]], ref: Float = 1.0, amin: Float = 1e-10, topDb: Float? = 80.0) -> [[Float]]
    public static func amplitudeToDb(_ S: [[Float]], ...) -> [[Float]]
    public static func dbToPower(_ S_db: [[Float]], ref: Float = 1.0) -> [[Float]]
    public static func dbToAmplitude(_ S_db: [[Float]], ...) -> [[Float]]
}
```

## Phase 3: Advanced Features

### 3.1 Griffin-Lim Algorithm
**Files:** `Sources/SwiftAudioPrimitives/GriffinLim.swift`

Phase reconstruction from magnitude spectrogram (useful for some audio generation tasks).

```swift
public struct GriffinLim: Sendable {
    public init(nIter: Int = 32, momentum: Float = 0.99, ...)
    public func reconstruct(_ magnitude: [[Float]]) async -> [Float]
}
```

### 3.2 Resampling
**Files:** `Sources/SwiftAudioPrimitives/Resample.swift`

```swift
public struct Resample {
    public static func resample(_ signal: [Float], origSR: Int, targetSR: Int) async -> [Float]
}
```

Use `vDSP_desamp` or polyphase filtering.

### 3.3 MFCCs (Optional)
**Files:** `Sources/SwiftAudioPrimitives/MFCC.swift`

Mel-frequency cepstral coefficients via DCT on mel spectrogram.

## Phase 4: Core ML Integration

### 4.1 Model Wrapper
**Files:** `Sources/SwiftAudioPrimitives/CoreML/AudioModelRunner.swift`

```swift
public actor AudioModelRunner {
    private let model: MLModel

    public init(modelURL: URL) async throws

    /// Run source separation on spectrogram
    public func separate(_ spectrogram: ComplexMatrix) async throws -> [ComplexMatrix]
}
```

### 4.2 End-to-End Pipeline
**Files:** `Sources/SwiftAudioPrimitives/CoreML/SourceSeparationPipeline.swift`

```swift
public actor SourceSeparationPipeline {
    public init(modelURL: URL, config: STFTConfig) async throws

    /// Full pipeline: audio → STFT → model → ISTFT → separated audio
    public func separate(_ audio: [Float]) async throws -> [[Float]]
}
```

## Phase 5: Performance Optimization

### 5.1 Metal Compute Shaders
**Files:** `Sources/SwiftAudioPrimitives/Metal/`

For operations that benefit from GPU parallelism:
- `OverlapAdd.metal` - Parallel overlap-add with atomic operations
- `BatchFFT.metal` - Batched FFT for multiple channels (if BNNS insufficient)
- `MelApply.metal` - Matrix multiply for mel filterbank

### 5.2 Streaming Support
**Files:** `Sources/SwiftAudioPrimitives/Streaming/`

For real-time applications:
```swift
public actor StreamingSTFT {
    public func pushSamples(_ samples: [Float]) async
    public func popFrames() async -> [ComplexMatrix]
}
```

## Testing Strategy

### Unit Tests
- Compare against librosa/scipy outputs for reference signals
- Test edge cases: zero-length, single sample, very long signals
- Numerical precision validation (Float32 tolerance ~1e-5)

### Integration Tests
- Round-trip: signal → STFT → ISTFT → signal (should match within tolerance)
- Full pipeline with mock Core ML model
- Memory usage profiling

### Device Tests
- Performance benchmarks on iPhone 8 (oldest target) vs iPhone 15 Pro
- Memory pressure testing
- Thermal throttling behavior

## Repository Structure

```
SwiftAudioPrimitives/
├── Package.swift
├── Sources/
│   └── SwiftAudioPrimitives/
│       ├── Core/
│       │   ├── Windows.swift
│       │   ├── STFT.swift
│       │   ├── ISTFT.swift
│       │   └── Complex.swift
│       ├── Mel/
│       │   ├── MelFilterbank.swift
│       │   ├── MelSpectrogram.swift
│       │   └── Convert.swift
│       ├── Advanced/
│       │   ├── GriffinLim.swift
│       │   ├── Resample.swift
│       │   └── MFCC.swift
│       ├── CoreML/
│       │   ├── AudioModelRunner.swift
│       │   └── SourceSeparationPipeline.swift
│       ├── Metal/
│       │   ├── Shaders.metal
│       │   └── MetalEngine.swift
│       └── Streaming/
│           └── StreamingSTFT.swift
├── Tests/
│   └── SwiftAudioPrimitivesTests/
│       ├── WindowsTests.swift
│       ├── STFTTests.swift
│       ├── ISTFTTests.swift
│       ├── MelTests.swift
│       └── RoundTripTests.swift
├── Benchmarks/
│   └── PerformanceTests.swift
└── README.md
```

## Dependencies

```swift
// Package.swift
let package = Package(
    name: "SwiftAudioPrimitives",
    platforms: [
        .iOS(.v16),        // Oldest reasonable target
        .macOS(.v13),      // For development/testing
    ],
    products: [
        .library(name: "SwiftAudioPrimitives", targets: ["SwiftAudioPrimitives"]),
    ],
    targets: [
        .target(
            name: "SwiftAudioPrimitives",
            dependencies: [],
            resources: [.process("Metal")]
        ),
        .testTarget(
            name: "SwiftAudioPrimitivesTests",
            dependencies: ["SwiftAudioPrimitives"]
        ),
    ]
)
```

**System Frameworks Used:**
- Accelerate (vDSP, vForce, BNNS)
- Metal (compute shaders)
- CoreML (model inference)
- AVFoundation (audio I/O - optional)

## Implementation Order

### Sprint 1: MVP (2-3 weeks of focused work)
1. Window functions (all types)
2. STFT (basic implementation)
3. ISTFT (basic overlap-add)
4. Round-trip validation tests

### Sprint 2: Mel Features (1-2 weeks)
1. Mel filterbank generation
2. MelSpectrogram
3. dB conversion utilities
4. Validation against librosa

### Sprint 3: Core ML Integration (1-2 weeks)
1. Convert Banquet model to Core ML format
2. AudioModelRunner wrapper
3. End-to-end pipeline
4. Integration tests

### Sprint 4: Optimization (1-2 weeks)
1. Profile on target devices
2. Metal shaders for bottlenecks
3. Memory optimization
4. Streaming support if needed

## Banquet/Query-Bandit Specific Notes

### Model Conversion
The Banquet model needs to be converted to Core ML format:
```bash
# Using coremltools
python -c "
import coremltools as ct
import torch
from banquet import QueryBandit

model = QueryBandit.from_pretrained(...)
traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=...)])
mlmodel.save('QueryBandit.mlpackage')
"
```

### Expected I/O Format
- **Input:** Complex spectrogram from STFT (magnitude + phase or real + imag)
- **Output:** Separated source spectrograms (vocals, drums, bass, other)
- **Post-processing:** ISTFT each source back to waveform

### Performance Targets
- iPhone 15 Pro: < 1x realtime (process 10s audio in < 10s)
- iPhone 12: < 2x realtime acceptable
- iPhone 8: < 5x realtime acceptable (batch processing only)

## Open Questions

1. **Float16 vs Float32:** Core ML supports Float16 for faster inference. Should DSP primitives support both?

2. **Chunked Processing:** For long audio files, process in chunks with overlap to manage memory?

3. **Background Processing:** Use `BGProcessingTask` for long separations on iOS?

4. **Model Variants:** Ship multiple Core ML model variants (Float16 small, Float32 accurate)?

---

## Next Steps

1. Create new repository: `SwiftAudioPrimitives`
2. Set up Swift package structure
3. Implement Phase 1 (Core DSP)
4. Validate against librosa reference outputs
5. Convert Banquet model to Core ML
6. Build end-to-end demo app
