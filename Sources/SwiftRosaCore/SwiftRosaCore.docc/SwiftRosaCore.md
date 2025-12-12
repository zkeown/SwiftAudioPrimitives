# ``SwiftRosaCore``

Foundation DSP primitives with Metal GPU acceleration for audio signal processing.

## Overview

SwiftRosaCore provides the fundamental building blocks for audio digital signal processing on Apple platforms. It includes Short-Time Fourier Transform (STFT), mel spectrograms, window functions, and a Metal-accelerated compute engine.

All components are designed to produce outputs compatible with Python's [librosa](https://librosa.org/) library, making it easy to port audio ML pipelines to iOS and macOS.

### Key Features

- **STFT/ISTFT**: Configurable Short-Time Fourier Transform with multiple window types
- **Mel Processing**: Mel filterbanks and mel spectrograms with Slaney normalization
- **Window Functions**: Hann, Hamming, Blackman, Bartlett, and rectangular windows
- **dB Conversion**: Power and amplitude to decibel scale conversions
- **Metal Acceleration**: GPU-accelerated operations for large transforms

### Performance

SwiftRosaCore uses Apple's Accelerate framework (vDSP) for CPU operations and Metal for GPU acceleration. The ``MetalEngine`` automatically handles device selection and shader compilation.

For most operations, GPU acceleration provides significant speedups when:
- FFT size exceeds 4096
- Processing many frames in batch
- Working with large spectrograms

## Topics

### Essentials

- ``STFT``
- ``STFTConfig``
- ``ISTFT``

### Mel Processing

- ``MelSpectrogram``
- ``MelFilterbank``

### Spectral Features

- ``SpectralFeatures``

### Window Functions

- ``Windows``
- ``WindowType``

### Utilities

- ``Convert``
- ``ComplexMatrix``

### Metal Acceleration

- ``MetalEngine``
