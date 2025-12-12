# ``SwiftRosaEffects``

Advanced audio transforms and effects including CQT, MFCC, chromagrams, and more.

## Overview

SwiftRosaEffects provides a comprehensive suite of audio transforms commonly used in music analysis, speech processing, and audio ML applications. All transforms are designed to match librosa's output for easy model porting.

### Key Features

- **Constant-Q Transform**: Logarithmically-spaced frequency analysis
- **MFCC**: Mel-frequency cepstral coefficients for speech/music features
- **Chromagram**: Pitch class profiles for harmonic analysis
- **Tonnetz**: Tonal centroid features
- **HPSS**: Harmonic-percussive source separation
- **Resampling**: High-quality sample rate conversion

## Topics

### Frequency Transforms

- ``CQT``
- ``CQTConfig``
- ``VQT``
- ``PseudoCQT``
- ``HarmonicCQT``

### Feature Extraction

- ``MFCC``
- ``MFCCConfig``
- ``Chromagram``
- ``ChromaConfig``
- ``Tonnetz``

### Audio Effects

- ``Resample``
- ``ResampleConfig``
- ``HPSS``
- ``HPSSConfig``
- ``GriffinLim``
- ``PhaseVocoder``

### Sequence Analysis

- ``DTW``
- ``DTWConfig``

### Decomposition

- ``NMF``
- ``PCA``
