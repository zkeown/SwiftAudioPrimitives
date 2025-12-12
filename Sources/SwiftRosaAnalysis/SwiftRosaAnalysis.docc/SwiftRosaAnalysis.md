# ``SwiftRosaAnalysis``

Audio analysis algorithms for onset detection, pitch estimation, rhythm analysis, and voice activity detection.

## Overview

SwiftRosaAnalysis provides higher-level audio analysis capabilities built on top of SwiftRosaCore. These algorithms are essential for music information retrieval (MIR) and speech processing applications.

### Key Features

- **Onset Detection**: Detect note onsets and transients in audio
- **Pitch Detection**: Estimate fundamental frequency (F0) of audio signals
- **Voice Activity Detection**: Identify speech vs silence regions
- **Rhythm Analysis**: Beat tracking and tempo estimation

## Topics

### Onset Detection

- ``OnsetDetector``
- ``OnsetConfig``

### Pitch Detection

- ``PitchDetector``
- ``PitchConfig``

### Voice Activity Detection

- ``VAD``
- ``VADConfig``
- ``VADMethod``

### Rhythm Analysis

- ``BeatTracker``
- ``TempoEstimator``
