# ``SwiftRosaStreaming``

Real-time audio processing with efficient ring buffer architecture.

## Overview

SwiftRosaStreaming provides streaming versions of core audio transforms, optimized for real-time processing with minimal latency. The ring buffer architecture enables efficient frame-by-frame processing without memory allocations in the hot path.

### Key Features

- **StreamingSTFT**: Process audio frame-by-frame as it arrives
- **StreamingISTFT**: Reconstruct audio with overlap-add in real-time
- **StreamingMel**: Compute mel spectrograms on streaming audio
- **StreamingMFCC**: Extract MFCC features in real-time
- **StreamingPitchDetector**: Continuous pitch tracking

### Architecture

All streaming processors use a ring buffer pattern:

1. Push audio samples as they arrive
2. Processor automatically handles windowing and overlap
3. Output frames are emitted when ready

This design minimizes latency while maintaining compatibility with the batch processing APIs in SwiftRosaCore.

## Topics

### Spectral Processing

- ``StreamingSTFT``
- ``StreamingISTFT``

### Feature Extraction

- ``StreamingMel``
- ``StreamingMFCC``

### Analysis

- ``StreamingPitchDetector``

### Utilities

- ``RingBuffer``
