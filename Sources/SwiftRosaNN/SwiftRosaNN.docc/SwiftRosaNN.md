# ``SwiftRosaNN``

Neural network primitives with Metal GPU acceleration for audio ML inference.

## Overview

SwiftRosaNN provides efficient implementations of neural network layers commonly used in audio ML models, with optional Metal GPU acceleration for improved performance on Apple Silicon.

### Key Features

- **LSTM**: Long Short-Term Memory layers for sequence modeling
- **BiLSTM**: Bidirectional LSTM for context-aware processing
- **Metal Acceleration**: GPU-accelerated LSTM inference
- **Weight Loading**: Utilities for loading PyTorch/TensorFlow weights

### Performance

The Metal-accelerated LSTM implementation provides significant speedups for:
- Large hidden sizes (>256)
- Long sequences
- Batch inference

For smaller models, CPU inference with Accelerate may be faster due to reduced overhead.

### Integration with CoreML

While CoreML handles most model inference, SwiftRosaNN is useful for:
- Custom layers not supported by CoreML
- Fine-grained control over inference
- Hybrid CPU/GPU processing pipelines

## Topics

### LSTM Layers

- ``LSTMLayer``
- ``LSTMConfig``
- ``LSTMCell``
- ``BiLSTMLayer``

### State Management

- ``LSTMState``
- ``LSTMWeights``

### Metal Acceleration

- ``LSTMEngine``

### Utilities

- ``WeightLoader``
