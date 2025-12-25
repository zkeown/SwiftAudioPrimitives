/// Type aliases for common audio processing data structures.
///
/// These aliases improve code readability and provide self-documenting types
/// for audio signal processing operations.

/// A raw audio signal as a 1D array of Float samples.
public typealias AudioSignal = [Float]

/// A spectrogram represented as a 2D array of Float values.
/// Shape: (nFreqs, nFrames) where rows are frequency bins and columns are time frames.
public typealias Spectrogram = [[Float]]

/// A mel spectrogram represented as a 2D array of Float values.
/// Shape: (nMels, nFrames) where rows are mel bands and columns are time frames.
public typealias MelSpectrogramData = [[Float]]

/// MFCC features represented as a 2D array of Float values.
/// Shape: (nMFCC, nFrames) where rows are MFCC coefficients and columns are time frames.
public typealias MFCCData = [[Float]]

/// A complex spectrogram represented as separate real and imaginary components.
/// Both arrays have shape (nFreqs, nFrames).
public typealias ComplexSpectrogramTuple = (real: [[Float]], imag: [[Float]])
