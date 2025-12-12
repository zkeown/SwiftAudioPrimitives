# Contributing to SwiftRosa

Thank you for your interest in contributing to SwiftRosa! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- macOS 13.0+ (for Metal support)
- Xcode 15.0+ with Swift 5.9+
- Python 3.11+ (for reference data generation)

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/zkeown/SwiftRosa.git
   cd SwiftRosa
   ```

2. Build the project:
   ```bash
   swift build
   ```

3. Run tests:
   ```bash
   swift test
   ```

4. Generate reference data (requires Python with librosa):
   ```bash
   pip install librosa numpy scipy
   python scripts/generate_reference_data.py
   ```

## Code Style

### Swift Conventions

- Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)
- Use 4-space indentation
- Maximum line length of 120 characters
- Use `///` documentation comments for all public API

### Documentation Requirements

All public types, methods, and properties must have documentation comments:

```swift
/// Computes the Short-Time Fourier Transform of an audio signal.
///
/// - Parameter signal: The input audio samples as Float array.
/// - Returns: A complex-valued spectrogram matrix.
/// - Throws: `STFTError.invalidSignal` if the input is empty.
public func transform(_ signal: [Float]) async throws -> ComplexMatrix {
    // ...
}
```

### Naming Conventions

- Match librosa function names where applicable (e.g., `stft`, `melSpectrogram`)
- Use Swift naming conventions (camelCase for methods/properties, PascalCase for types)
- Prefix internal/private helpers with underscore if needed for clarity

## Testing

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Reference Validation Tests**: Compare outputs against librosa reference data
3. **GPU/CPU Parity Tests**: Ensure Metal and vDSP produce consistent results
4. **Streaming Tests**: Verify real-time processing correctness

### Running Specific Tests

```bash
# Run all tests
swift test

# Run validation tests only
swift test --filter "Validation"

# Run tests for a specific package
swift test --filter SwiftRosaCoreTests
```

### Adding Reference Tests

When adding new functionality that mirrors librosa:

1. Add the librosa function call to `scripts/generate_reference_data.py`
2. Run the script to generate reference data
3. Create a test that compares SwiftRosa output to the reference

```swift
func testMyFeatureMatchesLibrosa() async throws {
    let reference = try loadReference("my_feature")
    let result = await myFeature.compute(testSignal)

    XCTAssertEqual(result, reference, accuracy: 1e-5)
}
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass: `swift test`
2. Build succeeds: `swift build`
3. Documentation builds: `swift package generate-documentation`
4. Code follows style guidelines

### PR Guidelines

- Create a feature branch from `main`
- Write clear commit messages describing the change
- Include tests for new functionality
- Update documentation as needed
- Reference any related issues

### Commit Message Format

```
[Package] Brief description of change

Longer description if needed, explaining:
- What changed
- Why it changed
- Any breaking changes or migration notes
```

Example:
```
[SwiftRosaCore] Add Kaiser window function

Implements Kaiser window with configurable beta parameter,
matching librosa.filters.get_window('kaiser', N, beta=14).
```

## Package Structure

When adding new functionality, place it in the appropriate package:

| Package | Purpose |
|---------|---------|
| SwiftRosaCore | Foundation DSP (STFT, Mel, Windows, Metal) |
| SwiftRosaAnalysis | Analysis (onset, pitch, rhythm, VAD) |
| SwiftRosaEffects | Transforms and effects (CQT, MFCC, resample) |
| SwiftRosaStreaming | Real-time processing |
| SwiftRosaML | CoreML integration, file I/O |
| SwiftRosaNN | Neural network layers |

## Performance Considerations

- Use `@inlinable` for hot-path functions
- Prefer vDSP operations over manual loops
- Consider Metal acceleration for operations on >50K elements
- Use workspace patterns to avoid repeated allocations
- Profile with Instruments before optimizing

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
