import SwiftRosaCore
import Accelerate
import Foundation

/// Configuration for Harmonic-Percussive Source Separation.
public struct HPSSConfig: Sendable {
    /// Kernel size for median filtering (harmonic, percussive).
    public let kernelSize: (harmonic: Int, percussive: Int)

    /// Power for soft masking (2.0 = Wiener filtering).
    public let power: Float

    /// Margin for mask separation (1.0 = H + P = S, >1.0 allows residual).
    public let margin: Float

    /// Create HPSS configuration.
    ///
    /// - Parameters:
    ///   - kernelSize: Median filter kernel sizes (default: (31, 31)).
    ///   - power: Soft mask power (default: 2.0).
    ///   - margin: Separation margin (default: 1.0).
    public init(
        kernelSize: (harmonic: Int, percussive: Int) = (31, 31),
        power: Float = 2.0,
        margin: Float = 1.0
    ) {
        // Ensure odd kernel sizes
        self.kernelSize = (
            harmonic: kernelSize.harmonic | 1,
            percussive: kernelSize.percussive | 1
        )
        self.power = power
        self.margin = margin
    }
}

/// Result of HPSS separation.
public struct HPSSResult: Sendable {
    /// Harmonic component (sustained tones).
    public let harmonic: [Float]

    /// Percussive component (transients).
    public let percussive: [Float]

    /// Residual component (when margin > 1).
    public let residual: [Float]?
}

/// HPSS soft masks for spectrogram-domain processing.
public struct HPSSMasks: Sendable {
    /// Mask for harmonic component.
    public let harmonicMask: [[Float]]

    /// Mask for percussive component.
    public let percussiveMask: [[Float]]
}

/// Harmonic-Percussive Source Separation.
///
/// HPSS separates an audio signal into harmonic (sustained) and percussive
/// (transient) components using median filtering in the time-frequency domain.
///
/// ## Algorithm Overview
///
/// 1. Compute STFT of input signal
/// 2. Apply median filter along time axis → harmonic-enhanced spectrogram
/// 3. Apply median filter along frequency axis → percussive-enhanced spectrogram
/// 4. Compute soft masks using Wiener filtering
/// 5. Apply masks to original spectrogram
/// 6. Reconstruct audio via inverse STFT
///
/// ## Example Usage
///
/// ```swift
/// let hpss = HPSS()
///
/// // Separate into harmonic and percussive
/// let result = await hpss.separate(audioSignal)
///
/// // Get only harmonic component
/// let harmonic = await hpss.harmonic(audioSignal)
/// ```
public struct HPSS: Sendable {
    /// HPSS configuration.
    public let config: HPSSConfig

    /// STFT configuration.
    public let stftConfig: STFTConfig

    private let stft: STFT
    private let istft: ISTFT

    /// Create an HPSS processor.
    ///
    /// - Parameters:
    ///   - hpssConfig: HPSS configuration.
    ///   - stftConfig: STFT configuration.
    public init(
        hpssConfig: HPSSConfig = HPSSConfig(),
        stftConfig: STFTConfig = STFTConfig()
    ) {
        self.config = hpssConfig
        self.stftConfig = stftConfig
        self.stft = STFT(config: stftConfig)
        self.istft = ISTFT(config: stftConfig)
    }

    /// Separate signal into harmonic and percussive components.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: HPSSResult with harmonic, percussive, and optional residual.
    public func separate(_ signal: [Float]) async -> HPSSResult {
        guard !signal.isEmpty else {
            return HPSSResult(harmonic: [], percussive: [], residual: nil)
        }

        let targetLength = signal.count

        // Compute STFT
        let spectrogram = await stft.transform(signal)

        // Get masks
        let masks = separateMasks(spectrogram: spectrogram)

        // Apply masks to complex spectrogram
        let harmonicSpec = applyMask(spectrogram, mask: masks.harmonicMask)
        let percussiveSpec = applyMask(spectrogram, mask: masks.percussiveMask)

        // Reconstruct audio
        var harmonic = await istft.transform(harmonicSpec)
        var percussive = await istft.transform(percussiveSpec)

        // Match output length to input length
        harmonic = matchLength(harmonic, targetLength: targetLength)
        percussive = matchLength(percussive, targetLength: targetLength)

        // Compute residual if margin > 1
        var residual: [Float]? = nil
        if config.margin > 1.0 {
            residual = [Float](repeating: 0, count: targetLength)
            for i in 0..<targetLength {
                residual![i] = signal[i] - harmonic[i] - percussive[i]
            }
        }

        return HPSSResult(
            harmonic: harmonic,
            percussive: percussive,
            residual: residual
        )
    }

    /// Match array length to target by trimming or zero-padding.
    private func matchLength(_ array: [Float], targetLength: Int) -> [Float] {
        if array.count == targetLength {
            return array
        } else if array.count > targetLength {
            return Array(array.prefix(targetLength))
        } else {
            var result = array
            result.append(contentsOf: [Float](repeating: 0, count: targetLength - array.count))
            return result
        }
    }

    /// Get only harmonic component.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Harmonic audio signal.
    public func harmonic(_ signal: [Float]) async -> [Float] {
        let result = await separate(signal)
        return result.harmonic
    }

    /// Get only percussive component.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: Percussive audio signal.
    public func percussive(_ signal: [Float]) async -> [Float] {
        let result = await separate(signal)
        return result.percussive
    }

    /// Compute separation masks from magnitude spectrogram.
    ///
    /// - Parameter magnitude: Magnitude spectrogram.
    /// - Returns: Harmonic and percussive masks.
    public func separateMasks(magnitude: [[Float]]) -> HPSSMasks {
        guard !magnitude.isEmpty && !magnitude[0].isEmpty else {
            return HPSSMasks(harmonicMask: [], percussiveMask: [])
        }

        let nFreqs = magnitude.count
        let nFrames = magnitude[0].count

        // OPTIMIZATION: Convert to contiguous storage for cache-friendly operations
        let contiguousMag = ContiguousMatrix(from: magnitude)

        // Harmonic enhancement: median filter along time (horizontal) - already cache-friendly
        let harmonicEnhanced = medianFilterHorizontalContiguous(contiguousMag, kernelSize: config.kernelSize.harmonic)

        // Percussive enhancement: median filter along frequency (vertical) - use transpose trick
        let percussiveEnhanced = medianFilterVerticalContiguous(contiguousMag, kernelSize: config.kernelSize.percussive)

        // OPTIMIZATION: Vectorized soft mask computation
        let (harmonicMaskData, percussiveMaskData) = computeSoftMasksVectorized(
            harmonicEnhanced: harmonicEnhanced,
            percussiveEnhanced: percussiveEnhanced,
            power: config.power,
            margin: config.margin
        )

        // Convert back to [[Float]] for API compatibility
        return HPSSMasks(
            harmonicMask: harmonicMaskData.to2DArray(),
            percussiveMask: percussiveMaskData.to2DArray()
        )
    }

    /// Compute separation masks from complex spectrogram.
    private func separateMasks(spectrogram: ComplexMatrix) -> HPSSMasks {
        let magnitude = spectrogram.magnitude
        return separateMasks(magnitude: magnitude)
    }

    // MARK: - Private Implementation

    /// Apply mask to complex spectrogram.
    private func applyMask(_ spectrogram: ComplexMatrix, mask: [[Float]]) -> ComplexMatrix {
        let nFreqs = spectrogram.rows
        let nFrames = spectrogram.cols
        let count = nFreqs * nFrames

        // OPTIMIZATION: Vectorized mask application
        var realMasked = [Float](repeating: 0, count: count)
        var imagMasked = [Float](repeating: 0, count: count)

        // Flatten mask
        var flatMask = [Float](repeating: 0, count: count)
        for f in 0..<nFreqs {
            let offset = f * nFrames
            mask[f].withUnsafeBufferPointer { ptr in
                memcpy(&flatMask[offset], ptr.baseAddress!, nFrames * MemoryLayout<Float>.size)
            }
        }

        // Flatten spectrogram
        var flatReal = [Float](repeating: 0, count: count)
        var flatImag = [Float](repeating: 0, count: count)
        for f in 0..<nFreqs {
            let offset = f * nFrames
            spectrogram.real[f].withUnsafeBufferPointer { ptr in
                memcpy(&flatReal[offset], ptr.baseAddress!, nFrames * MemoryLayout<Float>.size)
            }
            spectrogram.imag[f].withUnsafeBufferPointer { ptr in
                memcpy(&flatImag[offset], ptr.baseAddress!, nFrames * MemoryLayout<Float>.size)
            }
        }

        // Vectorized multiply
        vDSP_vmul(flatReal, 1, flatMask, 1, &realMasked, 1, vDSP_Length(count))
        vDSP_vmul(flatImag, 1, flatMask, 1, &imagMasked, 1, vDSP_Length(count))

        // Convert back to [[Float]]
        var realResult = [[Float]]()
        var imagResult = [[Float]]()
        realResult.reserveCapacity(nFreqs)
        imagResult.reserveCapacity(nFreqs)

        for f in 0..<nFreqs {
            let start = f * nFrames
            realResult.append(Array(realMasked[start..<(start + nFrames)]))
            imagResult.append(Array(imagMasked[start..<(start + nFrames)]))
        }

        return ComplexMatrix(real: realResult, imag: imagResult)
    }

    // MARK: - Optimized Median Filters

    /// Optimized horizontal median filter using vDSP_vsort with minimized allocations.
    private func medianFilterHorizontalContiguous(_ matrix: ContiguousMatrix, kernelSize: Int) -> ContiguousMatrix {
        let nFreqs = matrix.rows
        let nFrames = matrix.cols
        let halfKernel = kernelSize / 2

        var result = [Float](repeating: 0, count: nFreqs * nFrames)

        // Process rows in parallel - each row is independent and already contiguous
        result.withUnsafeMutableBufferPointer { resultPtr in
            matrix.data.withUnsafeBufferPointer { dataPtr in
                DispatchQueue.concurrentPerform(iterations: nFreqs) { f in
                    // Thread-local window buffer - allocate once per thread
                    var window = [Float](repeating: 0, count: kernelSize)
                    let rowOffset = f * nFrames

                    window.withUnsafeMutableBufferPointer { windowPtr in
                        for t in 0..<nFrames {
                            let tStart = max(0, t - halfKernel)
                            let tEnd = min(nFrames, t + halfKernel + 1)
                            let windowLen = tEnd - tStart

                            // Copy window from contiguous row (cache-friendly)
                            memcpy(windowPtr.baseAddress!, dataPtr.baseAddress! + rowOffset + tStart, windowLen * MemoryLayout<Float>.size)

                            // Sort window in place and take median
                            vDSP_vsort(windowPtr.baseAddress!, vDSP_Length(windowLen), 1)
                            resultPtr[rowOffset + t] = windowPtr[windowLen / 2]
                        }
                    }
                }
            }
        }

        return ContiguousMatrix(data: result, rows: nFreqs, cols: nFrames)
    }

    /// Optimized vertical median filter using transpose trick for cache-friendly access.
    private func medianFilterVerticalContiguous(_ matrix: ContiguousMatrix, kernelSize: Int) -> ContiguousMatrix {
        let nFreqs = matrix.rows
        let nFrames = matrix.cols

        // OPTIMIZATION: Transpose (nFreqs, nFrames) -> (nFrames, nFreqs)
        // This makes vertical filtering become horizontal (cache-friendly)
        var transposed = [Float](repeating: 0, count: nFreqs * nFrames)
        matrix.data.withUnsafeBufferPointer { ptr in
            vDSP_mtrans(ptr.baseAddress!, 1, &transposed, 1,
                        vDSP_Length(nFrames), vDSP_Length(nFreqs))
        }

        let transposedMatrix = ContiguousMatrix(data: transposed, rows: nFrames, cols: nFreqs)

        // Apply horizontal filter on transposed data (which is vertical on original)
        let filtered = medianFilterHorizontalContiguous(transposedMatrix, kernelSize: kernelSize)

        // Transpose back: (nFrames, nFreqs) -> (nFreqs, nFrames)
        var result = [Float](repeating: 0, count: nFreqs * nFrames)
        filtered.data.withUnsafeBufferPointer { ptr in
            vDSP_mtrans(ptr.baseAddress!, 1, &result, 1,
                        vDSP_Length(nFreqs), vDSP_Length(nFrames))
        }

        return ContiguousMatrix(data: result, rows: nFreqs, cols: nFrames)
    }

    // MARK: - Vectorized Mask Computation

    /// Compute soft masks using vectorized Accelerate operations.
    private func computeSoftMasksVectorized(
        harmonicEnhanced: ContiguousMatrix,
        percussiveEnhanced: ContiguousMatrix,
        power: Float,
        margin: Float
    ) -> (harmonic: ContiguousMatrix, percussive: ContiguousMatrix) {
        let count = harmonicEnhanced.count
        let nFreqs = harmonicEnhanced.rows
        let nFrames = harmonicEnhanced.cols

        // Allocate output arrays
        var hMask = [Float](repeating: 0, count: count)
        var pMask = [Float](repeating: 0, count: count)

        // Temporary arrays for vectorized operations
        var hPower = [Float](repeating: 0, count: count)
        var pPower = [Float](repeating: 0, count: count)
        var total = [Float](repeating: 0, count: count)

        // Compute H^power and P^power using vForce
        var n = Int32(count)
        harmonicEnhanced.data.withUnsafeBufferPointer { hPtr in
            percussiveEnhanced.data.withUnsafeBufferPointer { pPtr in
                // H^power
                if power == 2.0 {
                    // Optimized path for power=2 (most common case - Wiener filtering)
                    vDSP_vsq(hPtr.baseAddress!, 1, &hPower, 1, vDSP_Length(count))
                    vDSP_vsq(pPtr.baseAddress!, 1, &pPower, 1, vDSP_Length(count))
                } else {
                    // General power using vvpowsf
                    var powerVal = power
                    var powerArray = [Float](repeating: powerVal, count: count)
                    vvpowf(&hPower, &powerArray, Array(hPtr), &n)
                    vvpowf(&pPower, &powerArray, Array(pPtr), &n)
                }
            }
        }

        // Apply margin: hPower *= margin, pPower *= margin
        var marginVal = margin
        vDSP_vsmul(hPower, 1, &marginVal, &hPower, 1, vDSP_Length(count))
        vDSP_vsmul(pPower, 1, &marginVal, &pPower, 1, vDSP_Length(count))

        // total = hPower + pPower
        vDSP_vadd(hPower, 1, pPower, 1, &total, 1, vDSP_Length(count))

        // Compute masks: hMask = hPower / total, pMask = pPower / total
        // Handle division by zero by adding small epsilon
        var epsilon: Float = 1e-10
        vDSP_vsadd(total, 1, &epsilon, &total, 1, vDSP_Length(count))
        vDSP_vdiv(total, 1, hPower, 1, &hMask, 1, vDSP_Length(count))
        vDSP_vdiv(total, 1, pPower, 1, &pMask, 1, vDSP_Length(count))

        return (
            ContiguousMatrix(data: hMask, rows: nFreqs, cols: nFrames),
            ContiguousMatrix(data: pMask, rows: nFreqs, cols: nFrames)
        )
    }
}

// MARK: - Presets

extension HPSS {
    /// Default configuration suitable for most audio.
    public static var standard: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (31, 31),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 2048,
                hopLength: 512,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Configuration optimized for separating vocals from drums.
    public static var vocalsFromDrums: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (15, 15),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 4096,
                hopLength: 1024,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Configuration with residual extraction.
    public static var withResidual: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (31, 31),
                power: 2.0,
                margin: 2.0
            ),
            stftConfig: STFTConfig(
                nFFT: 2048,
                hopLength: 512,
                windowType: .hann,
                center: true
            )
        )
    }

    /// Faster configuration with smaller kernel.
    public static var fast: HPSS {
        HPSS(
            hpssConfig: HPSSConfig(
                kernelSize: (11, 11),
                power: 2.0,
                margin: 1.0
            ),
            stftConfig: STFTConfig(
                nFFT: 1024,
                hopLength: 256,
                windowType: .hann,
                center: true
            )
        )
    }
}

// MARK: - GPU Acceleration

extension HPSS {
    /// Compute separation masks using GPU acceleration.
    ///
    /// For large spectrograms (> 50K elements), uses Metal GPU for median filtering
    /// and mask computation.
    ///
    /// - Parameter magnitude: Magnitude spectrogram.
    /// - Returns: Harmonic and percussive masks.
    public func separateMasksGPU(magnitude: [[Float]]) async -> HPSSMasks {
        guard !magnitude.isEmpty && !magnitude[0].isEmpty else {
            return HPSSMasks(harmonicMask: [], percussiveMask: [])
        }

        let nFreqs = magnitude.count
        let nFrames = magnitude[0].count
        let totalElements = nFreqs * nFrames

        // Check if GPU should be used
        guard MetalEngine.isAvailable &&
              MetalEngine.shouldUseGPU(elementCount: totalElements, operationType: .general) else {
            return separateMasks(magnitude: magnitude)
        }

        // Initialize Metal engine
        guard let engine = try? MetalEngine() else {
            return separateMasks(magnitude: magnitude)
        }

        // Flatten magnitude to row-major
        var flatMag = [Float](repeating: 0, count: totalElements)
        for f in 0..<nFreqs {
            for t in 0..<nFrames {
                flatMag[f * nFrames + t] = magnitude[f][t]
            }
        }

        do {
            // GPU median filtering
            let harmonicEnhanced = try await engine.hpssMedianFilterHorizontal(
                input: flatMag,
                nFreqs: nFreqs,
                nFrames: nFrames,
                kernelSize: config.kernelSize.harmonic
            )

            let percussiveEnhanced = try await engine.hpssMedianFilterVertical(
                input: flatMag,
                nFreqs: nFreqs,
                nFrames: nFrames,
                kernelSize: config.kernelSize.percussive
            )

            // GPU soft mask computation
            let (harmonicMask, percussiveMask) = try await engine.hpssSoftMask(
                harmonicEnhanced: harmonicEnhanced,
                percussiveEnhanced: percussiveEnhanced,
                power: config.power,
                margin: config.margin,
                epsilon: 1e-10
            )

            // Unflatten to 2D arrays
            var harmonicMask2D = [[Float]]()
            var percussiveMask2D = [[Float]]()
            harmonicMask2D.reserveCapacity(nFreqs)
            percussiveMask2D.reserveCapacity(nFreqs)

            for f in 0..<nFreqs {
                let start = f * nFrames
                harmonicMask2D.append(Array(harmonicMask[start..<(start + nFrames)]))
                percussiveMask2D.append(Array(percussiveMask[start..<(start + nFrames)]))
            }

            return HPSSMasks(
                harmonicMask: harmonicMask2D,
                percussiveMask: percussiveMask2D
            )
        } catch {
            // Fall back to CPU
            return separateMasks(magnitude: magnitude)
        }
    }

    /// Separate signal using GPU acceleration.
    ///
    /// - Parameter signal: Input audio signal.
    /// - Returns: HPSSResult with harmonic, percussive, and optional residual.
    public func separateGPU(_ signal: [Float]) async -> HPSSResult {
        guard !signal.isEmpty else {
            return HPSSResult(harmonic: [], percussive: [], residual: nil)
        }

        let targetLength = signal.count

        // Compute STFT
        let spectrogram = await stft.transform(signal)

        // Get masks using GPU
        let masks = await separateMasksGPU(magnitude: spectrogram.magnitude)

        // Apply masks to complex spectrogram
        let harmonicSpec = applyMask(spectrogram, mask: masks.harmonicMask)
        let percussiveSpec = applyMask(spectrogram, mask: masks.percussiveMask)

        // Reconstruct audio
        var harmonic = await istft.transform(harmonicSpec)
        var percussive = await istft.transform(percussiveSpec)

        // Match output length to input length
        harmonic = matchLength(harmonic, targetLength: targetLength)
        percussive = matchLength(percussive, targetLength: targetLength)

        // Compute residual if margin > 1
        var residual: [Float]? = nil
        if config.margin > 1.0 {
            residual = [Float](repeating: 0, count: targetLength)
            for i in 0..<targetLength {
                residual![i] = signal[i] - harmonic[i] - percussive[i]
            }
        }

        return HPSSResult(
            harmonic: harmonic,
            percussive: percussive,
            residual: residual
        )
    }
}
