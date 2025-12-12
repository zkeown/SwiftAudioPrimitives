import Accelerate
import Foundation

/// A matrix of complex numbers stored in split format for Accelerate compatibility.
public struct ComplexMatrix: Sendable {
    /// Real components, shape: (rows, cols)
    public let real: [[Float]]
    /// Imaginary components, shape: (rows, cols)
    public let imag: [[Float]]

    /// Number of rows (frequency bins for spectrograms).
    public var rows: Int { real.count }

    /// Number of columns (time frames for spectrograms).
    public var cols: Int { real.first?.count ?? 0 }

    /// Create a complex matrix from real and imaginary parts.
    ///
    /// - Parameters:
    ///   - real: Real component matrix.
    ///   - imag: Imaginary component matrix.
    /// - Precondition: real and imag must have the same shape.
    public init(real: [[Float]], imag: [[Float]]) {
        precondition(real.count == imag.count, "Real and imag must have same number of rows")
        precondition(
            real.isEmpty || imag.isEmpty || real[0].count == imag[0].count,
            "Real and imag must have same number of columns"
        )
        self.real = real
        self.imag = imag
    }

    /// Create a zero-filled complex matrix.
    public init(rows: Int, cols: Int) {
        self.real = [[Float]](repeating: [Float](repeating: 0, count: cols), count: rows)
        self.imag = [[Float]](repeating: [Float](repeating: 0, count: cols), count: rows)
    }

    /// Compute magnitude: sqrt(real^2 + imag^2)
    ///
    /// Optimized to use vDSP_vdist for single-pass computation without intermediate allocations.
    public var magnitude: [[Float]] {
        var result = [[Float]]()
        result.reserveCapacity(rows)

        for i in 0..<rows {
            var magnitudeRow = [Float](repeating: 0, count: cols)

            // Use vDSP_vdist for optimized magnitude computation (single pass)
            real[i].withUnsafeBufferPointer { realPtr in
                imag[i].withUnsafeBufferPointer { imagPtr in
                    magnitudeRow.withUnsafeMutableBufferPointer { outPtr in
                        // vDSP_vdist computes sqrt(a^2 + b^2) in a single optimized pass
                        vDSP_vdist(
                            realPtr.baseAddress!, 1,
                            imagPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(cols)
                        )
                    }
                }
            }

            result.append(magnitudeRow)
        }

        return result
    }

    /// Compute phase angle: atan2(imag, real)
    public var phase: [[Float]] {
        var result = [[Float]]()
        result.reserveCapacity(rows)

        for i in 0..<rows {
            var phaseRow = [Float](repeating: 0, count: cols)

            real[i].withUnsafeBufferPointer { realPtr in
                imag[i].withUnsafeBufferPointer { imagPtr in
                    phaseRow.withUnsafeMutableBufferPointer { outPtr in
                        var count = Int32(cols)
                        vvatan2f(outPtr.baseAddress!, imagPtr.baseAddress!, realPtr.baseAddress!, &count)
                    }
                }
            }

            result.append(phaseRow)
        }

        return result
    }

    /// Create complex matrix from polar coordinates (magnitude and phase).
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude values.
    ///   - phase: Phase angles in radians.
    /// - Returns: Complex matrix with real = mag * cos(phase), imag = mag * sin(phase).
    public static func fromPolar(magnitude: [[Float]], phase: [[Float]]) -> ComplexMatrix {
        precondition(magnitude.count == phase.count, "Magnitude and phase must have same number of rows")

        let rows = magnitude.count
        guard rows > 0 else { return ComplexMatrix(rows: 0, cols: 0) }

        let cols = magnitude[0].count
        var real = [[Float]]()
        var imag = [[Float]]()
        real.reserveCapacity(rows)
        imag.reserveCapacity(rows)

        for i in 0..<rows {
            var realRow = [Float](repeating: 0, count: cols)
            var imagRow = [Float](repeating: 0, count: cols)

            magnitude[i].withUnsafeBufferPointer { magPtr in
                phase[i].withUnsafeBufferPointer { phasePtr in
                    // Compute cos(phase) and sin(phase)
                    var cosVals = [Float](repeating: 0, count: cols)
                    var sinVals = [Float](repeating: 0, count: cols)
                    var count = Int32(cols)

                    vvcosf(&cosVals, phasePtr.baseAddress!, &count)
                    vvsinf(&sinVals, phasePtr.baseAddress!, &count)

                    // real = mag * cos(phase)
                    vDSP_vmul(magPtr.baseAddress!, 1, cosVals, 1, &realRow, 1, vDSP_Length(cols))
                    // imag = mag * sin(phase)
                    vDSP_vmul(magPtr.baseAddress!, 1, sinVals, 1, &imagRow, 1, vDSP_Length(cols))
                }
            }

            real.append(realRow)
            imag.append(imagRow)
        }

        return ComplexMatrix(real: real, imag: imag)
    }

    /// Access a specific element.
    public subscript(row: Int, col: Int) -> (real: Float, imag: Float) {
        (real[row][col], imag[row][col])
    }
}

// MARK: - Flat Buffer Operations

extension ComplexMatrix {
    /// Flatten to contiguous arrays for Accelerate operations.
    public var flatReal: [Float] {
        real.flatMap { $0 }
    }

    /// Flatten to contiguous arrays for Accelerate operations.
    public var flatImag: [Float] {
        imag.flatMap { $0 }
    }

    /// Create from flat arrays.
    public static func fromFlat(real: [Float], imag: [Float], rows: Int, cols: Int) -> ComplexMatrix {
        precondition(real.count == rows * cols, "Real array size mismatch")
        precondition(imag.count == rows * cols, "Imag array size mismatch")

        var realMatrix = [[Float]]()
        var imagMatrix = [[Float]]()
        realMatrix.reserveCapacity(rows)
        imagMatrix.reserveCapacity(rows)

        for i in 0..<rows {
            let start = i * cols
            let end = start + cols
            realMatrix.append(Array(real[start..<end]))
            imagMatrix.append(Array(imag[start..<end]))
        }

        return ComplexMatrix(real: realMatrix, imag: imagMatrix)
    }
}

// MARK: - Contiguous Complex Matrix

/// A memory-efficient complex matrix using contiguous storage.
///
/// Unlike `ComplexMatrix` which uses `[[Float]]` (array of arrays), this type
/// stores data in flat contiguous arrays with row-major layout. This provides:
/// - Single allocation per component instead of one per row
/// - Better cache locality for sequential access
/// - Zero-cost flattening (data is already flat)
/// - More efficient vectorized operations
///
/// Use this type when memory efficiency is critical, especially for large
/// spectrograms or when performing many operations on the data.
///
/// ## Example Usage
///
/// ```swift
/// let matrix = ContiguousComplexMatrix(rows: 1025, cols: 1000)
///
/// // Access elements
/// let (r, i) = matrix[512, 100]
///
/// // Efficient magnitude computation
/// let mag = matrix.magnitude  // Single flat array
/// ```
public struct ContiguousComplexMatrix: Sendable {
    /// Real components stored contiguously in row-major order.
    public let real: [Float]
    /// Imaginary components stored contiguously in row-major order.
    public let imag: [Float]
    /// Number of rows (frequency bins for spectrograms).
    public let rows: Int
    /// Number of columns (time frames for spectrograms).
    public let cols: Int

    /// Total number of elements.
    public var count: Int { rows * cols }

    /// Create a contiguous complex matrix from flat arrays.
    ///
    /// - Parameters:
    ///   - real: Flat array of real components (row-major).
    ///   - imag: Flat array of imaginary components (row-major).
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    public init(real: [Float], imag: [Float], rows: Int, cols: Int) {
        precondition(real.count == rows * cols, "Real array size must equal rows * cols")
        precondition(imag.count == rows * cols, "Imag array size must equal rows * cols")
        self.real = real
        self.imag = imag
        self.rows = rows
        self.cols = cols
    }

    /// Create a zero-filled contiguous complex matrix.
    ///
    /// - Parameters:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    public init(rows: Int, cols: Int) {
        let size = rows * cols
        self.real = [Float](repeating: 0, count: size)
        self.imag = [Float](repeating: 0, count: size)
        self.rows = rows
        self.cols = cols
    }

    /// Access a specific element.
    ///
    /// - Parameters:
    ///   - row: Row index.
    ///   - col: Column index.
    /// - Returns: Tuple of (real, imaginary) values.
    @inline(__always)
    public subscript(row: Int, col: Int) -> (real: Float, imag: Float) {
        let idx = row * cols + col
        return (real[idx], imag[idx])
    }

    /// Compute magnitude: sqrt(real^2 + imag^2)
    ///
    /// Uses vDSP_vdist for optimal single-pass computation.
    /// Returns a flat array in row-major order.
    public var magnitude: [Float] {
        var result = [Float](repeating: 0, count: count)

        real.withUnsafeBufferPointer { realPtr in
            imag.withUnsafeBufferPointer { imagPtr in
                result.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_vdist(
                        realPtr.baseAddress!, 1,
                        imagPtr.baseAddress!, 1,
                        outPtr.baseAddress!, 1,
                        vDSP_Length(count)
                    )
                }
            }
        }

        return result
    }

    /// Compute phase angle: atan2(imag, real)
    ///
    /// Returns a flat array in row-major order.
    public var phase: [Float] {
        var result = [Float](repeating: 0, count: count)

        real.withUnsafeBufferPointer { realPtr in
            imag.withUnsafeBufferPointer { imagPtr in
                result.withUnsafeMutableBufferPointer { outPtr in
                    var n = Int32(count)
                    vvatan2f(outPtr.baseAddress!, imagPtr.baseAddress!, realPtr.baseAddress!, &n)
                }
            }
        }

        return result
    }

    /// Get a row as a slice of the underlying storage.
    ///
    /// - Parameter row: Row index.
    /// - Returns: Array slice containing the row's real and imaginary values.
    public func row(_ row: Int) -> (real: ArraySlice<Float>, imag: ArraySlice<Float>) {
        let start = row * cols
        let end = start + cols
        return (real[start..<end], imag[start..<end])
    }

    /// Convert to standard ComplexMatrix.
    ///
    /// Use this for compatibility with APIs that require `[[Float]]` layout.
    public func toComplexMatrix() -> ComplexMatrix {
        var realMatrix = [[Float]]()
        var imagMatrix = [[Float]]()
        realMatrix.reserveCapacity(rows)
        imagMatrix.reserveCapacity(rows)

        for i in 0..<rows {
            let start = i * cols
            let end = start + cols
            realMatrix.append(Array(real[start..<end]))
            imagMatrix.append(Array(imag[start..<end]))
        }

        return ComplexMatrix(real: realMatrix, imag: imagMatrix)
    }

    /// Create from polar coordinates (magnitude and phase).
    ///
    /// - Parameters:
    ///   - magnitude: Flat magnitude array (row-major).
    ///   - phase: Flat phase array (row-major).
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    /// - Returns: Contiguous complex matrix.
    public static func fromPolar(
        magnitude: [Float],
        phase: [Float],
        rows: Int,
        cols: Int
    ) -> ContiguousComplexMatrix {
        let count = rows * cols
        precondition(magnitude.count == count, "Magnitude size must equal rows * cols")
        precondition(phase.count == count, "Phase size must equal rows * cols")

        var real = [Float](repeating: 0, count: count)
        var imag = [Float](repeating: 0, count: count)

        // Compute cos and sin of phase
        var cosVals = [Float](repeating: 0, count: count)
        var sinVals = [Float](repeating: 0, count: count)
        var n = Int32(count)

        phase.withUnsafeBufferPointer { phasePtr in
            vvcosf(&cosVals, phasePtr.baseAddress!, &n)
            vvsinf(&sinVals, phasePtr.baseAddress!, &n)
        }

        // real = magnitude * cos(phase)
        // imag = magnitude * sin(phase)
        magnitude.withUnsafeBufferPointer { magPtr in
            vDSP_vmul(magPtr.baseAddress!, 1, cosVals, 1, &real, 1, vDSP_Length(count))
            vDSP_vmul(magPtr.baseAddress!, 1, sinVals, 1, &imag, 1, vDSP_Length(count))
        }

        return ContiguousComplexMatrix(real: real, imag: imag, rows: rows, cols: cols)
    }
}

// MARK: - Conversion Between Matrix Types

extension ComplexMatrix {
    /// Convert to contiguous storage format.
    ///
    /// This creates a copy with contiguous memory layout for better performance.
    public func toContiguous() -> ContiguousComplexMatrix {
        ContiguousComplexMatrix(
            real: flatReal,
            imag: flatImag,
            rows: rows,
            cols: cols
        )
    }
}

extension ContiguousComplexMatrix {
    /// Create from a standard ComplexMatrix.
    ///
    /// - Parameter matrix: The matrix to convert.
    public init(from matrix: ComplexMatrix) {
        self.init(
            real: matrix.flatReal,
            imag: matrix.flatImag,
            rows: matrix.rows,
            cols: matrix.cols
        )
    }
}
