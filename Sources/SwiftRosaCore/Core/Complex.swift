import Accelerate
import Foundation

// MARK: - Error Types

/// Errors that can occur during complex matrix operations.
public enum ComplexMatrixError: Error, Sendable {
    /// Real and imaginary arrays have different row counts.
    case rowCountMismatch(real: Int, imag: Int)
    /// Real and imaginary arrays have different column counts.
    case columnCountMismatch(real: Int, imag: Int)
    /// Array size doesn't match expected dimensions.
    case sizeMismatch(expected: Int, actual: Int)
    /// Magnitude and phase arrays have different sizes.
    case polarSizeMismatch(magnitude: Int, phase: Int)
    /// Vector length doesn't match matrix columns for multiplication.
    case vectorSizeMismatch(expected: Int, actual: Int)
    /// Data array size doesn't match rows × cols.
    case dataSizeMismatch(expected: Int, actual: Int)
}

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
    /// - Note: For dimension validation, use `validated(real:imag:)` which throws on mismatch.
    public init(real: [[Float]], imag: [[Float]]) {
        self.real = real
        self.imag = imag
    }

    /// Create a complex matrix with dimension validation.
    ///
    /// - Parameters:
    ///   - real: Real component matrix.
    ///   - imag: Imaginary component matrix.
    /// - Throws: `ComplexMatrixError` if real and imag have different shapes.
    public static func validated(real: [[Float]], imag: [[Float]]) throws -> ComplexMatrix {
        guard real.count == imag.count else {
            throw ComplexMatrixError.rowCountMismatch(real: real.count, imag: imag.count)
        }
        if !real.isEmpty && !imag.isEmpty {
            guard real[0].count == imag[0].count else {
                throw ComplexMatrixError.columnCountMismatch(real: real[0].count, imag: imag[0].count)
            }
        }
        return ComplexMatrix(real: real, imag: imag)
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
    ///
    /// - Note: For validation, use `fromFlatValidated` which throws on size mismatch.
    public static func fromFlat(real: [Float], imag: [Float], rows: Int, cols: Int) -> ComplexMatrix {
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

    /// Create from flat arrays with size validation.
    ///
    /// - Throws: `ComplexMatrixError.sizeMismatch` if array sizes don't match rows × cols.
    public static func fromFlatValidated(real: [Float], imag: [Float], rows: Int, cols: Int) throws -> ComplexMatrix {
        let expected = rows * cols
        guard real.count == expected else {
            throw ComplexMatrixError.sizeMismatch(expected: expected, actual: real.count)
        }
        guard imag.count == expected else {
            throw ComplexMatrixError.sizeMismatch(expected: expected, actual: imag.count)
        }
        return fromFlat(real: real, imag: imag, rows: rows, cols: cols)
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
    /// - Note: For validation, use `validated(real:imag:rows:cols:)` which throws on size mismatch.
    public init(real: [Float], imag: [Float], rows: Int, cols: Int) {
        self.real = real
        self.imag = imag
        self.rows = rows
        self.cols = cols
    }

    /// Create a contiguous complex matrix with validation.
    ///
    /// - Throws: `ComplexMatrixError.dataSizeMismatch` if array sizes don't match rows × cols.
    public static func validated(real: [Float], imag: [Float], rows: Int, cols: Int) throws -> ContiguousComplexMatrix {
        let expected = rows * cols
        guard real.count == expected else {
            throw ComplexMatrixError.dataSizeMismatch(expected: expected, actual: real.count)
        }
        guard imag.count == expected else {
            throw ComplexMatrixError.dataSizeMismatch(expected: expected, actual: imag.count)
        }
        return ContiguousComplexMatrix(real: real, imag: imag, rows: rows, cols: cols)
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

// MARK: - Contiguous Real Matrix

/// A memory-efficient real-valued matrix using contiguous storage.
///
/// This type stores data in a flat contiguous array with row-major layout, providing:
/// - Single allocation instead of one per row (unlike `[[Float]]`)
/// - Better cache locality for sequential access
/// - More efficient vectorized operations
/// - Zero-cost flattening (data is already flat)
///
/// Use this type for magnitude spectrograms, mel spectrograms, and other
/// real-valued 2D data where performance is critical.
///
/// ## Example Usage
///
/// ```swift
/// // Create from STFT magnitude
/// let stft = STFT()
/// let spec = await stft.magnitudeContiguous(audio)
///
/// // Access elements
/// let value = spec[512, 100]
///
/// // Access a column (time frame) efficiently
/// spec.withColumn(100) { column in
///     // column is a pointer to contiguous freq bins
///     let centroid = computeCentroid(column)
/// }
/// ```
public struct ContiguousMatrix: Sendable {
    /// Data stored contiguously in row-major order.
    public private(set) var data: [Float]
    /// Number of rows (e.g., frequency bins for spectrograms).
    public let rows: Int
    /// Number of columns (e.g., time frames for spectrograms).
    public let cols: Int

    /// Total number of elements.
    @inline(__always)
    public var count: Int { rows * cols }

    /// Create a contiguous matrix from a flat array.
    ///
    /// - Parameters:
    ///   - data: Flat array in row-major order.
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    /// - Note: For validation, use `validated(data:rows:cols:)` which throws on size mismatch.
    public init(data: [Float], rows: Int, cols: Int) {
        self.data = data
        self.rows = rows
        self.cols = cols
    }

    /// Create a contiguous matrix with validation.
    ///
    /// - Throws: `ComplexMatrixError.dataSizeMismatch` if data size doesn't match rows × cols.
    public static func validated(data: [Float], rows: Int, cols: Int) throws -> ContiguousMatrix {
        let expected = rows * cols
        guard data.count == expected else {
            throw ComplexMatrixError.dataSizeMismatch(expected: expected, actual: data.count)
        }
        return ContiguousMatrix(data: data, rows: rows, cols: cols)
    }

    // Internal-only initializer for backward compatibility
    @inline(__always)
    init(uncheckedData data: [Float], rows: Int, cols: Int) {
        self.data = data
        self.rows = rows
        self.cols = cols
    }

    /// Create a zero-filled contiguous matrix.
    ///
    /// - Parameters:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    public init(rows: Int, cols: Int) {
        self.data = [Float](repeating: 0, count: rows * cols)
        self.rows = rows
        self.cols = cols
    }

    /// Create from a 2D array (array of arrays).
    ///
    /// - Parameter array: 2D array in row-major layout.
    public init(from array: [[Float]]) {
        let rows = array.count
        let cols = array.first?.count ?? 0
        self.rows = rows
        self.cols = cols

        // Flatten with pre-allocated capacity
        var flat = [Float]()
        flat.reserveCapacity(rows * cols)
        for row in array {
            flat.append(contentsOf: row)
        }
        self.data = flat
    }

    /// Access a specific element.
    ///
    /// - Parameters:
    ///   - row: Row index.
    ///   - col: Column index.
    /// - Returns: The value at (row, col).
    @inline(__always)
    public subscript(row: Int, col: Int) -> Float {
        get { data[row * cols + col] }
        set { data[row * cols + col] = newValue }
    }

    /// Get a row as a slice of the underlying storage.
    ///
    /// - Parameter row: Row index.
    /// - Returns: Array slice containing the row.
    @inline(__always)
    public func row(_ row: Int) -> ArraySlice<Float> {
        let start = row * cols
        return data[start..<(start + cols)]
    }

    /// Access column data through a closure with unsafe pointer.
    ///
    /// This provides efficient column access by computing indices and
    /// passing a stride-based view. For spectrograms, columns are time frames.
    ///
    /// - Parameters:
    ///   - col: Column index.
    ///   - body: Closure receiving base pointer, stride, and count.
    /// - Returns: The result of the closure.
    @inline(__always)
    public func withColumn<T>(
        _ col: Int,
        _ body: (_ basePointer: UnsafePointer<Float>, _ stride: Int, _ count: Int) -> T
    ) -> T {
        data.withUnsafeBufferPointer { ptr in
            body(ptr.baseAddress! + col, cols, rows)
        }
    }

    /// Access column data through a closure with mutable unsafe pointer.
    ///
    /// - Parameters:
    ///   - col: Column index.
    ///   - body: Closure receiving mutable base pointer, stride, and count.
    /// - Returns: The result of the closure.
    @inline(__always)
    public mutating func withMutableColumn<T>(
        _ col: Int,
        _ body: (_ basePointer: UnsafeMutablePointer<Float>, _ stride: Int, _ count: Int) -> T
    ) -> T {
        data.withUnsafeMutableBufferPointer { ptr in
            body(ptr.baseAddress! + col, cols, rows)
        }
    }

    /// Extract a column as an array.
    ///
    /// Note: This allocates a new array. Prefer `withColumn` for performance-critical code.
    ///
    /// - Parameter col: Column index.
    /// - Returns: Array containing the column values.
    public func column(_ col: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            result[r] = data[r * cols + col]
        }
        return result
    }

    /// Convert to 2D array format.
    ///
    /// Use for compatibility with APIs that require `[[Float]]` layout.
    public func to2DArray() -> [[Float]] {
        var result = [[Float]]()
        result.reserveCapacity(rows)

        for r in 0..<rows {
            let start = r * cols
            result.append(Array(data[start..<(start + cols)]))
        }

        return result
    }

    /// Apply element-wise operation using Accelerate.
    ///
    /// - Parameter transform: Closure that transforms the flat data array.
    /// - Returns: New matrix with transformed data.
    /// - Note: The transform must return an array of the same size.
    public func map(_ transform: ([Float]) -> [Float]) -> ContiguousMatrix {
        // Safe to use unchecked - transform should preserve size
        ContiguousMatrix(data: transform(data), rows: rows, cols: cols)
    }

    /// Square all elements (power spectrogram from magnitude).
    public var squared: ContiguousMatrix {
        var result = [Float](repeating: 0, count: count)
        vDSP_vsq(data, 1, &result, 1, vDSP_Length(count))
        // Safe to use unchecked - result has same size
        return ContiguousMatrix(data: result, rows: rows, cols: cols)
    }

    /// Take square root of all elements.
    public var sqrt: ContiguousMatrix {
        var result = [Float](repeating: 0, count: count)
        var n = Int32(count)
        vvsqrtf(&result, data, &n)
        // Safe to use unchecked - result has same size
        return ContiguousMatrix(data: result, rows: rows, cols: cols)
    }

    /// Take natural log of all elements (with floor for numerical stability).
    ///
    /// - Parameter floor: Minimum value before taking log (default: 1e-10).
    /// - Returns: Log-transformed matrix.
    public func log(floor: Float = 1e-10) -> ContiguousMatrix {
        var clamped = [Float](repeating: 0, count: count)
        var floorVal = floor
        var maxVal: Float = .greatestFiniteMagnitude
        vDSP_vclip(data, 1, &floorVal, &maxVal, &clamped, 1, vDSP_Length(count))

        var result = [Float](repeating: 0, count: count)
        var n = Int32(count)
        vvlogf(&result, clamped, &n)

        // Safe to use unchecked - we created result with correct size
        return ContiguousMatrix(data: result, rows: rows, cols: cols)
    }

    /// Compute sum along rows (result has length = cols).
    public func sumRows() -> [Float] {
        var result = [Float](repeating: 0, count: cols)

        data.withUnsafeBufferPointer { ptr in
            for c in 0..<cols {
                var sum: Float = 0
                vDSP_sve(ptr.baseAddress! + c, vDSP_Stride(cols), &sum, vDSP_Length(rows))
                result[c] = sum
            }
        }

        return result
    }

    /// Compute sum along columns (result has length = rows).
    public func sumCols() -> [Float] {
        var result = [Float](repeating: 0, count: rows)

        for r in 0..<rows {
            var sum: Float = 0
            let start = r * cols
            vDSP_sve(data, 1, &sum, vDSP_Length(cols))
            data.withUnsafeBufferPointer { ptr in
                vDSP_sve(ptr.baseAddress! + start, 1, &sum, vDSP_Length(cols))
            }
            result[r] = sum
        }

        return result
    }

    /// Matrix-vector multiplication: result = self * vector.
    ///
    /// - Parameter vector: Column vector (length must equal cols).
    /// - Returns: Result vector (length = rows).
    /// - Note: Vector length must equal cols; behavior is undefined otherwise.
    public func matVecMul(_ vector: [Float]) -> [Float] {

        var result = [Float](repeating: 0, count: rows)

        data.withUnsafeBufferPointer { matPtr in
            vector.withUnsafeBufferPointer { vecPtr in
                result.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_mmul(
                        matPtr.baseAddress!, 1,
                        vecPtr.baseAddress!, 1,
                        outPtr.baseAddress!, 1,
                        vDSP_Length(rows),
                        1,
                        vDSP_Length(cols)
                    )
                }
            }
        }

        return result
    }
}

// MARK: - Conversion Extensions

extension Array where Element == [Float] {
    /// Convert 2D array to contiguous matrix.
    public func toContiguous() -> ContiguousMatrix {
        ContiguousMatrix(from: self)
    }
}
