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
    public var magnitude: [[Float]] {
        var result = [[Float]]()
        result.reserveCapacity(rows)

        for i in 0..<rows {
            var magnitudeRow = [Float](repeating: 0, count: cols)

            // Use vDSP for vectorized magnitude computation
            real[i].withUnsafeBufferPointer { realPtr in
                imag[i].withUnsafeBufferPointer { imagPtr in
                    magnitudeRow.withUnsafeMutableBufferPointer { outPtr in
                        // Compute real^2
                        var realSquared = [Float](repeating: 0, count: cols)
                        vDSP_vsq(realPtr.baseAddress!, 1, &realSquared, 1, vDSP_Length(cols))

                        // Compute imag^2
                        var imagSquared = [Float](repeating: 0, count: cols)
                        vDSP_vsq(imagPtr.baseAddress!, 1, &imagSquared, 1, vDSP_Length(cols))

                        // Add them
                        var sumSquared = [Float](repeating: 0, count: cols)
                        vDSP_vadd(realSquared, 1, imagSquared, 1, &sumSquared, 1, vDSP_Length(cols))

                        // Square root
                        var count = Int32(cols)
                        vvsqrtf(outPtr.baseAddress!, sumSquared, &count)
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
