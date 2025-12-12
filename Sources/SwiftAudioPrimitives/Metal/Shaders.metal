//
//  Shaders.metal
//  SwiftAudioPrimitives
//
//  Metal compute shaders for audio DSP operations.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Overlap-Add Kernel

/// Overlap-add kernel for ISTFT reconstruction.
///
/// Each thread handles one sample position within one frame.
/// Output is accumulated using atomic operations for thread safety.
///
/// Parameters:
/// - frames: Flattened array of IFFT output frames [frameCount * frameLength]
/// - window: Synthesis window [frameLength]
/// - output: Output signal buffer (must be pre-zeroed) [outputLength]
/// - frameCount: Number of frames
/// - frameLength: Length of each frame (nFFT)
/// - hopLength: Hop between frames
kernel void overlapAdd(
    device const float* frames [[buffer(0)]],
    device const float* window [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    constant uint& frameCount [[buffer(3)]],
    constant uint& frameLength [[buffer(4)]],
    constant uint& hopLength [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint sampleIdx = gid.x;
    uint frameIdx = gid.y;

    // Bounds check
    if (frameIdx >= frameCount || sampleIdx >= frameLength) {
        return;
    }

    // Calculate output position
    uint outputIdx = frameIdx * hopLength + sampleIdx;

    // Read frame value and apply window
    float frameValue = frames[frameIdx * frameLength + sampleIdx];
    float windowValue = window[sampleIdx];
    float windowedValue = frameValue * windowValue;

    // Accumulate to output (atomic for thread safety)
    atomic_fetch_add_explicit(&output[outputIdx], windowedValue, memory_order_relaxed);
}

/// Overlap-add with squared window normalization accumulation.
///
/// This kernel also accumulates the squared window values for normalization.
kernel void overlapAddWithNorm(
    device const float* frames [[buffer(0)]],
    device const float* window [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    device atomic_float* windowSum [[buffer(3)]],
    constant uint& frameCount [[buffer(4)]],
    constant uint& frameLength [[buffer(5)]],
    constant uint& hopLength [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint sampleIdx = gid.x;
    uint frameIdx = gid.y;

    if (frameIdx >= frameCount || sampleIdx >= frameLength) {
        return;
    }

    uint outputIdx = frameIdx * hopLength + sampleIdx;

    float frameValue = frames[frameIdx * frameLength + sampleIdx];
    float windowValue = window[sampleIdx];
    float windowedValue = frameValue * windowValue;
    float windowSquared = windowValue * windowValue;

    atomic_fetch_add_explicit(&output[outputIdx], windowedValue, memory_order_relaxed);
    atomic_fetch_add_explicit(&windowSum[outputIdx], windowSquared, memory_order_relaxed);
}

// MARK: - Mel Filterbank Application

/// Apply mel filterbank to spectrogram via matrix multiplication.
///
/// Computes: output[mel, frame] = sum_f(filterbank[mel, f] * spectrogram[f, frame])
///
/// Parameters:
/// - spectrogram: Power spectrogram [nFreqs * nFrames] (row-major, freq-first)
/// - filterbank: Mel filterbank matrix [nMels * nFreqs] (row-major)
/// - output: Output mel spectrogram [nMels * nFrames] (row-major)
/// - nFreqs: Number of frequency bins
/// - nMels: Number of mel bands
/// - nFrames: Number of time frames
kernel void applyMelFilterbank(
    device const float* spectrogram [[buffer(0)]],
    device const float* filterbank [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& nFreqs [[buffer(3)]],
    constant uint& nMels [[buffer(4)]],
    constant uint& nFrames [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint frameIdx = gid.x;
    uint melIdx = gid.y;

    if (frameIdx >= nFrames || melIdx >= nMels) {
        return;
    }

    float sum = 0.0f;

    // Dot product of filterbank row with spectrogram column
    for (uint f = 0; f < nFreqs; f++) {
        float filterVal = filterbank[melIdx * nFreqs + f];
        float specVal = spectrogram[f * nFrames + frameIdx];
        sum += filterVal * specVal;
    }

    output[melIdx * nFrames + frameIdx] = sum;
}

/// Tiled mel filterbank application for better cache utilization.
///
/// Uses threadgroup memory to cache tiles of the filterbank and spectrogram.
kernel void applyMelFilterbankTiled(
    device const float* spectrogram [[buffer(0)]],
    device const float* filterbank [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& nFreqs [[buffer(3)]],
    constant uint& nMels [[buffer(4)]],
    constant uint& nFrames [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    // Tile size
    constexpr uint TILE_SIZE = 16;

    // Threadgroup shared memory for tiles
    threadgroup float filterTile[TILE_SIZE][TILE_SIZE];
    threadgroup float specTile[TILE_SIZE][TILE_SIZE];

    uint frameIdx = gid.x;
    uint melIdx = gid.y;

    float sum = 0.0f;

    // Iterate over tiles along the frequency dimension
    uint numTiles = (nFreqs + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint freqOffset = t * TILE_SIZE;

        // Load filterbank tile
        uint filterRow = melIdx;
        uint filterCol = freqOffset + tid.x;
        if (filterRow < nMels && filterCol < nFreqs) {
            filterTile[tid.y][tid.x] = filterbank[filterRow * nFreqs + filterCol];
        } else {
            filterTile[tid.y][tid.x] = 0.0f;
        }

        // Load spectrogram tile
        uint specRow = freqOffset + tid.y;
        uint specCol = frameIdx;
        if (specRow < nFreqs && specCol < nFrames) {
            specTile[tid.y][tid.x] = spectrogram[specRow * nFrames + specCol];
        } else {
            specTile[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += filterTile[tid.y][k] * specTile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (melIdx < nMels && frameIdx < nFrames) {
        output[melIdx * nFrames + frameIdx] = sum;
    }
}

// MARK: - Complex Number Operations

/// Element-wise complex multiplication.
///
/// Computes: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
///
/// Parameters:
/// - aReal, aImag: First complex operand (flattened)
/// - bReal, bImag: Second complex operand (flattened)
/// - outReal, outImag: Output (flattened)
/// - count: Total number of elements
kernel void complexMultiply(
    device const float* aReal [[buffer(0)]],
    device const float* aImag [[buffer(1)]],
    device const float* bReal [[buffer(2)]],
    device const float* bImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    float ar = aReal[gid];
    float ai = aImag[gid];
    float br = bReal[gid];
    float bi = bImag[gid];

    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    outReal[gid] = ar * br - ai * bi;
    outImag[gid] = ar * bi + ai * br;
}

/// Compute magnitude of complex values.
///
/// Computes: magnitude = sqrt(real^2 + imag^2)
kernel void complexMagnitude(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    device float* magnitude [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    float r = real[gid];
    float i = imag[gid];
    magnitude[gid] = sqrt(r * r + i * i);
}

/// Compute phase angle of complex values.
///
/// Computes: phase = atan2(imag, real)
kernel void complexPhase(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    device float* phase [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    phase[gid] = atan2(imag[gid], real[gid]);
}

/// Convert polar to rectangular coordinates.
///
/// Computes: real = magnitude * cos(phase), imag = magnitude * sin(phase)
kernel void polarToRectangular(
    device const float* magnitude [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float* real [[buffer(2)]],
    device float* imag [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    float mag = magnitude[gid];
    float ph = phase[gid];
    real[gid] = mag * cos(ph);
    imag[gid] = mag * sin(ph);
}

// MARK: - Utility Kernels

/// Apply window function to signal frames.
///
/// Multiplies each frame element-wise by the window.
kernel void applyWindow(
    device float* frames [[buffer(0)]],
    device const float* window [[buffer(1)]],
    constant uint& frameLength [[buffer(2)]],
    constant uint& frameCount [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint sampleIdx = gid.x;
    uint frameIdx = gid.y;

    if (sampleIdx >= frameLength || frameIdx >= frameCount) {
        return;
    }

    uint idx = frameIdx * frameLength + sampleIdx;
    frames[idx] *= window[sampleIdx];
}

/// Normalize output by window sum (for ISTFT).
///
/// Divides each sample by the accumulated window sum, with epsilon for stability.
kernel void normalizeByWindowSum(
    device float* output [[buffer(0)]],
    device const float* windowSum [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    float sum = windowSum[gid];
    if (sum > epsilon) {
        output[gid] /= sum;
    }
}

/// Power to dB conversion.
///
/// Computes: dB = 10 * log10(max(power, amin))
kernel void powerToDb(
    device const float* power [[buffer(0)]],
    device float* db [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& amin [[buffer(3)]],
    constant float& refPower [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    float p = max(power[gid], amin);
    float ref = max(refPower, amin);
    db[gid] = 10.0f * log10(p / ref);
}

// MARK: - Vector Operations

/// Element-wise vector squaring (power of 2).
///
/// Computes: output[i] = input[i]^2
kernel void vectorSquare(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float val = input[gid];
    output[gid] = val * val;
}

/// Element-wise natural logarithm with floor.
///
/// Computes: output[i] = log(max(input[i], amin))
kernel void vectorLog(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& amin [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    output[gid] = log(max(input[gid], amin));
}

/// Element-wise exponential.
///
/// Computes: output[i] = exp(input[i])
kernel void vectorExp(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    output[gid] = exp(input[gid]);
}

/// Element-wise clamp.
///
/// Computes: output[i] = clamp(input[i], minVal, maxVal)
kernel void vectorClamp(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& minVal [[buffer(3)]],
    constant float& maxVal [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    output[gid] = clamp(input[gid], minVal, maxVal);
}

// MARK: - Spectral Operations

/// Compute spectral centroid for each frame.
///
/// centroid[t] = sum(freq[f] * mag[f,t]) / sum(mag[f,t])
kernel void spectralCentroid(
    device const float* magnitudes [[buffer(0)]],
    device const float* frequencies [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    constant uint& nFreqs [[buffer(3)]],
    constant uint& nFrames [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nFrames) {
        return;
    }

    float weightedSum = 0.0f;
    float totalWeight = 0.0f;

    for (uint f = 0; f < nFreqs; f++) {
        float mag = magnitudes[f * nFrames + gid];
        weightedSum += frequencies[f] * mag;
        totalWeight += mag;
    }

    centroids[gid] = (totalWeight > 0.0f) ? (weightedSum / totalWeight) : 0.0f;
}

/// Compute spectral flux (half-wave rectified difference between frames).
///
/// flux[t] = sum(max(0, mag[f,t] - mag[f,t-1]))
kernel void spectralFlux(
    device const float* magnitudes [[buffer(0)]],
    device float* flux [[buffer(1)]],
    constant uint& nFreqs [[buffer(2)]],
    constant uint& nFrames [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Frame 0 has no previous frame, flux = 0
    if (gid == 0 || gid >= nFrames) {
        if (gid == 0 && gid < nFrames) {
            flux[gid] = 0.0f;
        }
        return;
    }

    float fluxSum = 0.0f;

    for (uint f = 0; f < nFreqs; f++) {
        float curr = magnitudes[f * nFrames + gid];
        float prev = magnitudes[f * nFrames + (gid - 1)];
        float diff = curr - prev;
        fluxSum += max(0.0f, diff);  // Half-wave rectification
    }

    flux[gid] = fluxSum;
}

// MARK: - Resampling

/// Polyphase resampling kernel.
///
/// Performs upsampling by inserting zeros and filtering through polyphase branches.
/// Each output sample computes: sum(input[k] * filter[phase + k * nPhases])
///
/// Parameters:
/// - input: Input signal
/// - output: Output signal (pre-allocated to outputLength)
/// - filters: Polyphase filter coefficients (nPhases * filterLen)
/// - inputLength: Length of input signal
/// - outputLength: Length of output signal
/// - upFactor: Upsampling factor (number of phases)
/// - downFactor: Downsampling factor
/// - filterLen: Length of each polyphase filter branch
kernel void polyphaseResample(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* filters [[buffer(2)]],
    constant uint& inputLength [[buffer(3)]],
    constant uint& outputLength [[buffer(4)]],
    constant uint& upFactor [[buffer(5)]],
    constant uint& downFactor [[buffer(6)]],
    constant uint& filterLen [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= outputLength) {
        return;
    }

    // Which phase and input index for this output sample
    uint n = gid * downFactor;
    uint phase = n % upFactor;
    uint inputIdx = n / upFactor;

    float sum = 0.0f;

    // Convolution with polyphase filter
    for (uint k = 0; k < filterLen; k++) {
        int idx = int(inputIdx) - int(k);
        if (idx >= 0 && idx < int(inputLength)) {
            float filterVal = filters[phase * filterLen + k];
            sum += input[idx] * filterVal;
        }
    }

    output[gid] = sum * float(upFactor);  // Scale by upsampling factor
}

// MARK: - Distance Computations

/// Compute squared Euclidean distance matrix between two feature sets.
///
/// dist[i,j] = sum_f((x[f,i] - y[f,j])^2)
///
/// Parameters:
/// - xFeatures: First feature matrix [nFeatures * nX] (feature-major)
/// - yFeatures: Second feature matrix [nFeatures * nY] (feature-major)
/// - distances: Output distance matrix [nX * nY] (row-major)
/// - nFeatures: Number of features
/// - nX: Number of samples in X
/// - nY: Number of samples in Y
kernel void euclideanDistanceMatrix(
    device const float* xFeatures [[buffer(0)]],
    device const float* yFeatures [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& nFeatures [[buffer(3)]],
    constant uint& nX [[buffer(4)]],
    constant uint& nY [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;  // X index
    uint j = gid.y;  // Y index

    if (i >= nX || j >= nY) {
        return;
    }

    float distSq = 0.0f;

    for (uint f = 0; f < nFeatures; f++) {
        float xi = xFeatures[f * nX + i];
        float yj = yFeatures[f * nY + j];
        float diff = xi - yj;
        distSq += diff * diff;
    }

    distances[i * nY + j] = sqrt(distSq);
}
