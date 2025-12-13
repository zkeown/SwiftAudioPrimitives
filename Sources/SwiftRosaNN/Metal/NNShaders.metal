//
//  NNShaders.metal
//  SwiftRosaNN
//
//  Metal compute shaders for neural network operations.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - LSTM Kernels

/// Fused LSTM state update kernel.
///
/// Computes the LSTM state update after gate pre-activations have been computed:
///   i = sigmoid(gates[0:H])      // input gate
///   f = sigmoid(gates[H:2H])     // forget gate
///   g = tanh(gates[2H:3H])       // cell gate
///   o = sigmoid(gates[3H:4H])    // output gate
///   c_new = f * c_prev + i * g
///   h_new = o * tanh(c_new)
///
/// Parameters:
/// - gates: Pre-computed gate activations [batchSize, 4*hiddenSize]
///          (W_ih @ x + W_hh @ h_prev, biases already added)
/// - c_prev: Previous cell state [batchSize, hiddenSize]
/// - h_out: Output hidden state [batchSize, hiddenSize]
/// - c_out: Output cell state [batchSize, hiddenSize]
/// - hiddenSize: Size of hidden dimension
/// - batchSize: Batch size
kernel void lstm_state_update(
    device const float* gates [[buffer(0)]],
    device const float* c_prev [[buffer(1)]],
    device float* h_out [[buffer(2)]],
    device float* c_out [[buffer(3)]],
    constant uint& hiddenSize [[buffer(4)]],
    constant uint& batchSize [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hiddenIdx = gid.x;
    uint batchIdx = gid.y;

    if (hiddenIdx >= hiddenSize || batchIdx >= batchSize) {
        return;
    }

    uint H = hiddenSize;
    uint gateOffset = batchIdx * (4 * H);
    uint stateOffset = batchIdx * H + hiddenIdx;

    // Read gate pre-activations (gates order: i, f, g, o)
    float i_gate = gates[gateOffset + hiddenIdx];               // Input gate
    float f_gate = gates[gateOffset + H + hiddenIdx];           // Forget gate
    float g_gate = gates[gateOffset + 2 * H + hiddenIdx];       // Cell gate
    float o_gate = gates[gateOffset + 3 * H + hiddenIdx];       // Output gate

    // Apply activations
    float i = 1.0f / (1.0f + exp(-i_gate));    // sigmoid
    float f = 1.0f / (1.0f + exp(-f_gate));    // sigmoid
    float g = tanh(g_gate);                     // tanh
    float o = 1.0f / (1.0f + exp(-o_gate));    // sigmoid

    // State update
    float c_prev_val = c_prev[stateOffset];
    float c_new = f * c_prev_val + i * g;
    float h_new = o * tanh(c_new);

    // Write outputs
    c_out[stateOffset] = c_new;
    h_out[stateOffset] = h_new;
}

/// Add LSTM biases to gate pre-activations.
///
/// Computes: gates += bias_ih + bias_hh
/// This is separated from matmul to allow MPS to handle the matmuls.
///
/// Parameters:
/// - gates: Gate pre-activations [batchSize, 4*hiddenSize], modified in-place
/// - bias_ih: Input-to-hidden bias [4*hiddenSize]
/// - bias_hh: Hidden-to-hidden bias [4*hiddenSize]
/// - gateSize: 4 * hiddenSize
/// - batchSize: Batch size
kernel void lstm_add_biases(
    device float* gates [[buffer(0)]],
    device const float* bias_ih [[buffer(1)]],
    device const float* bias_hh [[buffer(2)]],
    constant uint& gateSize [[buffer(3)]],
    constant uint& batchSize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint gateIdx = gid.x;
    uint batchIdx = gid.y;

    if (gateIdx >= gateSize || batchIdx >= batchSize) {
        return;
    }

    uint offset = batchIdx * gateSize + gateIdx;
    gates[offset] += bias_ih[gateIdx] + bias_hh[gateIdx];
}

/// Add two gate matrices element-wise.
///
/// Computes: output = a + b
/// Used to combine W_ih @ x and W_hh @ h results.
///
/// Parameters:
/// - a: First gate matrix [batchSize, gateSize]
/// - b: Second gate matrix [batchSize, gateSize]
/// - output: Result matrix [batchSize, gateSize]
/// - count: Total number of elements
kernel void lstm_add_gates(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }

    output[gid] = a[gid] + b[gid];
}

// MARK: - GRU Kernels

/// Fused GRU state update kernel.
///
/// Computes the GRU state update from separate input and hidden gate pre-activations.
/// The separation is required because the "new" gate uses the reset gate to modulate
/// only the hidden-to-hidden contribution:
///
///   r = sigmoid(gates_ih[0:H] + gates_hh[0:H])           // reset gate
///   z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])         // update gate
///   n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])      // new gate (reset modulates hh only)
///   h_new = (1 - z) * n + z * h_prev
///
/// Parameters:
/// - gates_ih: Input-to-hidden gate pre-activations [batchSize, 3*hiddenSize]
///             (W_ih @ x + bias_ih)
/// - gates_hh: Hidden-to-hidden gate pre-activations [batchSize, 3*hiddenSize]
///             (W_hh @ h_prev + bias_hh)
/// - h_prev: Previous hidden state [batchSize, hiddenSize]
/// - h_out: Output hidden state [batchSize, hiddenSize]
/// - hiddenSize: Size of hidden dimension
/// - batchSize: Batch size
kernel void gru_state_update(
    device const float* gates_ih [[buffer(0)]],
    device const float* gates_hh [[buffer(1)]],
    device const float* h_prev [[buffer(2)]],
    device float* h_out [[buffer(3)]],
    constant uint& hiddenSize [[buffer(4)]],
    constant uint& batchSize [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hiddenIdx = gid.x;
    uint batchIdx = gid.y;

    if (hiddenIdx >= hiddenSize || batchIdx >= batchSize) {
        return;
    }

    uint H = hiddenSize;
    uint gateOffset = batchIdx * (3 * H);
    uint stateOffset = batchIdx * H + hiddenIdx;

    // Read gate pre-activations (gates order: r, z, n)
    // Reset gate: sum of ih and hh contributions
    float r_ih = gates_ih[gateOffset + hiddenIdx];
    float r_hh = gates_hh[gateOffset + hiddenIdx];

    // Update gate: sum of ih and hh contributions
    float z_ih = gates_ih[gateOffset + H + hiddenIdx];
    float z_hh = gates_hh[gateOffset + H + hiddenIdx];

    // New gate: ih contribution and hh contribution (kept separate for reset modulation)
    float n_ih = gates_ih[gateOffset + 2 * H + hiddenIdx];
    float n_hh = gates_hh[gateOffset + 2 * H + hiddenIdx];

    // Compute reset gate: r = sigmoid(r_ih + r_hh)
    float r = 1.0f / (1.0f + exp(-(r_ih + r_hh)));

    // Compute update gate: z = sigmoid(z_ih + z_hh)
    float z = 1.0f / (1.0f + exp(-(z_ih + z_hh)));

    // Compute new gate: n = tanh(n_ih + r * n_hh)
    // Note: reset gate modulates only the hidden-to-hidden contribution
    float n = tanh(n_ih + r * n_hh);

    // State update: h_new = (1 - z) * n + z * h_prev
    float h_prev_val = h_prev[stateOffset];
    float h_new = (1.0f - z) * n + z * h_prev_val;

    // Write output
    h_out[stateOffset] = h_new;
}

/// Add GRU biases to gate pre-activations (input-to-hidden path).
///
/// Computes: gates_ih += bias_ih
/// This is separated to allow MPS to handle the matmuls.
///
/// Parameters:
/// - gates_ih: Gate pre-activations [batchSize, 3*hiddenSize], modified in-place
/// - bias_ih: Input-to-hidden bias [3*hiddenSize]
/// - gateSize: 3 * hiddenSize
/// - batchSize: Batch size
kernel void gru_add_bias_ih(
    device float* gates_ih [[buffer(0)]],
    device const float* bias_ih [[buffer(1)]],
    constant uint& gateSize [[buffer(2)]],
    constant uint& batchSize [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint gateIdx = gid.x;
    uint batchIdx = gid.y;

    if (gateIdx >= gateSize || batchIdx >= batchSize) {
        return;
    }

    uint offset = batchIdx * gateSize + gateIdx;
    gates_ih[offset] += bias_ih[gateIdx];
}

/// Add GRU biases to gate pre-activations (hidden-to-hidden path).
///
/// Computes: gates_hh += bias_hh
/// This is separated to allow MPS to handle the matmuls.
///
/// Parameters:
/// - gates_hh: Gate pre-activations [batchSize, 3*hiddenSize], modified in-place
/// - bias_hh: Hidden-to-hidden bias [3*hiddenSize]
/// - gateSize: 3 * hiddenSize
/// - batchSize: Batch size
kernel void gru_add_bias_hh(
    device float* gates_hh [[buffer(0)]],
    device const float* bias_hh [[buffer(1)]],
    constant uint& gateSize [[buffer(2)]],
    constant uint& batchSize [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint gateIdx = gid.x;
    uint batchIdx = gid.y;

    if (gateIdx >= gateSize || batchIdx >= batchSize) {
        return;
    }

    uint offset = batchIdx * gateSize + gateIdx;
    gates_hh[offset] += bias_hh[gateIdx];
}

// MARK: - Sequence Operations

/// Reverse sequence along time dimension.
///
/// For backward LSTM pass, reverses the input sequence.
/// Input layout: [batchSize, seqLen, featureSize] (row-major)
///
/// Parameters:
/// - input: Input sequence
/// - output: Reversed sequence
/// - batchSize: Batch dimension
/// - seqLen: Sequence length
/// - featureSize: Feature dimension
kernel void reverse_sequence(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batchSize [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& featureSize [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint featIdx = gid.x;
    uint timeIdx = gid.y;
    uint batchIdx = gid.z;

    if (featIdx >= featureSize || timeIdx >= seqLen || batchIdx >= batchSize) {
        return;
    }

    uint inputOffset = batchIdx * seqLen * featureSize +
                       timeIdx * featureSize +
                       featIdx;

    uint reversedTime = seqLen - 1 - timeIdx;
    uint outputOffset = batchIdx * seqLen * featureSize +
                        reversedTime * featureSize +
                        featIdx;

    output[outputOffset] = input[inputOffset];
}

// MARK: - BiLSTM Operations

/// Concatenate forward and backward LSTM outputs along feature dimension.
///
/// For BiLSTM, combines the outputs from both directions:
///   output[:, :, :H] = forward
///   output[:, :, H:] = backward
///
/// Input layout: [batchSize, seqLen, hiddenSize] for each direction
/// Output layout: [batchSize, seqLen, 2*hiddenSize]
///
/// Parameters:
/// - forward: Forward LSTM output [batchSize, seqLen, hiddenSize]
/// - backward: Backward LSTM output [batchSize, seqLen, hiddenSize]
/// - output: Concatenated output [batchSize, seqLen, 2*hiddenSize]
/// - batchSize: Batch dimension
/// - seqLen: Sequence length
/// - hiddenSize: Hidden dimension (per direction)
kernel void concat_bidirectional(
    device const float* forward [[buffer(0)]],
    device const float* backward [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batchSize [[buffer(3)]],
    constant uint& seqLen [[buffer(4)]],
    constant uint& hiddenSize [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hiddenIdx = gid.x;
    uint timeIdx = gid.y;
    uint batchIdx = gid.z;

    if (hiddenIdx >= hiddenSize * 2 || timeIdx >= seqLen || batchIdx >= batchSize) {
        return;
    }

    uint outputOffset = batchIdx * seqLen * hiddenSize * 2 +
                        timeIdx * hiddenSize * 2 +
                        hiddenIdx;

    uint sourceOffset = batchIdx * seqLen * hiddenSize +
                        timeIdx * hiddenSize;

    if (hiddenIdx < hiddenSize) {
        // First half: forward output
        output[outputOffset] = forward[sourceOffset + hiddenIdx];
    } else {
        // Second half: backward output
        output[outputOffset] = backward[sourceOffset + (hiddenIdx - hiddenSize)];
    }
}

// MARK: - Utility Kernels

/// Copy data from one buffer to another.
///
/// Simple memcpy-like kernel for GPU buffer operations.
kernel void buffer_copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    output[gid] = input[gid];
}

/// Zero-fill a buffer.
///
/// Initialize buffer with zeros.
kernel void buffer_zero(
    device float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    buffer[gid] = 0.0f;
}

/// Extract a slice from a sequence at a specific timestep.
///
/// For processing one timestep at a time from a batched sequence.
/// Input: [batchSize, seqLen, featureSize]
/// Output: [batchSize, featureSize]
///
/// Parameters:
/// - input: Input sequence
/// - output: Output slice for single timestep
/// - batchSize: Batch dimension
/// - seqLen: Sequence length
/// - featureSize: Feature dimension
/// - timestep: Which timestep to extract
kernel void extract_timestep(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batchSize [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& featureSize [[buffer(4)]],
    constant uint& timestep [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint featIdx = gid.x;
    uint batchIdx = gid.y;

    if (featIdx >= featureSize || batchIdx >= batchSize) {
        return;
    }

    uint inputOffset = batchIdx * seqLen * featureSize +
                       timestep * featureSize +
                       featIdx;

    uint outputOffset = batchIdx * featureSize + featIdx;

    output[outputOffset] = input[inputOffset];
}

/// Store output for a specific timestep into a sequence buffer.
///
/// For collecting outputs during sequence processing.
/// Input: [batchSize, featureSize]
/// Output: [batchSize, seqLen, featureSize]
///
/// Parameters:
/// - input: Input slice for single timestep
/// - output: Output sequence buffer
/// - batchSize: Batch dimension
/// - seqLen: Sequence length
/// - featureSize: Feature dimension
/// - timestep: Which timestep to store
kernel void store_timestep(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batchSize [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& featureSize [[buffer(4)]],
    constant uint& timestep [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint featIdx = gid.x;
    uint batchIdx = gid.y;

    if (featIdx >= featureSize || batchIdx >= batchSize) {
        return;
    }

    uint inputOffset = batchIdx * featureSize + featIdx;

    uint outputOffset = batchIdx * seqLen * featureSize +
                        timestep * featureSize +
                        featIdx;

    output[outputOffset] = input[inputOffset];
}
