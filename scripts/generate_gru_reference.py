#!/usr/bin/env python3
"""
Generate GRU reference data for validating SwiftRosaNN GRU implementation.

This script creates:
1. Known weights for a GRU layer
2. Known input sequence
3. Expected outputs from PyTorch GRU
4. Exports everything to binary files for Swift tests

Usage:
    python scripts/generate_gru_reference.py

Output directory: Tests/SwiftRosaNNTests/ReferenceData/gru/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

# Configuration
OUTPUT_DIR = "Tests/SwiftRosaNNTests/ReferenceData/gru"
SEED = 42

# Test configurations
CONFIGS = [
    # Simple unidirectional GRU
    {
        "name": "simple",
        "input_size": 4,
        "hidden_size": 8,
        "batch_size": 1,
        "seq_len": 5,
        "bidirectional": False,
    },
    # Bidirectional GRU
    {
        "name": "bidirectional",
        "input_size": 4,
        "hidden_size": 8,
        "batch_size": 1,
        "seq_len": 5,
        "bidirectional": True,
    },
    # Banquet-like config (smaller for testing)
    {
        "name": "banquet_small",
        "input_size": 16,
        "hidden_size": 32,
        "batch_size": 1,
        "seq_len": 10,
        "bidirectional": True,
    },
]


def generate_reference_data(config):
    """Generate reference data for a single GRU configuration."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    name = config["name"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    bidirectional = config["bidirectional"]

    print(f"\nGenerating reference data for '{name}'...")
    print(f"  input_size={input_size}, hidden_size={hidden_size}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  bidirectional={bidirectional}")

    # Create output directory
    config_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(config_dir, exist_ok=True)

    # Create GRU layer
    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=bidirectional,
        bias=True,
    )

    # Initialize with small random weights for numerical stability
    with torch.no_grad():
        for param in gru.parameters():
            param.data = torch.randn_like(param) * 0.1

    # Create input sequence
    input_tensor = torch.randn(batch_size, seq_len, input_size) * 0.5

    # Create initial hidden state (zeros)
    num_directions = 2 if bidirectional else 1
    h0 = torch.zeros(num_directions, batch_size, hidden_size)

    # Run forward pass
    gru.eval()
    with torch.no_grad():
        output, hn = gru(input_tensor, h0)

    # Extract weights
    # PyTorch GRU weight order: reset, update, new (r, z, n)
    weight_ih = gru.weight_ih_l0.data.numpy().astype(np.float32)
    weight_hh = gru.weight_hh_l0.data.numpy().astype(np.float32)
    bias_ih = gru.bias_ih_l0.data.numpy().astype(np.float32)
    bias_hh = gru.bias_hh_l0.data.numpy().astype(np.float32)

    print(f"  weight_ih shape: {weight_ih.shape}")
    print(f"  weight_hh shape: {weight_hh.shape}")
    print(f"  bias_ih shape: {bias_ih.shape}")
    print(f"  bias_hh shape: {bias_hh.shape}")

    # Save forward weights
    weight_ih.tofile(os.path.join(config_dir, "weight_ih_l0.bin"))
    weight_hh.tofile(os.path.join(config_dir, "weight_hh_l0.bin"))
    bias_ih.tofile(os.path.join(config_dir, "bias_ih_l0.bin"))
    bias_hh.tofile(os.path.join(config_dir, "bias_hh_l0.bin"))

    # Save backward weights if bidirectional
    if bidirectional:
        weight_ih_reverse = gru.weight_ih_l0_reverse.data.numpy().astype(np.float32)
        weight_hh_reverse = gru.weight_hh_l0_reverse.data.numpy().astype(np.float32)
        bias_ih_reverse = gru.bias_ih_l0_reverse.data.numpy().astype(np.float32)
        bias_hh_reverse = gru.bias_hh_l0_reverse.data.numpy().astype(np.float32)

        weight_ih_reverse.tofile(os.path.join(config_dir, "weight_ih_l0_reverse.bin"))
        weight_hh_reverse.tofile(os.path.join(config_dir, "weight_hh_l0_reverse.bin"))
        bias_ih_reverse.tofile(os.path.join(config_dir, "bias_ih_l0_reverse.bin"))
        bias_hh_reverse.tofile(os.path.join(config_dir, "bias_hh_l0_reverse.bin"))

    # Save input
    input_np = input_tensor.numpy().astype(np.float32)
    input_np.tofile(os.path.join(config_dir, "input.bin"))
    print(f"  input shape: {input_np.shape}")

    # Save expected output
    output_np = output.numpy().astype(np.float32)
    output_np.tofile(os.path.join(config_dir, "expected_output.bin"))
    print(f"  output shape: {output_np.shape}")

    # Save final hidden state
    hn_np = hn.numpy().astype(np.float32)
    hn_np.tofile(os.path.join(config_dir, "expected_hidden.bin"))
    print(f"  final hidden shape: {hn_np.shape}")

    # Save metadata
    metadata = {
        "name": name,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "bidirectional": bidirectional,
        "num_directions": num_directions,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "weight_ih_shape": list(weight_ih.shape),
        "weight_hh_shape": list(weight_hh.shape),
        "input_shape": list(input_np.shape),
        "output_shape": list(output_np.shape),
        "hidden_shape": list(hn_np.shape),
    }

    with open(os.path.join(config_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Also print some actual values for debugging
    print(f"\n  Sample values:")
    print(f"    input[0,0,:4]: {input_np[0, 0, :min(4, input_size)]}")
    print(f"    output[0,0,:4]: {output_np[0, 0, :min(4, hidden_size * num_directions)]}")
    print(f"    output[0,-1,:4]: {output_np[0, -1, :min(4, hidden_size * num_directions)]}")

    return metadata


def generate_single_step_reference():
    """Generate reference data for single GRU cell step validation."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("\nGenerating single-step GRU cell reference...")

    input_size = 3
    hidden_size = 4
    batch_size = 1

    config_dir = os.path.join(OUTPUT_DIR, "single_step")
    os.makedirs(config_dir, exist_ok=True)

    # Create GRU cell
    gru_cell = nn.GRUCell(input_size, hidden_size, bias=True)

    # Initialize with known small weights
    with torch.no_grad():
        gru_cell.weight_ih.data = torch.randn(3 * hidden_size, input_size) * 0.1
        gru_cell.weight_hh.data = torch.randn(3 * hidden_size, hidden_size) * 0.1
        gru_cell.bias_ih.data = torch.randn(3 * hidden_size) * 0.1
        gru_cell.bias_hh.data = torch.randn(3 * hidden_size) * 0.1

    # Create input and initial hidden state
    input_tensor = torch.randn(batch_size, input_size) * 0.5
    h0 = torch.randn(batch_size, hidden_size) * 0.1

    # Run forward pass
    gru_cell.eval()
    with torch.no_grad():
        h1 = gru_cell(input_tensor, h0)

    # Also compute intermediate values for debugging
    # PyTorch GRU equations:
    # r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    # z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    # n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    # h' = (1 - z) * n + z * h

    with torch.no_grad():
        W_ih = gru_cell.weight_ih.data
        W_hh = gru_cell.weight_hh.data
        b_ih = gru_cell.bias_ih.data
        b_hh = gru_cell.bias_hh.data

        H = hidden_size

        # Split weights by gate
        W_ir, W_iz, W_in = W_ih[:H], W_ih[H:2*H], W_ih[2*H:3*H]
        W_hr, W_hz, W_hn = W_hh[:H], W_hh[H:2*H], W_hh[2*H:3*H]
        b_ir, b_iz, b_in = b_ih[:H], b_ih[H:2*H], b_ih[2*H:3*H]
        b_hr, b_hz, b_hn = b_hh[:H], b_hh[H:2*H], b_hh[2*H:3*H]

        x = input_tensor
        h = h0

        # Compute gates
        r = torch.sigmoid(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
        z = torch.sigmoid(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
        n = torch.tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))
        h_new = (1 - z) * n + z * h

        print(f"  Intermediate gate values:")
        print(f"    r (reset): {r.numpy().flatten()}")
        print(f"    z (update): {z.numpy().flatten()}")
        print(f"    n (new): {n.numpy().flatten()}")
        print(f"    h_new: {h_new.numpy().flatten()}")
        print(f"    h1 (from cell): {h1.numpy().flatten()}")
        print(f"    Match: {torch.allclose(h_new, h1)}")

    # Save weights
    gru_cell.weight_ih.data.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "weight_ih.bin")
    )
    gru_cell.weight_hh.data.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "weight_hh.bin")
    )
    gru_cell.bias_ih.data.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "bias_ih.bin")
    )
    gru_cell.bias_hh.data.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "bias_hh.bin")
    )

    # Save input, initial state, and expected output
    input_tensor.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "input.bin")
    )
    h0.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "h0.bin")
    )
    h1.numpy().astype(np.float32).tofile(
        os.path.join(config_dir, "expected_h1.bin")
    )

    # Save intermediate values for debugging
    r.numpy().astype(np.float32).tofile(os.path.join(config_dir, "expected_r.bin"))
    z.numpy().astype(np.float32).tofile(os.path.join(config_dir, "expected_z.bin"))
    n.numpy().astype(np.float32).tofile(os.path.join(config_dir, "expected_n.bin"))

    # Save metadata
    metadata = {
        "name": "single_step",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "pytorch_version": torch.__version__,
        "seed": SEED,
    }

    with open(os.path.join(config_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to {config_dir}")


def main():
    print("=" * 60)
    print("GRU Reference Data Generator")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Output directory: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate single-step reference first (most important for debugging)
    generate_single_step_reference()

    # Generate reference data for each configuration
    all_metadata = []
    for config in CONFIGS:
        metadata = generate_reference_data(config)
        all_metadata.append(metadata)

    # Save combined metadata
    with open(os.path.join(OUTPUT_DIR, "all_configs.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Reference data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
