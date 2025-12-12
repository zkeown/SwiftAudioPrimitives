#!/usr/bin/env python3
"""
Benchmark librosa operations for performance comparison with SwiftRosa.

This script times all major librosa operations and outputs JSON for
comparison against the Swift implementation.

Usage:
    python benchmark_librosa.py

Output:
    Tests/SwiftRosaBenchmarks/ReferenceTimings/librosa_timings.json

Requirements:
    pip install librosa numpy scipy
"""

import json
import time
import platform
import subprocess
import numpy as np
import librosa
import scipy.signal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Callable, Any


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    operation: str
    parameters: dict
    signal_length_seconds: float
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples_per_second: float
    real_time_factor: float  # > 1 means faster than real-time


def get_hardware_info() -> dict:
    """Get hardware information for the benchmark."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }

    # Try to get chip info on macOS
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["cpu_brand"] = result.stdout.strip()
    except:
        pass

    # Try to get Mac model
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["mac_model"] = result.stdout.strip()
    except:
        pass

    return info


def benchmark_operation(
    name: str,
    operation: Callable,
    iterations: int = 10,
    warmup: int = 2,
    signal_length_seconds: float = 1.0,
    sample_rate: int = 22050,
    parameters: dict = None
) -> BenchmarkResult:
    """Benchmark a single operation."""
    parameters = parameters or {}

    # Warmup runs
    for _ in range(warmup):
        operation()

    # Timed runs
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms

    timings = np.array(timings)
    mean_ms = float(np.mean(timings))

    # Calculate throughput
    total_samples = int(signal_length_seconds * sample_rate)
    processing_time_seconds = mean_ms / 1000
    samples_per_second = total_samples / processing_time_seconds if processing_time_seconds > 0 else 0
    real_time_factor = signal_length_seconds / processing_time_seconds if processing_time_seconds > 0 else 0

    return BenchmarkResult(
        operation=name,
        parameters=parameters,
        signal_length_seconds=signal_length_seconds,
        iterations=iterations,
        mean_ms=mean_ms,
        std_ms=float(np.std(timings)),
        min_ms=float(np.min(timings)),
        max_ms=float(np.max(timings)),
        samples_per_second=samples_per_second,
        real_time_factor=real_time_factor
    )


def generate_test_signal(duration: float, sr: int = 22050) -> np.ndarray:
    """Generate a multi-tone test signal."""
    t = np.arange(int(sr * duration)) / sr
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.3 * np.sin(2 * np.pi * 880 * t) +
              0.2 * np.sin(2 * np.pi * 1760 * t))
    return signal.astype(np.float32)


def benchmark_stft(signals: dict, sr: int) -> list:
    """Benchmark STFT operations."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        for n_fft in [1024, 2048, 4096]:
            hop_length = n_fft // 4

            def stft_op():
                return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

            result = benchmark_operation(
                name=f"stft_{duration_name}_nfft{n_fft}",
                operation=stft_op,
                signal_length_seconds=duration,
                sample_rate=sr,
                parameters={"n_fft": n_fft, "hop_length": hop_length}
            )
            results.append(result)
            print(f"  STFT {duration_name} nFFT={n_fft}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_istft(signals: dict, sr: int) -> list:
    """Benchmark ISTFT operations."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        n_fft = 2048
        hop_length = 512

        # Pre-compute STFT
        S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

        def istft_op():
            return librosa.istft(S, hop_length=hop_length, length=len(signal))

        result = benchmark_operation(
            name=f"istft_{duration_name}",
            operation=istft_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  ISTFT {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_mel_spectrogram(signals: dict, sr: int) -> list:
    """Benchmark Mel spectrogram operations."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        for n_mels in [40, 64, 128]:
            n_fft = 2048
            hop_length = 512

            def mel_op():
                return librosa.feature.melspectrogram(
                    y=signal, sr=sr, n_fft=n_fft,
                    hop_length=hop_length, n_mels=n_mels
                )

            result = benchmark_operation(
                name=f"mel_{duration_name}_nmels{n_mels}",
                operation=mel_op,
                signal_length_seconds=duration,
                sample_rate=sr,
                parameters={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels}
            )
            results.append(result)
            print(f"  Mel {duration_name} nMels={n_mels}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_mfcc(signals: dict, sr: int) -> list:
    """Benchmark MFCC operations."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        for n_mfcc in [13, 20, 40]:
            n_fft = 2048
            hop_length = 512
            n_mels = 128

            def mfcc_op():
                return librosa.feature.mfcc(
                    y=signal, sr=sr, n_mfcc=n_mfcc,
                    n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                )

            result = benchmark_operation(
                name=f"mfcc_{duration_name}_nmfcc{n_mfcc}",
                operation=mfcc_op,
                signal_length_seconds=duration,
                sample_rate=sr,
                parameters={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "n_mfcc": n_mfcc}
            )
            results.append(result)
            print(f"  MFCC {duration_name} nMFCC={n_mfcc}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_cqt(signals: dict, sr: int) -> list:
    """Benchmark CQT operations."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        hop_length = 512
        n_bins = 84
        bins_per_octave = 12

        def cqt_op():
            return librosa.cqt(
                signal, sr=sr, hop_length=hop_length,
                n_bins=n_bins, bins_per_octave=bins_per_octave
            )

        result = benchmark_operation(
            name=f"cqt_{duration_name}",
            operation=cqt_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"hop_length": hop_length, "n_bins": n_bins, "bins_per_octave": bins_per_octave}
        )
        results.append(result)
        print(f"  CQT {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_spectral_features(signals: dict, sr: int) -> list:
    """Benchmark spectral feature extraction."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        n_fft = 2048
        hop_length = 512

        # Pre-compute magnitude spectrogram
        S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

        # Spectral centroid
        def centroid_op():
            return librosa.feature.spectral_centroid(S=S, sr=sr)

        result = benchmark_operation(
            name=f"spectral_centroid_{duration_name}",
            operation=centroid_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Spectral Centroid {duration_name}: {result.mean_ms:.2f}ms")

        # Spectral bandwidth
        def bandwidth_op():
            return librosa.feature.spectral_bandwidth(S=S, sr=sr)

        result = benchmark_operation(
            name=f"spectral_bandwidth_{duration_name}",
            operation=bandwidth_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Spectral Bandwidth {duration_name}: {result.mean_ms:.2f}ms")

        # Spectral rolloff
        def rolloff_op():
            return librosa.feature.spectral_rolloff(S=S, sr=sr)

        result = benchmark_operation(
            name=f"spectral_rolloff_{duration_name}",
            operation=rolloff_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Spectral Rolloff {duration_name}: {result.mean_ms:.2f}ms")

        # Spectral flatness
        def flatness_op():
            return librosa.feature.spectral_flatness(S=S)

        result = benchmark_operation(
            name=f"spectral_flatness_{duration_name}",
            operation=flatness_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Spectral Flatness {duration_name}: {result.mean_ms:.2f}ms")

        # Spectral contrast
        def contrast_op():
            return librosa.feature.spectral_contrast(S=S, sr=sr)

        result = benchmark_operation(
            name=f"spectral_contrast_{duration_name}",
            operation=contrast_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Spectral Contrast {duration_name}: {result.mean_ms:.2f}ms")

    return results


def benchmark_onset(signals: dict, sr: int) -> list:
    """Benchmark onset detection."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        hop_length = 512

        # Onset strength
        def onset_strength_op():
            return librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)

        result = benchmark_operation(
            name=f"onset_strength_{duration_name}",
            operation=onset_strength_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"hop_length": hop_length}
        )
        results.append(result)
        print(f"  Onset Strength {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

        # Onset detect
        def onset_detect_op():
            return librosa.onset.onset_detect(y=signal, sr=sr, hop_length=hop_length)

        result = benchmark_operation(
            name=f"onset_detect_{duration_name}",
            operation=onset_detect_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"hop_length": hop_length}
        )
        results.append(result)
        print(f"  Onset Detect {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_resample(signals: dict, sr: int) -> list:
    """Benchmark resampling operations."""
    results = []

    resample_targets = [
        (22050, 16000),
        (22050, 44100),
        (22050, 8000),
    ]

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        for orig_sr, target_sr in resample_targets:
            def resample_op():
                return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)

            result = benchmark_operation(
                name=f"resample_{duration_name}_{orig_sr}to{target_sr}",
                operation=resample_op,
                signal_length_seconds=duration,
                sample_rate=sr,
                parameters={"orig_sr": orig_sr, "target_sr": target_sr}
            )
            results.append(result)
            print(f"  Resample {duration_name} {orig_sr}->{target_sr}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_hpss(signals: dict, sr: int) -> list:
    """Benchmark HPSS decomposition."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        def hpss_op():
            return librosa.effects.hpss(signal)

        result = benchmark_operation(
            name=f"hpss_{duration_name}",
            operation=hpss_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={}
        )
        results.append(result)
        print(f"  HPSS {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_chromagram(signals: dict, sr: int) -> list:
    """Benchmark chromagram extraction."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        n_fft = 2048
        hop_length = 512

        # STFT-based chroma
        def chroma_stft_op():
            return librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)

        result = benchmark_operation(
            name=f"chroma_stft_{duration_name}",
            operation=chroma_stft_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  Chroma STFT {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

        # CQT-based chroma
        def chroma_cqt_op():
            return librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=hop_length)

        result = benchmark_operation(
            name=f"chroma_cqt_{duration_name}",
            operation=chroma_cqt_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"hop_length": hop_length}
        )
        results.append(result)
        print(f"  Chroma CQT {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_rms(signals: dict, sr: int) -> list:
    """Benchmark RMS energy extraction."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        frame_length = 2048
        hop_length = 512

        def rms_op():
            return librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)

        result = benchmark_operation(
            name=f"rms_{duration_name}",
            operation=rms_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"frame_length": frame_length, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  RMS {duration_name}: {result.mean_ms:.2f}ms")

    return results


def benchmark_zcr(signals: dict, sr: int) -> list:
    """Benchmark zero crossing rate."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        frame_length = 2048
        hop_length = 512

        def zcr_op():
            return librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length)

        result = benchmark_operation(
            name=f"zcr_{duration_name}",
            operation=zcr_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"frame_length": frame_length, "hop_length": hop_length}
        )
        results.append(result)
        print(f"  ZCR {duration_name}: {result.mean_ms:.2f}ms")

    return results


def benchmark_delta(signals: dict, sr: int) -> list:
    """Benchmark delta feature computation."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        # Pre-compute MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

        for width in [3, 5, 9]:
            def delta_op():
                return librosa.feature.delta(mfccs, width=width)

            result = benchmark_operation(
                name=f"delta_{duration_name}_width{width}",
                operation=delta_op,
                signal_length_seconds=duration,
                sample_rate=sr,
                parameters={"width": width}
            )
            results.append(result)
            print(f"  Delta {duration_name} width={width}: {result.mean_ms:.2f}ms")

    return results


def benchmark_tonnetz(signals: dict, sr: int) -> list:
    """Benchmark tonnetz feature extraction."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr

        def tonnetz_op():
            return librosa.feature.tonnetz(y=signal, sr=sr)

        result = benchmark_operation(
            name=f"tonnetz_{duration_name}",
            operation=tonnetz_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={}
        )
        results.append(result)
        print(f"  Tonnetz {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_tempogram(signals: dict, sr: int) -> list:
    """Benchmark tempogram computation."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        hop_length = 512

        # Pre-compute onset envelope
        onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)

        def tempogram_op():
            return librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

        result = benchmark_operation(
            name=f"tempogram_{duration_name}",
            operation=tempogram_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"hop_length": hop_length}
        )
        results.append(result)
        print(f"  Tempogram {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def benchmark_pcen(signals: dict, sr: int) -> list:
    """Benchmark PCEN computation."""
    results = []

    for duration_name, signal in signals.items():
        duration = len(signal) / sr
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        # Pre-compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        def pcen_op():
            return librosa.pcen(mel_spec * (2**31))

        result = benchmark_operation(
            name=f"pcen_{duration_name}",
            operation=pcen_op,
            signal_length_seconds=duration,
            sample_rate=sr,
            parameters={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels}
        )
        results.append(result)
        print(f"  PCEN {duration_name}: {result.mean_ms:.2f}ms ({result.real_time_factor:.1f}x RT)")

    return results


def main():
    print("=" * 70)
    print("SwiftRosa Performance Baseline: librosa Benchmarks")
    print("=" * 70)

    # Get hardware info
    hardware = get_hardware_info()
    print(f"\nHardware: {hardware.get('cpu_brand', hardware.get('processor', 'Unknown'))}")
    print(f"librosa version: {librosa.__version__}")
    print(f"numpy version: {np.__version__}")
    print()

    sr = 22050

    # Generate test signals
    print("Generating test signals...")
    signals = {
        "1s": generate_test_signal(1.0, sr),
        "10s": generate_test_signal(10.0, sr),
        "60s": generate_test_signal(60.0, sr),
    }
    print(f"  1s: {len(signals['1s']):,} samples")
    print(f"  10s: {len(signals['10s']):,} samples")
    print(f"  60s: {len(signals['60s']):,} samples")
    print()

    all_results = []

    # Run benchmarks
    print("=" * 70)
    print("STFT Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_stft(signals, sr))

    print("\n" + "=" * 70)
    print("ISTFT Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_istft(signals, sr))

    print("\n" + "=" * 70)
    print("Mel Spectrogram Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_mel_spectrogram(signals, sr))

    print("\n" + "=" * 70)
    print("MFCC Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_mfcc(signals, sr))

    print("\n" + "=" * 70)
    print("CQT Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_cqt(signals, sr))

    print("\n" + "=" * 70)
    print("Spectral Features Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_spectral_features(signals, sr))

    print("\n" + "=" * 70)
    print("Onset Detection Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_onset(signals, sr))

    print("\n" + "=" * 70)
    print("Resampling Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_resample(signals, sr))

    print("\n" + "=" * 70)
    print("HPSS Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_hpss(signals, sr))

    print("\n" + "=" * 70)
    print("Chromagram Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_chromagram(signals, sr))

    print("\n" + "=" * 70)
    print("RMS Energy Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_rms(signals, sr))

    print("\n" + "=" * 70)
    print("Zero Crossing Rate Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_zcr(signals, sr))

    print("\n" + "=" * 70)
    print("Delta Features Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_delta(signals, sr))

    print("\n" + "=" * 70)
    print("Tonnetz Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_tonnetz(signals, sr))

    print("\n" + "=" * 70)
    print("Tempogram Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_tempogram(signals, sr))

    print("\n" + "=" * 70)
    print("PCEN Benchmarks")
    print("=" * 70)
    all_results.extend(benchmark_pcen(signals, sr))

    # Prepare output
    output_data = {
        "metadata": {
            "generator": "benchmark_librosa.py",
            "generated_at": datetime.now().isoformat(),
            "librosa_version": librosa.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "sample_rate": sr,
            "hardware": hardware
        },
        "benchmarks": {r.operation: asdict(r) for r in all_results}
    }

    # Save to file
    output_dir = Path(__file__).parent.parent / "Tests" / "SwiftRosaBenchmarks" / "ReferenceTimings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "librosa_timings.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total benchmarks: {len(all_results)}")
    print(f"Output saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print summary table
    print("\n" + "=" * 70)
    print("Quick Reference (10s signal, default params)")
    print("=" * 70)
    print(f"{'Operation':<30} {'Time (ms)':<12} {'RT Factor':<12}")
    print("-" * 54)

    key_ops = [
        "stft_10s_nfft2048",
        "istft_10s",
        "mel_10s_nmels128",
        "mfcc_10s_nmfcc13",
        "cqt_10s",
        "onset_strength_10s",
        "resample_10s_22050to16000",
        "hpss_10s",
        "chroma_stft_10s",
    ]

    for op_name in key_ops:
        if op_name in output_data["benchmarks"]:
            r = output_data["benchmarks"][op_name]
            print(f"{op_name:<30} {r['mean_ms']:<12.2f} {r['real_time_factor']:<12.1f}x")


if __name__ == "__main__":
    main()
