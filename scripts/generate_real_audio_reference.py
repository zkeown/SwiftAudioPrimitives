#!/usr/bin/env python3
"""
Generate librosa reference data from MUSDB18-HQ for SwiftRosa validation.

This script processes selected tracks from MUSDB18-HQ and generates reference
outputs for all major librosa functions, enabling validation against real audio.

Usage:
    python generate_real_audio_reference.py --musdb-path /path/to/musdb18hq

Requirements:
    pip install librosa musdb numpy soundfile
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

try:
    import musdb
except ImportError:
    print("musdb not installed. Install with: pip install musdb")
    sys.exit(1)

# Track selection for diverse coverage
# Format: (track_name_substring, stem, duration_seconds, start_seconds, purpose)
TRACK_SELECTIONS = [
    # Full mixes - diverse genres
    ("Actions - One Minute Smile", "mixture", 30, 30, "rock_drums"),
    ("Alexander Ross - Goodbye Bolero", "mixture", 30, 60, "orchestral"),
    ("Aimee Norwich - Child", "mixture", 30, 30, "acoustic_folk"),
    ("Angels In Amplifiers - I'm Alright", "mixture", 30, 45, "rock_guitar"),
    ("ANiMAL - Rockshow", "mixture", 30, 30, "electronic_rock"),
    ("Arise - Run Run Run", "mixture", 30, 30, "pop_vocals"),
    ("BKS - Bulldozer", "mixture", 30, 30, "electronic_bass"),
    ("BKS - Too Much", "mixture", 30, 45, "electronic_synth"),
    ("Clara Berry And Wooldog - Waltz For My Victims", "mixture", 30, 30, "indie_sparse"),
    ("Cristina Vane - So Easy", "mixture", 30, 30, "jazz_vocals"),

    # Isolated stems for targeted testing
    ("Actions - One Minute Smile", "drums", 30, 30, "drums_isolated"),
    ("Arise - Run Run Run", "vocals", 30, 30, "vocals_isolated"),
    ("Alexander Ross - Goodbye Bolero", "other", 30, 60, "orchestral_no_drums"),
    ("BKS - Bulldozer", "bass", 30, 30, "bass_isolated"),
]

# Features to compute
FEATURES = [
    "stft",
    "mel_spectrogram",
    "mfcc",
    "cqt",
    "chroma_stft",
    "chroma_cqt",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "rms",
    "zero_crossing_rate",
    "onset_strength",
    "onset_detect",
    "tempo",
]

# Standard parameters matching SwiftRosa defaults
PARAMS = {
    "sr": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "n_mfcc": 13,
    "fmin": 32.70,  # C1
    "n_bins": 84,   # 7 octaves
    "bins_per_octave": 12,
}


def find_track(mus, name_substring: str):
    """Find a track by name substring."""
    for track in mus:
        if name_substring.lower() in track.name.lower():
            return track
    return None


def extract_clip(track, stem: str, duration: float, start: float, sr: int) -> np.ndarray:
    """Extract a clip from a track stem."""
    if stem == "mixture":
        audio = track.audio
    else:
        audio = track.sources[stem].audio

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    original_sr = track.rate
    if original_sr != sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)

    # Extract clip
    start_sample = int(start * sr)
    end_sample = int((start + duration) * sr)

    if end_sample > len(audio):
        end_sample = len(audio)
        start_sample = max(0, end_sample - int(duration * sr))

    return audio[start_sample:end_sample].astype(np.float32)


def compute_features(audio: np.ndarray, sr: int, params: dict) -> dict:
    """Compute all features for an audio clip."""
    results = {}

    # STFT
    stft = librosa.stft(audio, n_fft=params["n_fft"], hop_length=params["hop_length"])
    stft_mag = np.abs(stft)
    results["stft"] = {
        "magnitude": stft_mag.tolist(),
        "shape": list(stft_mag.shape),
        "n_fft": params["n_fft"],
        "hop_length": params["hop_length"],
    }

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"]
    )
    results["mel_spectrogram"] = {
        "values": mel.tolist(),
        "shape": list(mel.shape),
        "n_mels": params["n_mels"],
    }

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=params["n_mfcc"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["mfcc"] = {
        "values": mfcc.tolist(),
        "shape": list(mfcc.shape),
        "n_mfcc": params["n_mfcc"],
    }

    # CQT
    cqt = librosa.cqt(
        audio, sr=sr,
        hop_length=params["hop_length"],
        fmin=params["fmin"],
        n_bins=params["n_bins"],
        bins_per_octave=params["bins_per_octave"]
    )
    cqt_mag = np.abs(cqt)
    results["cqt"] = {
        "magnitude": cqt_mag.tolist(),
        "shape": list(cqt_mag.shape),
        "fmin": params["fmin"],
        "n_bins": params["n_bins"],
        "bins_per_octave": params["bins_per_octave"],
    }

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(
        y=audio, sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["chroma_stft"] = {
        "values": chroma_stft.tolist(),
        "shape": list(chroma_stft.shape),
    }

    # Chroma CQT
    chroma_cqt = librosa.feature.chroma_cqt(
        y=audio, sr=sr,
        hop_length=params["hop_length"]
    )
    results["chroma_cqt"] = {
        "values": chroma_cqt.tolist(),
        "shape": list(chroma_cqt.shape),
    }

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["spectral_centroid"] = {
        "values": centroid[0].tolist(),
        "shape": list(centroid.shape),
    }

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["spectral_bandwidth"] = {
        "values": bandwidth[0].tolist(),
        "shape": list(bandwidth.shape),
    }

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["spectral_rolloff"] = {
        "values": rolloff[0].tolist(),
        "shape": list(rolloff.shape),
    }

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["spectral_flatness"] = {
        "values": flatness[0].tolist(),
        "shape": list(flatness.shape),
    }

    # RMS
    rms = librosa.feature.rms(
        y=audio,
        frame_length=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["rms"] = {
        "values": rms[0].tolist(),
        "shape": list(rms.shape),
    }

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=params["n_fft"],
        hop_length=params["hop_length"]
    )
    results["zero_crossing_rate"] = {
        "values": zcr[0].tolist(),
        "shape": list(zcr.shape),
    }

    # Onset strength
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr,
        hop_length=params["hop_length"]
    )
    results["onset_strength"] = {
        "values": onset_env.tolist(),
        "length": len(onset_env),
    }

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr,
        hop_length=params["hop_length"]
    )
    results["onset_detect"] = {
        "frames": onset_frames.tolist(),
        "count": len(onset_frames),
    }

    # Tempo - use librosa.feature.tempo() to match BeatTracker.estimateTempo()
    # Note: librosa.beat.beat_track returns tempo derived from beat intervals (dynamic programming)
    # while librosa.feature.tempo uses pure tempogram analysis
    tempo = librosa.feature.tempo(
        y=audio, sr=sr,
        hop_length=params["hop_length"]
    )
    # Handle both old and new librosa versions
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    results["tempo"] = {
        "bpm": tempo,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate real audio reference data")
    parser.add_argument("--musdb-path", required=True, help="Path to MUSDB18-HQ dataset")
    parser.add_argument("--output", default="Tests/RealAudioValidation/ReferenceData/musdb18_reference.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Initialize MUSDB - load both subsets
    print(f"Loading MUSDB18-HQ from {args.musdb_path}")
    mus_train = musdb.DB(root=args.musdb_path, subsets="train", is_wav=True)
    mus_test = musdb.DB(root=args.musdb_path, subsets="test", is_wav=True)
    mus = list(mus_train) + list(mus_test)

    print(f"Found {len(mus)} tracks ({len(mus_train)} train + {len(mus_test)} test)")

    # Process each selection
    all_results = {
        "metadata": {
            "librosa_version": librosa.__version__,
            "sample_rate": PARAMS["sr"],
            "parameters": PARAMS,
            "track_count": len(TRACK_SELECTIONS),
        },
        "tracks": {}
    }

    for track_name, stem, duration, start, purpose in TRACK_SELECTIONS:
        print(f"\nProcessing: {track_name} ({stem}) - {purpose}")

        track = find_track(mus, track_name)
        if track is None:
            print(f"  WARNING: Track not found, skipping")
            continue

        try:
            # Extract audio clip
            audio = extract_clip(track, stem, duration, start, PARAMS["sr"])
            print(f"  Extracted {len(audio)} samples ({len(audio)/PARAMS['sr']:.1f}s)")

            # Compute features
            features = compute_features(audio, PARAMS["sr"], PARAMS)

            # Store results
            clip_id = f"{purpose}"
            all_results["tracks"][clip_id] = {
                "source_track": track.name,
                "stem": stem,
                "duration": duration,
                "start": start,
                "purpose": purpose,
                "audio_samples": len(audio),
                "audio": audio.tolist(),  # Store raw audio for Swift to load
                "features": features,
            }

            print(f"  Computed {len(features)} feature sets")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_results, f)

    # Print size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output size: {size_mb:.1f} MB")
    print(f"Processed {len(all_results['tracks'])} clips")


if __name__ == "__main__":
    main()
