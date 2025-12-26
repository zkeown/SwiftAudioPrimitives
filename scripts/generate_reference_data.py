#!/usr/bin/env python3
"""
Generate librosa reference data for SwiftRosa validation.

This script generates comprehensive ground truth values from librosa for comparison
against the Swift implementation. Output is JSON format for easy parsing.

Usage:
    python generate_reference_data.py

Requirements:
    pip install librosa numpy scipy
"""

import json
import numpy as np
import librosa
import scipy.signal
from pathlib import Path
from datetime import datetime


# =============================================================================
# TEST SIGNAL GENERATION
# =============================================================================

def generate_test_signals():
    """Generate comprehensive test signals for validation."""
    signals = {}
    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr

    # Pure tones at multiple frequencies
    for freq in [110, 220, 440, 880, 1760]:
        signals[f'sine_{freq}hz'] = {
            'data': np.sin(2 * np.pi * freq * t).astype(np.float32).tolist(),
            'sample_rate': sr,
            'description': f'Pure {freq} Hz sine wave'
        }

    # Multi-tone signal (harmonic series)
    signals['multi_tone'] = {
        'data': (0.5 * np.sin(2 * np.pi * 440 * t) +
                 0.3 * np.sin(2 * np.pi * 880 * t) +
                 0.2 * np.sin(2 * np.pi * 1760 * t)).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': '440Hz + 880Hz + 1760Hz harmonics'
    }

    # White noise (seeded for reproducibility)
    np.random.seed(42)
    signals['white_noise'] = {
        'data': (np.random.randn(int(sr * duration)) * 0.1).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': 'White noise (seed=42, amplitude=0.1)'
    }

    # Frequency sweep (chirp) - uses same formula as Swift's TestSignalGenerator.chirp()
    # Swift formula: instantFreq = fmin + rate * t / 2; sin(2*pi * instantFreq * t)
    # This is a linear frequency sweep (quadratic phase)
    fmin_chirp = 100
    fmax_chirp = 8000
    rate = (fmax_chirp - fmin_chirp) / duration
    instant_freq = fmin_chirp + rate * t / 2
    chirp_signal = np.sin(2 * np.pi * instant_freq * t).astype(np.float32)
    signals['chirp'] = {
        'data': chirp_signal.tolist(),
        'sample_rate': sr,
        'description': 'Linear chirp 100Hz to 8000Hz (Swift-compatible formula)'
    }

    # Click train (impulses at regular intervals)
    click_signal = np.zeros(int(sr * duration), dtype=np.float32)
    click_positions = np.arange(0, int(sr * duration), sr // 4)  # 4 clicks per second
    click_signal[click_positions] = 1.0
    signals['click_train'] = {
        'data': click_signal.tolist(),
        'sample_rate': sr,
        'description': 'Click train with 4 clicks per second'
    }

    # AM-modulated signal
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5 Hz modulation
    signals['am_modulated'] = {
        'data': (carrier * modulator).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': '1000Hz carrier with 5Hz AM modulation'
    }

    # Edge case signals
    signals['silence'] = {
        'data': np.zeros(int(sr * duration), dtype=np.float32).tolist(),
        'sample_rate': sr,
        'description': 'Silence (all zeros)'
    }

    signals['dc_offset'] = {
        'data': np.full(int(sr * duration), 0.5, dtype=np.float32).tolist(),
        'sample_rate': sr,
        'description': 'DC offset at 0.5'
    }

    signals['impulse'] = {
        'data': np.concatenate([
            np.zeros(sr // 2, dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.zeros(sr // 2 - 1, dtype=np.float32)
        ]).tolist(),
        'sample_rate': sr,
        'description': 'Single impulse at center'
    }

    # Short signal for Griffin-Lim testing
    n_samples = 1024
    signals['short_sine'] = {
        'data': np.sin(0.1 * np.arange(n_samples)).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': 'Short signal for Griffin-Lim tests'
    }

    return signals


# =============================================================================
# CORE OPERATIONS
# =============================================================================

def generate_stft_reference(signal, sr, n_fft=2048, hop_length=512, win_length=None):
    """Generate STFT reference data."""
    y = np.array(signal, dtype=np.float32)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length or n_fft,
                     window='hann', center=True, pad_mode='constant')

    magnitude = np.abs(S)
    phase = np.angle(S)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length or n_fft,
        'window': 'hann',
        'center': True,
        'pad_mode': 'constant',
        'magnitude': magnitude.T.tolist(),
        'phase': phase.T.tolist(),
        'shape': list(S.shape),
        'librosa_version': librosa.__version__
    }


def generate_istft_reference(signal, sr, n_fft=2048, hop_length=512):
    """Generate ISTFT reference data (round-trip verification)."""
    y = np.array(signal, dtype=np.float32)

    # Forward STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     window='hann', center=True, pad_mode='constant')

    # Inverse STFT
    y_reconstructed = librosa.istft(S, hop_length=hop_length,
                                     win_length=n_fft, window='hann',
                                     center=True, length=len(y))

    # Compute reconstruction error
    error = np.sqrt(np.sum((y - y_reconstructed)**2) / np.sum(y**2 + 1e-10))

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'original_length': len(y),
        'reconstructed_length': len(y_reconstructed),
        'reconstructed_first_100': y_reconstructed[:100].tolist(),
        'reconstruction_error': float(error),
        'librosa_version': librosa.__version__
    }


def generate_db_conversion_reference():
    """Generate dB conversion reference data."""
    # Test values spanning typical audio range
    power_values = np.array([1e-10, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0])
    amplitude_values = np.sqrt(power_values)

    power_to_db = librosa.power_to_db(power_values, ref=1.0)
    amplitude_to_db = librosa.amplitude_to_db(amplitude_values, ref=1.0)

    # Round-trip
    db_to_power = librosa.db_to_power(power_to_db, ref=1.0)
    db_to_amplitude = librosa.db_to_amplitude(amplitude_to_db, ref=1.0)

    return {
        'power_values': power_values.tolist(),
        'amplitude_values': amplitude_values.tolist(),
        'power_to_db': power_to_db.tolist(),
        'amplitude_to_db': amplitude_to_db.tolist(),
        'db_to_power': db_to_power.tolist(),
        'db_to_amplitude': db_to_amplitude.tolist(),
        'ref': 1.0,
        'librosa_version': librosa.__version__
    }


# =============================================================================
# MEL AND FILTERBANK
# =============================================================================

def generate_mel_filterbank_reference(sr=22050, n_fft=2048, n_mels=128, fmin=0, fmax=None):
    """Generate Mel filterbank reference data."""
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                     fmin=fmin, fmax=fmax or sr/2,
                                     norm='slaney', htk=False)

    return {
        'sample_rate': sr,
        'n_fft': n_fft,
        'n_mels': n_mels,
        'fmin': fmin,
        'fmax': fmax or sr/2,
        'norm': 'slaney',
        'htk': False,
        'filterbank': mel_basis.tolist(),
        'shape': list(mel_basis.shape),
        'librosa_version': librosa.__version__
    }


def generate_mel_scale_reference():
    """Generate Mel scale conversion reference data."""
    test_frequencies = [0, 100, 200, 440, 700, 1000, 2000, 4000, 8000, 16000]

    htk_mel = librosa.hz_to_mel(test_frequencies, htk=True)
    htk_hz_back = librosa.mel_to_hz(htk_mel, htk=True)

    slaney_mel = librosa.hz_to_mel(test_frequencies, htk=False)
    slaney_hz_back = librosa.mel_to_hz(slaney_mel, htk=False)

    return {
        'test_frequencies_hz': test_frequencies,
        'htk': {
            'mel_values': htk_mel.tolist(),
            'round_trip_hz': htk_hz_back.tolist()
        },
        'slaney': {
            'mel_values': slaney_mel.tolist(),
            'round_trip_hz': slaney_hz_back.tolist()
        },
        'librosa_version': librosa.__version__
    }


def generate_mel_spectrogram_reference(signal, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Generate mel spectrogram reference data."""
    y = np.array(signal, dtype=np.float32)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length, n_mels=n_mels)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'mel_spectrogram': mel_spec.T.tolist(),
        'shape': list(mel_spec.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# SPECTRAL FEATURES
# =============================================================================

def generate_spectral_features_reference(signal, sr, n_fft=2048, hop_length=512):
    """Generate comprehensive spectral feature reference data."""
    y = np.array(signal, dtype=np.float32)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'centroid': centroid[0].tolist(),
        'bandwidth': bandwidth[0].tolist(),
        'rolloff': rolloff[0].tolist(),
        'flatness': flatness[0].tolist(),
        'contrast': contrast.T.tolist(),
        'contrast_shape': list(contrast.shape),
        'librosa_version': librosa.__version__
    }


def generate_zcr_reference(signal, sr, frame_length=2048, hop_length=512):
    """Generate zero crossing rate reference data."""
    y = np.array(signal, dtype=np.float32)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length,
                                              hop_length=hop_length)

    return {
        'frame_length': frame_length,
        'hop_length': hop_length,
        'zcr': zcr[0].tolist(),
        'shape': list(zcr.shape),
        'librosa_version': librosa.__version__
    }


def generate_rms_reference(signal, sr, frame_length=2048, hop_length=512):
    """Generate RMS energy reference data."""
    y = np.array(signal, dtype=np.float32)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    return {
        'frame_length': frame_length,
        'hop_length': hop_length,
        'rms': rms[0].tolist(),
        'shape': list(rms.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# MFCC AND RELATED
# =============================================================================

def generate_mfcc_reference(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
    """Generate MFCC reference data."""
    y = np.array(signal, dtype=np.float32)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                  hop_length=hop_length, n_mels=n_mels)

    return {
        'n_mfcc': n_mfcc,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'mfcc': mfccs.T.tolist(),
        'shape': list(mfccs.shape),
        'librosa_version': librosa.__version__
    }


def generate_delta_reference(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """Generate delta feature reference data."""
    y = np.array(signal, dtype=np.float32)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                  hop_length=hop_length)

    # Delta with different widths
    delta_3 = librosa.feature.delta(mfccs, width=3)
    delta_5 = librosa.feature.delta(mfccs, width=5)
    delta_9 = librosa.feature.delta(mfccs, width=9)

    # Second order delta (acceleration)
    delta2_5 = librosa.feature.delta(mfccs, width=5, order=2)

    return {
        'n_mfcc': n_mfcc,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'delta_width_3': delta_3.T.tolist(),
        'delta_width_5': delta_5.T.tolist(),
        'delta_width_9': delta_9.T.tolist(),
        'delta2_width_5': delta2_5.T.tolist(),
        'delta_shape': list(delta_5.shape),
        'librosa_version': librosa.__version__
    }


def generate_pcen_reference(signal, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Generate PCEN reference data.

    Note: We use the mel spectrogram directly without scaling. The original
    librosa PCEN defaults were designed for int32 PCM audio (2^31 range),
    but modern usage with normalized float audio [-1, 1] works correctly
    without scaling. SwiftRosa's PCEN implementation uses this approach.
    """
    y = np.array(signal, dtype=np.float32)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length, n_mels=n_mels)

    # Apply PCEN with librosa default parameters (explicitly specified for clarity)
    # Note: librosa defaults are gain=0.98, bias=2, power=0.5, time_constant=0.4, eps=1e-6
    pcen_default = librosa.pcen(mel_spec, sr=sr, hop_length=hop_length,
                                gain=0.98, bias=2.0, power=0.5,
                                time_constant=0.4, eps=1e-6)

    # Apply PCEN with custom parameters
    pcen_custom = librosa.pcen(mel_spec, sr=sr, hop_length=hop_length,
                               gain=0.8, bias=10, power=0.25,
                               time_constant=0.06, eps=1e-6)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'pcen_default': pcen_default.T.tolist(),
        'pcen_custom': pcen_custom.T.tolist(),
        'pcen_shape': list(pcen_default.shape),
        'default_params': {
            'gain': 0.98,
            'bias': 2.0,
            'power': 0.5,
            'time_constant': 0.4,
            'eps': 1e-6
        },
        'custom_params': {
            'gain': 0.8,
            'bias': 10,
            'power': 0.25,
            'time_constant': 0.06,
            'eps': 1e-6
        },
        'librosa_version': librosa.__version__
    }


# =============================================================================
# CQT AND RELATED TRANSFORMS
# =============================================================================

def generate_cqt_reference(signal, sr, hop_length=512, fmin=32.70, n_bins=84, bins_per_octave=12):
    """Generate CQT reference data."""
    y = np.array(signal, dtype=np.float32)

    C = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)

    magnitude = np.abs(C)
    phase = np.angle(C)

    # Get center frequencies
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin,
                                     bins_per_octave=bins_per_octave)

    return {
        'hop_length': hop_length,
        'fmin': fmin,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'magnitude': magnitude.T.tolist(),
        'phase': phase.T.tolist(),
        'center_frequencies': freqs.tolist(),
        'shape': list(C.shape),
        'librosa_version': librosa.__version__
    }


def generate_vqt_reference(signal, sr, hop_length=512, fmin=32.70, n_bins=84, bins_per_octave=12):
    """Generate VQT reference data with multiple gamma values."""
    y = np.array(signal, dtype=np.float32)

    results = {}
    for gamma in [0, 0.5, 1.0]:
        V = librosa.vqt(y, sr=sr, hop_length=hop_length, fmin=fmin,
                        n_bins=n_bins, bins_per_octave=bins_per_octave, gamma=gamma)
        results[f'gamma_{gamma}'] = {
            'magnitude': np.abs(V).T.tolist(),
            'shape': list(V.shape)
        }

    return {
        'hop_length': hop_length,
        'fmin': fmin,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'results': results,
        'librosa_version': librosa.__version__
    }


def generate_pseudo_cqt_reference(signal, sr, hop_length=512, fmin=32.70, n_bins=84, bins_per_octave=12):
    """Generate pseudo-CQT reference data."""
    y = np.array(signal, dtype=np.float32)

    C = librosa.pseudo_cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins)
    magnitude = np.abs(C)

    return {
        'hop_length': hop_length,
        'fmin': fmin,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'magnitude': magnitude.T.tolist(),
        'shape': list(C.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# CHROMAGRAM AND TONNETZ
# =============================================================================

def generate_chromagram_reference(signal, sr, n_fft=2048, hop_length=512, n_chroma=12):
    """Generate chromagram reference data for both methods."""
    y = np.array(signal, dtype=np.float32)

    # STFT-based chroma
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length, n_chroma=n_chroma)

    # CQT-based chroma
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length,
                                             n_chroma=n_chroma)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_chroma': n_chroma,
        'chroma_stft': chroma_stft.T.tolist(),
        'chroma_cqt': chroma_cqt.T.tolist(),
        'shape': list(chroma_stft.shape),
        'librosa_version': librosa.__version__
    }


def generate_tonnetz_reference(signal, sr, hop_length=512):
    """Generate tonnetz reference data.

    Includes chroma_cqt for isolated tonnetz transformation validation.
    """
    y = np.array(signal, dtype=np.float32)

    # Get chroma_cqt (used internally by tonnetz)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Get tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)

    return {
        'hop_length': hop_length,
        'chroma_cqt': chroma.T.tolist(),  # (nFrames, 12) for JSON
        'chroma_shape': list(chroma.shape),
        'tonnetz': tonnetz.T.tolist(),
        'shape': list(tonnetz.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# ONSET AND BEAT
# =============================================================================

def generate_onset_reference(signal, sr, hop_length=512):
    """Generate onset detection reference data."""
    y = np.array(signal, dtype=np.float32)

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames')
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    return {
        'hop_length': hop_length,
        'onset_strength': onset_env.tolist(),
        'onset_frames': onsets.tolist(),
        'onset_times': onset_times.tolist(),
        'librosa_version': librosa.__version__
    }


def generate_beat_reference(signal, sr, hop_length=512):
    """Generate beat tracking reference data."""
    y = np.array(signal, dtype=np.float32)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    return {
        'hop_length': hop_length,
        'tempo': float(tempo) if np.isscalar(tempo) else float(tempo[0]),
        'beat_frames': beats.tolist(),
        'beat_times': beat_times.tolist(),
        'librosa_version': librosa.__version__
    }


def generate_tempogram_reference(signal, sr, hop_length=512):
    """Generate tempogram reference data."""
    y = np.array(signal, dtype=np.float32)

    # Onset strength for tempogram
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                           hop_length=hop_length)

    return {
        'hop_length': hop_length,
        'tempogram': tempogram.T.tolist(),
        'shape': list(tempogram.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# EFFECTS AND DECOMPOSITION
# =============================================================================

def generate_hpss_reference(signal, sr, n_fft=2048, hop_length=512):
    """Generate HPSS reference data."""
    y = np.array(signal, dtype=np.float32)

    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # HPSS on spectrogram
    H, P = librosa.decompose.hpss(S)

    harmonic = librosa.istft(H, hop_length=hop_length, length=len(y))
    percussive = librosa.istft(P, hop_length=hop_length, length=len(y))

    # Energy metrics
    original_energy = float(np.sum(y**2))
    harmonic_energy = float(np.sum(harmonic**2))
    percussive_energy = float(np.sum(percussive**2))

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'harmonic_first_1000': harmonic[:1000].tolist(),
        'percussive_first_1000': percussive[:1000].tolist(),
        'original_energy': original_energy,
        'harmonic_energy': harmonic_energy,
        'percussive_energy': percussive_energy,
        'energy_ratio': (harmonic_energy + percussive_energy) / (original_energy + 1e-10),
        'librosa_version': librosa.__version__
    }


def generate_nmf_reference(signal, sr, n_fft=2048, hop_length=512, n_components=4):
    """Generate NMF decomposition reference data."""
    y = np.array(signal, dtype=np.float32)

    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # NMF decomposition
    W, H = librosa.decompose.decompose(S, n_components=n_components, sort=True)

    # Reconstruction error
    S_reconstructed = np.dot(W, H)
    error = np.sqrt(np.sum((S - S_reconstructed)**2) / np.sum(S**2 + 1e-10))

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_components': n_components,
        'W_shape': list(W.shape),
        'H_shape': list(H.shape),
        'W': W.tolist(),
        'H': H.T.tolist(),
        'reconstruction_error': float(error),
        'librosa_version': librosa.__version__
    }


def generate_griffin_lim_reference(magnitude, n_fft=256, hop_length=64, n_iter=32):
    """Generate Griffin-Lim reference data."""
    mag = np.array(magnitude, dtype=np.float32).T

    reconstructed = librosa.griffinlim(mag, n_iter=n_iter, hop_length=hop_length,
                                        win_length=n_fft, window='hann',
                                        center=True, pad_mode='constant',
                                        momentum=0.99, init='random',
                                        random_state=42)

    S_reconstructed = librosa.stft(reconstructed, n_fft=n_fft, hop_length=hop_length,
                                    win_length=n_fft, window='hann',
                                    center=True, pad_mode='constant')
    mag_reconstructed = np.abs(S_reconstructed)

    error = np.sqrt(np.sum((mag - mag_reconstructed)**2) / np.sum(mag**2))

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_iter': n_iter,
        'momentum': 0.99,
        'reconstructed_first_100': reconstructed[:100].tolist(),
        'spectral_error': float(error),
        'librosa_version': librosa.__version__
    }


def generate_phase_vocoder_reference(signal, sr, n_fft=2048, hop_length=512):
    """Generate phase vocoder reference data for various stretch rates."""
    y = np.array(signal, dtype=np.float32)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    results = {}
    for rate in [0.5, 0.8, 1.0, 1.25, 2.0]:
        S_stretched = librosa.phase_vocoder(S, rate=rate, hop_length=hop_length)
        y_stretched = librosa.istft(S_stretched, hop_length=hop_length)

        results[f'rate_{rate}'] = {
            'output_length': len(y_stretched),
            'first_100_samples': y_stretched[:100].tolist(),
            'energy': float(np.sum(y_stretched**2))
        }

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'original_length': len(y),
        'results': results,
        'librosa_version': librosa.__version__
    }


# =============================================================================
# WINDOW FUNCTIONS
# =============================================================================

def generate_window_reference():
    """Generate window function reference data."""
    window_length = 1024

    windows = {}

    # Hann window
    windows['hann_periodic'] = scipy.signal.get_window('hann', window_length, fftbins=True).tolist()
    windows['hann_symmetric'] = scipy.signal.get_window('hann', window_length, fftbins=False).tolist()

    # Hamming window
    windows['hamming_periodic'] = scipy.signal.get_window('hamming', window_length, fftbins=True).tolist()
    windows['hamming_symmetric'] = scipy.signal.get_window('hamming', window_length, fftbins=False).tolist()

    # Blackman window
    windows['blackman_periodic'] = scipy.signal.get_window('blackman', window_length, fftbins=True).tolist()
    windows['blackman_symmetric'] = scipy.signal.get_window('blackman', window_length, fftbins=False).tolist()

    # Bartlett window
    windows['bartlett_periodic'] = scipy.signal.get_window('bartlett', window_length, fftbins=True).tolist()
    windows['bartlett_symmetric'] = scipy.signal.get_window('bartlett', window_length, fftbins=False).tolist()

    return {
        'window_length': window_length,
        'windows': windows,
        'scipy_version': scipy.__version__
    }


# =============================================================================
# RESAMPLING
# =============================================================================

def generate_resample_reference(signal, sr_orig, sr_target):
    """Generate resampling reference data."""
    y = np.array(signal, dtype=np.float32)

    y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)

    return {
        'original_sr': sr_orig,
        'target_sr': sr_target,
        'original_length': len(y),
        'resampled_length': len(y_resampled),
        'resampled_first_100': y_resampled[:100].tolist(),
        'resampled_energy': float(np.sum(y_resampled**2)),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# POLY FEATURES
# =============================================================================

def generate_poly_features_reference(signal, sr, n_fft=2048, hop_length=512, order=1):
    """Generate polynomial feature reference data."""
    y = np.array(signal, dtype=np.float32)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    poly = librosa.feature.poly_features(S=S, order=order)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'order': order,
        'poly_features': poly.T.tolist(),
        'shape': list(poly.shape),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# EMPHASIS (PREEMPHASIS / DEEMPHASIS)
# =============================================================================

def generate_emphasis_reference(signal, sr, coef=0.97):
    """Generate preemphasis and deemphasis reference data."""
    y = np.array(signal, dtype=np.float32)

    # Apply preemphasis
    preemph = librosa.effects.preemphasis(y, coef=coef)

    # Apply deemphasis
    deemph = librosa.effects.deemphasis(y, coef=coef)

    # Also test round-trip: preemph -> deemph should recover original
    roundtrip = librosa.effects.deemphasis(preemph, coef=coef)

    return {
        'coef': coef,
        'original_length': len(y),
        'preemphasis': preemph.tolist(),
        'deemphasis': deemph.tolist(),
        'roundtrip': roundtrip.tolist(),
        'preemphasis_energy': float(np.sum(preemph**2)),
        'deemphasis_energy': float(np.sum(deemph**2)),
        'roundtrip_error': float(np.max(np.abs(y - roundtrip))),
        'librosa_version': librosa.__version__
    }


# =============================================================================
# TIME STRETCH (END-TO-END)
# =============================================================================

def generate_time_stretch_reference(signal, sr, rate=1.5):
    """Generate time_stretch reference data (end-to-end, not just phase vocoder)."""
    y = np.array(signal, dtype=np.float32)

    # librosa.effects.time_stretch uses phase_vocoder internally
    stretched = librosa.effects.time_stretch(y, rate=rate)

    return {
        'rate': rate,
        'original_length': len(y),
        'stretched_length': len(stretched),
        'first_500_samples': stretched[:500].tolist(),
        'last_100_samples': stretched[-100:].tolist() if len(stretched) >= 100 else stretched.tolist(),
        'stretched_energy': float(np.sum(stretched**2)),
        'original_energy': float(np.sum(y**2)),
        'energy_ratio': float(np.sum(stretched**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0,
        'librosa_version': librosa.__version__
    }


# =============================================================================
# PITCH SHIFT
# =============================================================================

def generate_pitch_shift_reference(signal, sr, n_steps=4):
    """Generate pitch_shift reference data."""
    y = np.array(signal, dtype=np.float32)

    # librosa.effects.pitch_shift combines time_stretch + resample
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    return {
        'n_steps': n_steps,
        'sample_rate': sr,
        'original_length': len(y),
        'shifted_length': len(shifted),
        'first_500_samples': shifted[:500].tolist(),
        'last_100_samples': shifted[-100:].tolist() if len(shifted) >= 100 else shifted.tolist(),
        'shifted_energy': float(np.sum(shifted**2)),
        'original_energy': float(np.sum(y**2)),
        'energy_ratio': float(np.sum(shifted**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0,
        'librosa_version': librosa.__version__
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path(__file__).parent.parent / 'Tests' / 'SwiftRosaCoreTests' / 'ReferenceData'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SwiftRosa Reference Data Generator")
    print("=" * 60)
    print(f"librosa version: {librosa.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"scipy version: {scipy.__version__}")
    print(f"Generated at: {datetime.now().isoformat()}")
    print()

    # Generate test signals
    print("Generating test signals...")
    signals = generate_test_signals()

    reference_data = {
        'metadata': {
            'generator': 'generate_reference_data.py',
            'librosa_version': librosa.__version__,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
            'generated_at': datetime.now().isoformat(),
            'float_precision': 'float32'
        },
        'test_signals': {name: {
            'data': sig['data'],
            'sample_rate': sig['sample_rate'],
            'description': sig['description']
        } for name, sig in signals.items()}
    }

    # STFT reference
    print("Generating STFT reference...")
    stft_refs = {}
    for name, sig_data in signals.items():
        if name not in ['silence', 'dc_offset']:  # Skip edge cases for STFT
            stft_refs[name] = generate_stft_reference(
                sig_data['data'], sig_data['sample_rate'],
                n_fft=1024, hop_length=256
            )
    reference_data['stft'] = stft_refs

    # ISTFT round-trip reference
    print("Generating ISTFT reference...")
    istft_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp']:
        istft_refs[name] = generate_istft_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['istft'] = istft_refs

    # dB conversion reference
    print("Generating dB conversion reference...")
    reference_data['db_conversion'] = generate_db_conversion_reference()

    # Mel filterbank reference
    print("Generating Mel filterbank reference...")
    reference_data['mel_filterbank'] = {
        '128_mels': generate_mel_filterbank_reference(n_mels=128),
        '64_mels': generate_mel_filterbank_reference(n_mels=64),
        '40_mels': generate_mel_filterbank_reference(n_mels=40),
    }

    # Mel spectrogram reference
    print("Generating Mel spectrogram reference...")
    mel_spec_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp', 'white_noise']:
        mel_spec_refs[name] = generate_mel_spectrogram_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['mel_spectrogram'] = mel_spec_refs

    # Mel scale reference
    print("Generating Mel scale reference...")
    reference_data['mel_scale'] = generate_mel_scale_reference()

    # Spectral features reference
    print("Generating spectral features reference...")
    spectral_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp', 'white_noise']:
        spectral_refs[name] = generate_spectral_features_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['spectral_features'] = spectral_refs

    # ZCR reference
    print("Generating ZCR reference...")
    zcr_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'white_noise', 'click_train']:
        zcr_refs[name] = generate_zcr_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['zcr'] = zcr_refs

    # RMS reference
    print("Generating RMS reference...")
    rms_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'white_noise', 'am_modulated']:
        rms_refs[name] = generate_rms_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['rms'] = rms_refs

    # MFCC reference
    print("Generating MFCC reference...")
    mfcc_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp']:
        mfcc_refs[name] = generate_mfcc_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['mfcc'] = mfcc_refs

    # Delta reference
    print("Generating Delta reference...")
    delta_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        delta_refs[name] = generate_delta_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['delta'] = delta_refs

    # PCEN reference
    print("Generating PCEN reference...")
    pcen_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp']:
        pcen_refs[name] = generate_pcen_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['pcen'] = pcen_refs

    # CQT reference
    print("Generating CQT reference...")
    cqt_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp']:
        cqt_refs[name] = generate_cqt_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['cqt'] = cqt_refs

    # VQT reference
    print("Generating VQT reference...")
    vqt_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        vqt_refs[name] = generate_vqt_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['vqt'] = vqt_refs

    # Pseudo-CQT reference
    print("Generating Pseudo-CQT reference...")
    pseudo_cqt_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        pseudo_cqt_refs[name] = generate_pseudo_cqt_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['pseudo_cqt'] = pseudo_cqt_refs

    # Chromagram reference
    print("Generating Chromagram reference...")
    chroma_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        chroma_refs[name] = generate_chromagram_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['chromagram'] = chroma_refs

    # Tonnetz reference
    print("Generating Tonnetz reference...")
    tonnetz_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        tonnetz_refs[name] = generate_tonnetz_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['tonnetz'] = tonnetz_refs

    # Onset reference
    print("Generating Onset reference...")
    onset_refs = {}
    for name in ['multi_tone', 'click_train', 'am_modulated']:
        onset_refs[name] = generate_onset_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['onset'] = onset_refs

    # Beat reference
    print("Generating Beat reference...")
    beat_refs = {}
    for name in ['multi_tone', 'click_train']:
        beat_refs[name] = generate_beat_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['beat'] = beat_refs

    # Tempogram reference
    print("Generating Tempogram reference...")
    tempogram_refs = {}
    for name in ['multi_tone', 'click_train']:
        tempogram_refs[name] = generate_tempogram_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['tempogram'] = tempogram_refs

    # HPSS reference
    print("Generating HPSS reference...")
    hpss_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'click_train']:
        hpss_refs[name] = generate_hpss_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['hpss'] = hpss_refs

    # NMF reference
    print("Generating NMF reference...")
    nmf_refs = {}
    for name in ['multi_tone']:
        nmf_refs[name] = generate_nmf_reference(
            signals[name]['data'], signals[name]['sample_rate']
        )
    reference_data['nmf'] = nmf_refs

    # Window functions reference
    print("Generating window functions reference...")
    reference_data['windows'] = generate_window_reference()

    # Griffin-Lim reference
    print("Generating Griffin-Lim reference...")
    short_signal = signals['short_sine']['data']
    short_stft = generate_stft_reference(short_signal, 22050, n_fft=256, hop_length=64)
    reference_data['griffin_lim'] = generate_griffin_lim_reference(
        short_stft['magnitude'], n_fft=256, hop_length=64, n_iter=32
    )

    # Phase vocoder reference
    print("Generating Phase vocoder reference...")
    reference_data['phase_vocoder'] = generate_phase_vocoder_reference(
        signals['sine_440hz']['data'], signals['sine_440hz']['sample_rate']
    )

    # Resampling reference
    print("Generating resampling reference...")
    reference_data['resample'] = {
        '44100_to_22050': generate_resample_reference(
            signals['sine_440hz']['data'], 44100, 22050
        ),
        '22050_to_16000': generate_resample_reference(
            signals['sine_440hz']['data'], 22050, 16000
        ),
        '22050_to_8000': generate_resample_reference(
            signals['sine_440hz']['data'], 22050, 8000
        ),
        '16000_to_44100': generate_resample_reference(
            signals['sine_440hz']['data'], 16000, 44100
        ),
    }

    # Poly features reference
    print("Generating poly features reference...")
    poly_refs = {}
    for order in [1, 2]:
        poly_refs[f'order_{order}'] = generate_poly_features_reference(
            signals['multi_tone']['data'], signals['multi_tone']['sample_rate'], order=order
        )
    reference_data['poly_features'] = poly_refs

    # Emphasis (preemphasis / deemphasis) reference
    print("Generating emphasis reference...")
    emphasis_refs = {}
    for name in ['sine_440hz', 'multi_tone', 'chirp']:
        emphasis_refs[name] = {
            'coef_0.97': generate_emphasis_reference(
                signals[name]['data'], signals[name]['sample_rate'], coef=0.97
            ),
            'coef_0.95': generate_emphasis_reference(
                signals[name]['data'], signals[name]['sample_rate'], coef=0.95
            ),
        }
    reference_data['emphasis'] = emphasis_refs

    # Time stretch reference
    print("Generating time stretch reference...")
    time_stretch_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        time_stretch_refs[name] = {
            'rate_0.5': generate_time_stretch_reference(
                signals[name]['data'], signals[name]['sample_rate'], rate=0.5
            ),
            'rate_0.8': generate_time_stretch_reference(
                signals[name]['data'], signals[name]['sample_rate'], rate=0.8
            ),
            'rate_1.25': generate_time_stretch_reference(
                signals[name]['data'], signals[name]['sample_rate'], rate=1.25
            ),
            'rate_2.0': generate_time_stretch_reference(
                signals[name]['data'], signals[name]['sample_rate'], rate=2.0
            ),
        }
    reference_data['time_stretch'] = time_stretch_refs

    # Pitch shift reference
    print("Generating pitch shift reference...")
    pitch_shift_refs = {}
    for name in ['sine_440hz', 'multi_tone']:
        pitch_shift_refs[name] = {
            'steps_2': generate_pitch_shift_reference(
                signals[name]['data'], signals[name]['sample_rate'], n_steps=2
            ),
            'steps_4': generate_pitch_shift_reference(
                signals[name]['data'], signals[name]['sample_rate'], n_steps=4
            ),
            'steps_-3': generate_pitch_shift_reference(
                signals[name]['data'], signals[name]['sample_rate'], n_steps=-3
            ),
        }
    reference_data['pitch_shift'] = pitch_shift_refs

    # Save to JSON
    output_file = output_dir / 'librosa_reference.json'
    print()
    print(f"Saving to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    # Calculate file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print()
    print("=" * 60)
    print("Generation complete!")
    print(f"Reference data saved to: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
