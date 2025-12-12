#!/usr/bin/env python3
"""
Generate librosa reference data for SwiftAudioPrimitives validation.

This script generates ground truth values from librosa for comparison
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


def generate_test_signals():
    """Generate standard test signals."""
    signals = {}

    # Simple sine wave at 440 Hz
    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    signals['sine_440hz'] = {
        'data': np.sin(2 * np.pi * 440 * t).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': 'Pure 440 Hz sine wave'
    }

    # Multi-tone signal
    signals['multi_tone'] = {
        'data': (0.5 * np.sin(2 * np.pi * 440 * t) +
                 0.3 * np.sin(2 * np.pi * 880 * t) +
                 0.2 * np.sin(2 * np.pi * 1760 * t)).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': '440Hz + 880Hz + 1760Hz harmonics'
    }

    # Short signal for Griffin-Lim testing
    n_samples = 1024
    t_short = np.arange(n_samples) / sr
    signals['short_sine'] = {
        'data': np.sin(0.1 * np.arange(n_samples)).astype(np.float32).tolist(),
        'sample_rate': sr,
        'description': 'Short signal for Griffin-Lim tests'
    }

    return signals


def generate_stft_reference(signal, sr, n_fft=2048, hop_length=512, win_length=None):
    """Generate STFT reference data."""
    y = np.array(signal, dtype=np.float32)

    # Use librosa's STFT with default parameters
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
        'magnitude': magnitude.T.tolist(),  # Transpose to (nFrames, nFreqs) for easier comparison
        'phase': phase.T.tolist(),
        'shape': list(S.shape),
        'librosa_version': librosa.__version__
    }


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
        'mfcc': mfccs.T.tolist(),  # Transpose to (nFrames, nCoeffs)
        'shape': list(mfccs.shape),
        'librosa_version': librosa.__version__
    }


def generate_griffin_lim_reference(magnitude, n_fft=256, hop_length=64, n_iter=32):
    """Generate Griffin-Lim reference data."""
    mag = np.array(magnitude, dtype=np.float32).T  # Transpose back to librosa format

    # Use librosa's Griffin-Lim
    reconstructed = librosa.griffinlim(mag, n_iter=n_iter, hop_length=hop_length,
                                        win_length=n_fft, window='hann',
                                        center=True, pad_mode='constant',
                                        momentum=0.99, init='random',
                                        random_state=42)

    # Compute spectral error
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
        'reconstructed_first_10': reconstructed[:10].tolist(),
        'spectral_error': float(error),
        'librosa_version': librosa.__version__
    }


def generate_mel_scale_reference():
    """Generate Mel scale conversion reference data."""
    test_frequencies = [0, 100, 200, 440, 700, 1000, 2000, 4000, 8000, 16000]

    # HTK formula
    htk_mel = librosa.hz_to_mel(test_frequencies, htk=True)
    htk_hz_back = librosa.mel_to_hz(htk_mel, htk=True)

    # Slaney formula (default)
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


def generate_spectral_features_reference(signal, sr, n_fft=2048, hop_length=512):
    """Generate spectral feature reference data."""
    y = np.array(signal, dtype=np.float32)

    # Compute spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(S=S)

    return {
        'n_fft': n_fft,
        'hop_length': hop_length,
        'centroid': centroid[0].tolist(),
        'bandwidth': bandwidth[0].tolist(),
        'rolloff': rolloff[0].tolist(),
        'flatness': flatness[0].tolist(),
        'librosa_version': librosa.__version__
    }


def generate_window_reference():
    """Generate window function reference data."""
    window_length = 1024

    windows = {}

    # Hann window (periodic)
    windows['hann_periodic'] = scipy.signal.get_window('hann', window_length, fftbins=True).tolist()

    # Hann window (symmetric)
    windows['hann_symmetric'] = scipy.signal.get_window('hann', window_length, fftbins=False).tolist()

    # Hamming window (periodic)
    windows['hamming_periodic'] = scipy.signal.get_window('hamming', window_length, fftbins=True).tolist()

    # Blackman window (periodic)
    windows['blackman_periodic'] = scipy.signal.get_window('blackman', window_length, fftbins=True).tolist()

    return {
        'window_length': window_length,
        'windows': windows,
        'scipy_version': scipy.__version__
    }


def generate_resample_reference(signal, sr_orig, sr_target):
    """Generate resampling reference data."""
    y = np.array(signal, dtype=np.float32)

    y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)

    return {
        'original_sr': sr_orig,
        'target_sr': sr_target,
        'original_length': len(y),
        'resampled_length': len(y_resampled),
        'resampled_first_20': y_resampled[:20].tolist(),
        'resampled_energy': float(np.sum(y_resampled**2)),
        'librosa_version': librosa.__version__
    }


def main():
    output_dir = Path(__file__).parent.parent / 'Tests' / 'ReferenceData'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating librosa reference data...")
    print(f"librosa version: {librosa.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"scipy version: {scipy.__version__}")

    # Generate test signals
    signals = generate_test_signals()

    reference_data = {
        'generator': 'generate_reference_data.py',
        'librosa_version': librosa.__version__,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
    }

    # STFT reference
    print("Generating STFT reference...")
    stft_refs = {}
    for name, sig_data in signals.items():
        stft_refs[name] = generate_stft_reference(
            sig_data['data'], sig_data['sample_rate'],
            n_fft=1024, hop_length=256
        )
    reference_data['stft'] = stft_refs

    # Mel filterbank reference
    print("Generating Mel filterbank reference...")
    reference_data['mel_filterbank'] = {
        '128_mels': generate_mel_filterbank_reference(n_mels=128),
        '64_mels': generate_mel_filterbank_reference(n_mels=64),
        '40_mels': generate_mel_filterbank_reference(n_mels=40),
    }

    # MFCC reference
    print("Generating MFCC reference...")
    mfcc_refs = {}
    for name, sig_data in signals.items():
        mfcc_refs[name] = generate_mfcc_reference(
            sig_data['data'], sig_data['sample_rate']
        )
    reference_data['mfcc'] = mfcc_refs

    # Mel scale reference
    print("Generating Mel scale reference...")
    reference_data['mel_scale'] = generate_mel_scale_reference()

    # Spectral features reference
    print("Generating spectral features reference...")
    spectral_refs = {}
    for name, sig_data in signals.items():
        spectral_refs[name] = generate_spectral_features_reference(
            sig_data['data'], sig_data['sample_rate']
        )
    reference_data['spectral_features'] = spectral_refs

    # Window functions reference
    print("Generating window functions reference...")
    reference_data['windows'] = generate_window_reference()

    # Griffin-Lim reference (using short signal)
    print("Generating Griffin-Lim reference...")
    short_signal = signals['short_sine']['data']
    short_stft = generate_stft_reference(short_signal, 22050, n_fft=256, hop_length=64)
    reference_data['griffin_lim'] = generate_griffin_lim_reference(
        short_stft['magnitude'], n_fft=256, hop_length=64, n_iter=32
    )

    # Resampling reference
    print("Generating resampling reference...")
    resample_refs = {
        '44100_to_22050': generate_resample_reference(
            signals['sine_440hz']['data'], 44100, 22050
        ),
        '22050_to_16000': generate_resample_reference(
            signals['sine_440hz']['data'], 22050, 16000
        ),
    }
    reference_data['resample'] = resample_refs

    # Save to JSON
    output_file = output_dir / 'librosa_reference.json'
    print(f"Saving to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print("Done!")
    print(f"Reference data saved to: {output_file}")


if __name__ == '__main__':
    main()
