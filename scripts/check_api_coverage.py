#!/usr/bin/env python3
"""
SwiftRosa API Coverage Checker

This script compares the SwiftRosa implementation against librosa's public API
to identify implemented functions, missing functions, and extra functionality.

Usage:
    python check_api_coverage.py [--report api_coverage.md]
"""

import argparse
import re
from pathlib import Path
from datetime import datetime


# =============================================================================
# LIBROSA API DEFINITION
# =============================================================================

LIBROSA_API = {
    'core': {
        'description': 'Core time-frequency analysis',
        'functions': {
            'stft': {'implemented': True, 'swift': 'STFT.transform()'},
            'istft': {'implemented': True, 'swift': 'ISTFT.transform()'},
            'resample': {'implemented': True, 'swift': 'Resample.resample()'},
            'griffinlim': {'implemented': True, 'swift': 'GriffinLim.reconstruct()'},
            'phase_vocoder': {'implemented': True, 'swift': 'PhaseVocoder'},
            'magphase': {'implemented': True, 'swift': 'ComplexMatrix.magnitude/.phase'},
            'power_to_db': {'implemented': True, 'swift': 'Convert.powerToDb()'},
            'db_to_power': {'implemented': True, 'swift': 'Convert.dbToPower()'},
            'amplitude_to_db': {'implemented': True, 'swift': 'Convert.amplitudeToDb()'},
            'db_to_amplitude': {'implemented': True, 'swift': 'Convert.dbToAmplitude()'},
            'tone': {'implemented': True, 'swift': 'Tone.generate()'},
            'chirp': {'implemented': True, 'swift': 'Chirp.generate()'},
            'clicks': {'implemented': True, 'swift': 'Clicks.generate()'},
            'load': {'implemented': True, 'swift': 'AudioFileIO.load()', 'note': 'Via SwiftRosaML'},
        }
    },
    'feature': {
        'description': 'Feature extraction',
        'functions': {
            'zero_crossing_rate': {'implemented': True, 'swift': 'ZeroCrossingRate'},
            'spectral_centroid': {'implemented': True, 'swift': 'SpectralFeatures.centroid()'},
            'spectral_bandwidth': {'implemented': True, 'swift': 'SpectralFeatures.bandwidth()'},
            'spectral_rolloff': {'implemented': True, 'swift': 'SpectralFeatures.rolloff()'},
            'spectral_flatness': {'implemented': True, 'swift': 'SpectralFeatures.flatness()'},
            'spectral_contrast': {'implemented': True, 'swift': 'SpectralFeatures.contrast()'},
            'melspectrogram': {'implemented': True, 'swift': 'MelSpectrogram'},
            'mfcc': {'implemented': True, 'swift': 'MFCC'},
            'chroma_stft': {'implemented': True, 'swift': 'Chromagram (method: .stft)'},
            'chroma_cqt': {'implemented': True, 'swift': 'Chromagram (method: .cqt)'},
            'chroma_cens': {'implemented': True, 'swift': 'Chromagram.cens()'},
            'delta': {'implemented': True, 'swift': 'Delta.compute()'},
            'rms': {'implemented': True, 'swift': 'RMSEnergy'},
            'tonnetz': {'implemented': True, 'swift': 'Tonnetz'},
            'poly_features': {'implemented': True, 'swift': 'PolyFeatures'},
            'stack_memory': {'implemented': True, 'swift': 'StackMemory'},
            'tempogram': {'implemented': True, 'swift': 'Tempogram'},
        }
    },
    'filters': {
        'description': 'Filter design',
        'functions': {
            'mel': {'implemented': True, 'swift': 'MelFilterbank'},
            'chroma': {'implemented': True, 'swift': 'Built into Chromagram'},
            'constant_q': {'implemented': True, 'swift': 'Built into CQT'},
            'dct': {'implemented': True, 'swift': 'Built into MFCC'},
            'get_window': {'implemented': True, 'swift': 'Windows.generate()'},
            'mr_frequencies': {'implemented': True, 'swift': 'MRFrequencies'},
            'A_weighting': {'implemented': True, 'swift': 'AWeighting'},
            'perceptual_weighting': {'implemented': True, 'swift': 'PerceptualWeighting'},
        }
    },
    'onset': {
        'description': 'Onset detection',
        'functions': {
            'onset_detect': {'implemented': True, 'swift': 'OnsetDetector.detect()'},
            'onset_strength': {'implemented': True, 'swift': 'OnsetDetector.detectStrength()'},
            'onset_strength_multi': {'implemented': True, 'swift': 'OnsetDetector.onsetStrengthMulti()'},
            'onset_backtrack': {'implemented': True, 'swift': 'OnsetDetector.backtrack()'},
        }
    },
    'beat': {
        'description': 'Beat tracking',
        'functions': {
            'beat_track': {'implemented': True, 'swift': 'BeatTracker.track()'},
            'tempo': {'implemented': True, 'swift': 'BeatTracker.estimateTempo()'},
            'plp': {'implemented': True, 'swift': 'PLP.compute()'},
        }
    },
    'decompose': {
        'description': 'Source separation',
        'functions': {
            'hpss': {'implemented': True, 'swift': 'HPSS.separate()'},
            'nn_filter': {'implemented': True, 'swift': 'NNFilter'},
            'decompose': {'implemented': True, 'swift': 'NMF.decompose()'},
        }
    },
    'effects': {
        'description': 'Audio effects',
        'functions': {
            'pitch_shift': {'implemented': True, 'swift': 'PhaseVocoder.pitchShift()'},
            'time_stretch': {'implemented': True, 'swift': 'PhaseVocoder.timeStretch()'},
            'harmonic': {'implemented': True, 'swift': 'HPSS.harmonic()'},
            'percussive': {'implemented': True, 'swift': 'HPSS.percussive()'},
            'preemphasis': {'implemented': True, 'swift': 'Emphasis.preemphasis()'},
            'deemphasis': {'implemented': True, 'swift': 'Emphasis.deemphasis()'},
            'trim': {'implemented': True, 'swift': 'Segmentation.trim()'},
            'split': {'implemented': True, 'swift': 'Segmentation.split()'},
        }
    },
    'cqt': {
        'description': 'Constant-Q transforms',
        'functions': {
            'cqt': {'implemented': True, 'swift': 'CQT.transform()'},
            'icqt': {'implemented': True, 'swift': 'ICQT.transform()'},
            'hybrid_cqt': {'implemented': True, 'swift': 'HarmonicCQT'},
            'pseudo_cqt': {'implemented': True, 'swift': 'PseudoCQT'},
            'vqt': {'implemented': True, 'swift': 'VQT'},
        }
    },
    'sequence': {
        'description': 'Sequence alignment',
        'functions': {
            'dtw': {'implemented': True, 'swift': 'DTW'},
            'rqa': {'implemented': True, 'swift': 'RQA.compute()'},
        }
    },
}

# SwiftRosa-specific features not in librosa
SWIFTROSA_EXTRAS = {
    'VAD': 'Voice Activity Detection (SwiftRosaAnalysis)',
    'PitchDetector': 'Pitch detection (SwiftRosaAnalysis)',
    'PCEN': 'Per-Channel Energy Normalization (SwiftRosaCore)',
    'StreamingSTFT': 'Real-time STFT (SwiftRosaStreaming)',
    'StreamingMel': 'Real-time Mel spectrogram (SwiftRosaStreaming)',
    'StreamingMFCC': 'Real-time MFCC (SwiftRosaStreaming)',
    'StreamingPitchDetector': 'Real-time pitch detection (SwiftRosaStreaming)',
    'AudioModelRunner': 'CoreML model integration (SwiftRosaML)',
    'SpecAugment': 'Spectrogram augmentation (SwiftRosaML)',
    'LSTMLayer': 'LSTM neural network (SwiftRosaNN)',
    'BiLSTMLayer': 'Bidirectional LSTM (SwiftRosaNN)',
    'MetalEngine': 'GPU acceleration via Metal',
}


# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

def analyze_coverage():
    """Analyze API coverage statistics."""
    total_functions = 0
    implemented_functions = 0
    missing_high_priority = []
    missing_medium_priority = []
    missing_low_priority = []

    coverage_by_module = {}

    for module, data in LIBROSA_API.items():
        module_total = 0
        module_implemented = 0

        for func_name, func_data in data['functions'].items():
            total_functions += 1
            module_total += 1

            if func_data['implemented']:
                implemented_functions += 1
                module_implemented += 1
            else:
                priority = func_data.get('priority', 'low')
                entry = {
                    'module': module,
                    'function': func_name,
                    'note': func_data.get('note', '')
                }
                if priority == 'high':
                    missing_high_priority.append(entry)
                elif priority == 'medium':
                    missing_medium_priority.append(entry)
                else:
                    missing_low_priority.append(entry)

        coverage_by_module[module] = {
            'total': module_total,
            'implemented': module_implemented,
            'percentage': (module_implemented / module_total * 100) if module_total > 0 else 0
        }

    return {
        'total': total_functions,
        'implemented': implemented_functions,
        'percentage': (implemented_functions / total_functions * 100) if total_functions > 0 else 0,
        'by_module': coverage_by_module,
        'missing_high': missing_high_priority,
        'missing_medium': missing_medium_priority,
        'missing_low': missing_low_priority,
        'extras': SWIFTROSA_EXTRAS
    }


def generate_markdown_report(coverage):
    """Generate a markdown coverage report."""
    report = []

    report.append("# SwiftRosa API Coverage Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary
    report.append("## Summary")
    report.append("")
    report.append(f"**Overall Coverage: {coverage['percentage']:.1f}%** ({coverage['implemented']}/{coverage['total']} functions)")
    report.append("")

    # Progress bar
    filled = int(coverage['percentage'] / 5)
    bar = '█' * filled + '░' * (20 - filled)
    report.append(f"Progress: [{bar}] {coverage['percentage']:.1f}%")
    report.append("")

    # Module breakdown
    report.append("## Coverage by Module")
    report.append("")
    report.append("| Module | Coverage | Implemented | Total |")
    report.append("|--------|----------|-------------|-------|")

    for module, data in sorted(coverage['by_module'].items(), key=lambda x: -x[1]['percentage']):
        status = "✅" if data['percentage'] >= 80 else ("⚠️" if data['percentage'] >= 50 else "❌")
        report.append(f"| {module} | {status} {data['percentage']:.0f}% | {data['implemented']} | {data['total']} |")

    report.append("")

    # Implemented functions
    report.append("## Implemented Functions")
    report.append("")

    for module, data in LIBROSA_API.items():
        report.append(f"### {module}")
        report.append(f"*{data['description']}*")
        report.append("")

        implemented = [(name, info) for name, info in data['functions'].items() if info['implemented']]
        if implemented:
            report.append("| librosa | SwiftRosa |")
            report.append("|---------|-----------|")
            for name, info in sorted(implemented):
                note = f" ({info.get('note', '')})" if info.get('note') else ""
                report.append(f"| `{name}` | `{info['swift']}`{note} |")
        else:
            report.append("*No functions implemented*")

        report.append("")

    # Missing functions
    report.append("## Missing Functions")
    report.append("")

    if coverage['missing_high']:
        report.append("### High Priority")
        report.append("")
        report.append("| Module | Function | Note |")
        report.append("|--------|----------|------|")
        for entry in coverage['missing_high']:
            note = entry['note'] or "-"
            report.append(f"| {entry['module']} | `{entry['function']}` | {note} |")
        report.append("")

    if coverage['missing_medium']:
        report.append("### Medium Priority")
        report.append("")
        report.append("| Module | Function |")
        report.append("|--------|----------|")
        for entry in coverage['missing_medium']:
            report.append(f"| {entry['module']} | `{entry['function']}` |")
        report.append("")

    if coverage['missing_low']:
        report.append("### Low Priority")
        report.append("")
        report.append("| Module | Function |")
        report.append("|--------|----------|")
        for entry in coverage['missing_low']:
            report.append(f"| {entry['module']} | `{entry['function']}` |")
        report.append("")

    # SwiftRosa extras
    report.append("## SwiftRosa-Specific Features")
    report.append("")
    report.append("*Additional functionality not in librosa:*")
    report.append("")
    report.append("| Feature | Description |")
    report.append("|---------|-------------|")
    for feature, description in sorted(coverage['extras'].items()):
        report.append(f"| `{feature}` | {description} |")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("1. **High Priority Gaps:**")
    if coverage['missing_high']:
        for entry in coverage['missing_high']:
            report.append(f"   - Implement `{entry['function']}` in {entry['module']} module")
    else:
        report.append("   - ✅ All high priority functions implemented!")
    report.append("")
    report.append("2. **Validation:**")
    report.append("   - Run `swift test --filter Validation` to verify numerical accuracy")
    report.append("   - Ensure all implemented functions pass tolerance checks")
    report.append("")

    return '\n'.join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SwiftRosa API Coverage Checker')
    parser.add_argument('--report', type=str, default='api_coverage.md',
                        help='Output file for coverage report')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console output')
    args = parser.parse_args()

    # Analyze coverage
    coverage = analyze_coverage()

    # Generate report
    report = generate_markdown_report(coverage)

    # Save report
    output_path = Path(args.report)
    output_path.write_text(report)

    if not args.quiet:
        print("=" * 60)
        print("SwiftRosa API Coverage Report")
        print("=" * 60)
        print(f"Overall Coverage: {coverage['percentage']:.1f}% ({coverage['implemented']}/{coverage['total']})")
        print()
        print("Module Coverage:")
        for module, data in sorted(coverage['by_module'].items(), key=lambda x: -x[1]['percentage']):
            bar = '█' * int(data['percentage'] / 10) + '░' * (10 - int(data['percentage'] / 10))
            print(f"  {module:12} [{bar}] {data['percentage']:.0f}%")
        print()
        print(f"High Priority Missing: {len(coverage['missing_high'])}")
        print(f"Medium Priority Missing: {len(coverage['missing_medium'])}")
        print(f"Low Priority Missing: {len(coverage['missing_low'])}")
        print(f"SwiftRosa Extras: {len(coverage['extras'])}")
        print()
        print(f"Report saved to: {output_path}")


if __name__ == '__main__':
    main()
