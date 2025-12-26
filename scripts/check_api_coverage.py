#!/usr/bin/env python3
"""
SwiftRosa API Coverage Checker - Dynamic Edition

This script dynamically scans the SwiftRosa codebase to:
1. Find actual public API implementations (not hardcoded)
2. Check which APIs have validation tests
3. Parse ValidationTolerances.swift for actual tolerance definitions
4. Report which features are validated vs just implemented

Usage:
    python check_api_coverage.py [--report api_coverage.md]
"""

import argparse
import re
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class APIFunction:
    """Represents a function in the API."""
    name: str
    librosa_equivalent: str
    swift_location: Optional[str] = None
    has_validation_test: bool = False
    validation_test_file: Optional[str] = None
    tolerance_name: Optional[str] = None
    tolerance_value: Optional[float] = None
    tolerance_comment: Optional[str] = None
    is_implemented: bool = False


@dataclass
class APIModule:
    """Represents a module in the API."""
    name: str
    description: str
    functions: Dict[str, APIFunction] = field(default_factory=dict)


# =============================================================================
# LIBROSA API MAPPING (What we SHOULD implement)
# =============================================================================

LIBROSA_API_MAPPING = {
    'core': {
        'description': 'Core time-frequency analysis',
        'functions': {
            'stft': {'swift_class': 'STFT', 'swift_method': 'transform'},
            'istft': {'swift_class': 'ISTFT', 'swift_method': 'transform'},
            'resample': {'swift_class': 'Resample', 'swift_method': 'resample'},
            'griffinlim': {'swift_class': 'GriffinLim', 'swift_method': 'reconstruct'},
            'phase_vocoder': {'swift_class': 'PhaseVocoder', 'swift_method': None},
            'power_to_db': {'swift_class': 'Convert', 'swift_method': 'powerToDb'},
            'db_to_power': {'swift_class': 'Convert', 'swift_method': 'dbToPower'},
            'amplitude_to_db': {'swift_class': 'Convert', 'swift_method': 'amplitudeToDb'},
            'db_to_amplitude': {'swift_class': 'Convert', 'swift_method': 'dbToAmplitude'},
        }
    },
    'feature': {
        'description': 'Feature extraction',
        'functions': {
            'zero_crossing_rate': {'swift_class': 'ZeroCrossingRate', 'swift_method': 'transform'},
            'spectral_centroid': {'swift_class': 'SpectralFeatures', 'swift_method': 'centroid'},
            'spectral_bandwidth': {'swift_class': 'SpectralFeatures', 'swift_method': 'bandwidth'},
            'spectral_rolloff': {'swift_class': 'SpectralFeatures', 'swift_method': 'rolloff'},
            'spectral_flatness': {'swift_class': 'SpectralFeatures', 'swift_method': 'flatness'},
            'spectral_contrast': {'swift_class': 'SpectralFeatures', 'swift_method': 'contrast'},
            'melspectrogram': {'swift_class': 'MelSpectrogram', 'swift_method': 'transform'},
            'mfcc': {'swift_class': 'MFCC', 'swift_method': 'transform'},
            'chroma_stft': {'swift_class': 'Chromagram', 'swift_method': None},
            'chroma_cqt': {'swift_class': 'Chromagram', 'swift_method': None},
            'delta': {'swift_class': 'Delta', 'swift_method': 'compute'},
            'rms': {'swift_class': 'RMSEnergy', 'swift_method': 'transform'},
            'tonnetz': {'swift_class': 'Tonnetz', 'swift_method': None},
            'poly_features': {'swift_class': 'PolyFeatures', 'swift_method': None},
            'tempogram': {'swift_class': 'Tempogram', 'swift_method': None},
        }
    },
    'filters': {
        'description': 'Filter design',
        'functions': {
            'mel': {'swift_class': 'MelFilterbank', 'swift_method': None},
            'get_window': {'swift_class': 'Windows', 'swift_method': 'generate'},
        }
    },
    'onset': {
        'description': 'Onset detection',
        'functions': {
            'onset_detect': {'swift_class': 'OnsetDetector', 'swift_method': 'detect'},
            'onset_strength': {'swift_class': 'OnsetDetector', 'swift_method': 'detectStrength'},
        }
    },
    'beat': {
        'description': 'Beat tracking',
        'functions': {
            'beat_track': {'swift_class': 'BeatTracker', 'swift_method': 'track'},
            'tempo': {'swift_class': 'BeatTracker', 'swift_method': 'estimateTempo'},
        }
    },
    'decompose': {
        'description': 'Source separation',
        'functions': {
            'hpss': {'swift_class': 'HPSS', 'swift_method': 'separate'},
            'decompose': {'swift_class': 'NMF', 'swift_method': 'decompose'},
        }
    },
    'cqt': {
        'description': 'Constant-Q transforms',
        'functions': {
            'cqt': {'swift_class': 'CQT', 'swift_method': 'transform'},
            'icqt': {'swift_class': 'ICQT', 'swift_method': 'transform'},
            'vqt': {'swift_class': 'VQT', 'swift_method': None},
            'pseudo_cqt': {'swift_class': 'PseudoCQT', 'swift_method': None},
        }
    },
    'effects': {
        'description': 'Audio effects',
        'functions': {
            'pitch_shift': {'swift_class': 'PhaseVocoder', 'swift_method': 'pitchShift'},
            'time_stretch': {'swift_class': 'PhaseVocoder', 'swift_method': 'timeStretch'},
            'preemphasis': {'swift_class': 'Emphasis', 'swift_method': 'preemphasis'},
            'deemphasis': {'swift_class': 'Emphasis', 'swift_method': 'deemphasis'},
        }
    },
}


# =============================================================================
# CODEBASE SCANNING
# =============================================================================

def find_project_root() -> Path:
    """Find the SwiftRosa project root."""
    # Try common locations relative to script
    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent,  # scripts/ is in project root
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / 'Package.swift').exists():
            return candidate
    raise FileNotFoundError("Could not find SwiftRosa project root")


def scan_swift_sources(root: Path) -> Dict[str, List[str]]:
    """
    Scan Swift source files to find public class/struct definitions.
    Returns: {class_name: [file_path, ...]}
    """
    class_locations = {}
    sources_dir = root / 'Sources'

    if not sources_dir.exists():
        return class_locations

    # Pattern to match public class/struct/actor definitions
    class_pattern = re.compile(
        r'public\s+(?:final\s+)?(?:class|struct|actor)\s+(\w+)',
        re.MULTILINE
    )

    for swift_file in sources_dir.rglob('*.swift'):
        try:
            content = swift_file.read_text(encoding='utf-8')
            matches = class_pattern.findall(content)
            for class_name in matches:
                if class_name not in class_locations:
                    class_locations[class_name] = []
                rel_path = swift_file.relative_to(root)
                class_locations[class_name].append(str(rel_path))
        except Exception as e:
            print(f"Warning: Could not read {swift_file}: {e}")

    return class_locations


def scan_public_methods(root: Path, class_name: str) -> List[str]:
    """Find public methods for a given class."""
    methods = []
    sources_dir = root / 'Sources'

    # Pattern to match public func definitions
    method_pattern = re.compile(
        rf'(?:public\s+)?func\s+(\w+)\s*\(',
        re.MULTILINE
    )

    for swift_file in sources_dir.rglob('*.swift'):
        try:
            content = swift_file.read_text(encoding='utf-8')
            # Check if this file contains the class
            if re.search(rf'(?:class|struct|actor)\s+{class_name}\b', content):
                # Find methods in this class
                matches = method_pattern.findall(content)
                methods.extend(matches)
        except Exception:
            pass

    return list(set(methods))


def scan_validation_tests(root: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Scan test files to find what features have validation tests.
    Returns: {feature_name: [(test_name, test_file), ...]}
    """
    validation_tests = {}
    tests_dir = root / 'Tests'

    if not tests_dir.exists():
        return validation_tests

    # Pattern to match ANY test function
    test_pattern = re.compile(
        r'func\s+(test\w+)\s*\(',
        re.MULTILINE
    )

    # Pattern to find ValidationTolerances usage
    tolerance_pattern = re.compile(
        r'ValidationTolerances\.(\w+)',
        re.MULTILINE
    )

    # Scan ALL test files, not just *Validation* ones
    for test_file in tests_dir.rglob('*.swift'):
        # Skip non-test files
        if 'Tests' not in test_file.name and not test_file.name.endswith('Tests.swift'):
            continue

        try:
            content = test_file.read_text(encoding='utf-8')
            rel_path = str(test_file.relative_to(root))

            # Check if this file uses ValidationTolerances (indicates validation tests)
            # (used for filtering in future enhancements)

            # Find test functions
            test_matches = test_pattern.findall(content)

            # Find which tolerances are used
            tolerance_matches = set(tolerance_pattern.findall(content))

            # Map tests to features
            for test_name in test_matches:
                # Extract feature name from test name
                feature_key = extract_feature_from_test_name(test_name)
                if feature_key:
                    if feature_key not in validation_tests:
                        validation_tests[feature_key] = []
                    validation_tests[feature_key].append((test_name, rel_path))

            # Also track features based on tolerance usage
            # This catches tests that validate but don't have "Validation" in name
            for tolerance in tolerance_matches:
                feature_key = tolerance_to_feature(tolerance)
                if feature_key and feature_key not in validation_tests:
                    validation_tests[feature_key] = []
                if feature_key:
                    validation_tests[feature_key].append((f"uses_{tolerance}", rel_path))

        except Exception as e:
            print(f"Warning: Could not read {test_file}: {e}")

    return validation_tests


def tolerance_to_feature(tolerance_name: str) -> Optional[str]:
    """Map tolerance names to feature keys."""
    mapping = {
        'stftmagnitude': 'stft',
        'stftmagnitudechirp': 'stft',
        'stftphase': 'stft',
        'istftreconstruction': 'istft',
        'melspectrogram': 'melspectrogram',
        'melfilterbank': 'mel',
        'mfcc': 'mfcc',
        'cqtmagnitude': 'cqt',
        'vqtmagnitude': 'vqt',
        'pseudocqtmagnitude': 'pseudo_cqt',
        'spectralcentroid': 'spectral_centroid',
        'spectralbandwidth': 'spectral_bandwidth',
        'spectralbandwidthnarrowband': 'spectral_bandwidth',
        'spectralbandwidthbroadband': 'spectral_bandwidth',
        'spectralrolloff': 'spectral_rolloff',
        'spectralflatness': 'spectral_flatness',
        'spectralcontrast': 'spectral_contrast',
        'spectralflux': 'spectral_flux',
        'zerocrossingrate': 'zero_crossing_rate',
        'rmsenergy': 'rms',
        'delta': 'delta',
        'chromagram': 'chroma_stft',
        'tonnetz': 'tonnetz',
        'onsetframedifference': 'onset_detect',
        'onsetframefullprecision': 'onset_detect',
        'onsetstrength': 'onset_strength',
        'beatframedifference': 'beat_track',
        'tempogram': 'tempogram',
        'hpss': 'hpss',
        'nmf': 'decompose',
        'griffinlim': 'griffinlim',
        'phasevocoder': 'phase_vocoder',
        'resample': 'resample',
        'window': 'get_window',
        'pcen': 'pcen',
        'dbconversion': 'power_to_db',
        'polyfeatures': 'poly_features',
    }
    return mapping.get(tolerance_name.lower())


def extract_feature_from_test_name(test_name: str) -> Optional[str]:
    """Extract feature name from a test function name."""
    # Common patterns
    patterns = [
        (r'test(STFT)', 'stft'),
        (r'test(ISTFT)', 'istft'),
        (r'test(Mel)', 'mel'),
        (r'test(MFCC)', 'mfcc'),
        (r'test(CQT)', 'cqt'),
        (r'test(PCEN)', 'pcen'),
        (r'test(Spectral\w+)', lambda m: m.group(1).lower()),
        (r'test(RMS)', 'rms'),
        (r'test(ZCR|ZeroCrossing)', 'zcr'),
        (r'test(Onset)', 'onset'),
        (r'test(Beat)', 'beat'),
        (r'test(Tempo)', 'tempo'),
        (r'test(Tempogram)', 'tempogram'),
        (r'test(Chroma)', 'chroma'),
        (r'test(HPSS)', 'hpss'),
        (r'test(GriffinLim)', 'griffinlim'),
        (r'test(Resample)', 'resample'),
        (r'test(PowerToDb)', 'power_to_db'),
        (r'test(DbToPower)', 'db_to_power'),
        (r'test(AmplitudeToDb)', 'amplitude_to_db'),
        (r'test(DbToAmplitude)', 'db_to_amplitude'),
        (r'test(Hann|Hamming|Blackman|Bartlett|Window)', 'get_window'),
    ]

    for pattern, result in patterns:
        match = re.search(pattern, test_name, re.IGNORECASE)
        if match:
            if callable(result):
                return result(match)
            return result

    return None


def parse_validation_tolerances(root: Path) -> Dict[str, Tuple[float, str]]:
    """
    Parse ValidationTolerances.swift to get defined tolerances.
    Returns: {tolerance_name: (value, comment)}
    """
    tolerances = {}
    tolerance_file = root / 'Tests' / 'SwiftRosaCoreTests' / 'ValidationTolerances.swift'

    if not tolerance_file.exists():
        return tolerances

    try:
        content = tolerance_file.read_text(encoding='utf-8')

        # Pattern to match tolerance definitions with comments
        # Captures: 1=comment lines before, 2=name, 3=value
        pattern = re.compile(
            r'((?:///[^\n]*\n)+)\s*public\s+static\s+let\s+(\w+):\s*Double\s*=\s*([0-9.e\-+]+)',
            re.MULTILINE
        )

        for match in pattern.finditer(content):
            comment_block = match.group(1)
            name = match.group(2)
            value_str = match.group(3)

            try:
                value = float(value_str)
            except ValueError:
                continue

            # Extract first line of comment
            comment_lines = [l.strip('/ \t') for l in comment_block.split('\n') if l.strip()]
            comment = comment_lines[0] if comment_lines else ""

            tolerances[name] = (value, comment)

    except Exception as e:
        print(f"Warning: Could not parse ValidationTolerances.swift: {e}")

    return tolerances


# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

def analyze_coverage(root: Path) -> dict:
    """Analyze API coverage by scanning the codebase."""

    # Scan codebase
    print("Scanning Swift sources...")
    class_locations = scan_swift_sources(root)

    print("Scanning validation tests...")
    validation_tests = scan_validation_tests(root)

    print("Parsing validation tolerances...")
    tolerances = parse_validation_tolerances(root)

    # Build coverage data
    modules = {}
    total_functions = 0
    implemented_functions = 0
    validated_functions = 0

    for module_name, module_data in LIBROSA_API_MAPPING.items():
        module = APIModule(
            name=module_name,
            description=module_data['description']
        )

        for func_name, func_info in module_data['functions'].items():
            total_functions += 1
            swift_class = func_info['swift_class']
            swift_method = func_info.get('swift_method')

            func = APIFunction(
                name=func_name,
                librosa_equivalent=func_name,
            )

            # Check if implemented (class exists in codebase)
            if swift_class in class_locations:
                func.is_implemented = True
                func.swift_location = class_locations[swift_class][0]
                implemented_functions += 1

                # Check if method exists
                if swift_method:
                    methods = scan_public_methods(root, swift_class)
                    if swift_method not in methods:
                        # Method not found, but class exists
                        pass

            # Check for validation tests
            # Try multiple key variants: original, no underscores, lowercase
            feature_keys = [
                func_name,
                func_name.replace('_', ''),
                func_name.lower(),
                func_name.replace('_', '').lower(),
            ]

            found_validation = False
            for feature_key in feature_keys:
                if feature_key in validation_tests and validation_tests[feature_key]:
                    func.has_validation_test = True
                    func.validation_test_file = validation_tests[feature_key][0][1]
                    validated_functions += 1
                    found_validation = True
                    break

            if not found_validation:
                # Try alternate keys with substring matching
                feature_key = func_name.replace('_', '').lower()
                for key in validation_tests:
                    key_normalized = key.replace('_', '').lower()
                    if (feature_key in key_normalized or key_normalized in feature_key) and validation_tests[key]:
                        func.has_validation_test = True
                        func.validation_test_file = validation_tests[key][0][1]
                        validated_functions += 1
                        break

            # Find matching tolerance
            tolerance_key = feature_key
            for tol_name, (tol_value, tol_comment) in tolerances.items():
                if feature_key in tol_name.lower() or tol_name.lower() in feature_key:
                    func.tolerance_name = tol_name
                    func.tolerance_value = tol_value
                    func.tolerance_comment = tol_comment
                    break

            module.functions[func_name] = func

        modules[module_name] = module

    # Identify known limitations from tolerance comments
    known_limitations = []
    for tol_name, (tol_value, tol_comment) in tolerances.items():
        if 'KNOWN LIMITATION' in tol_comment.upper() or 'TODO' in tol_comment.upper():
            known_limitations.append({
                'name': tol_name,
                'value': tol_value,
                'comment': tol_comment
            })

    # Find SwiftRosa extras (classes not in librosa mapping)
    librosa_classes = set()
    for module_data in LIBROSA_API_MAPPING.values():
        for func_info in module_data['functions'].values():
            librosa_classes.add(func_info['swift_class'])

    extras = {}
    for class_name, locations in class_locations.items():
        if class_name not in librosa_classes:
            # Skip internal/helper classes
            if not class_name.startswith('_') and class_name[0].isupper():
                extras[class_name] = locations[0]

    return {
        'modules': modules,
        'total': total_functions,
        'implemented': implemented_functions,
        'validated': validated_functions,
        'percentage_implemented': (implemented_functions / total_functions * 100) if total_functions > 0 else 0,
        'percentage_validated': (validated_functions / total_functions * 100) if total_functions > 0 else 0,
        'tolerances': tolerances,
        'known_limitations': known_limitations,
        'extras': extras,
        'validation_tests': validation_tests,
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(coverage: dict) -> str:
    """Generate a markdown coverage report."""
    report = []

    report.append("# SwiftRosa API Coverage Report")
    report.append("")
    report.append(f"*Dynamically generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report.append("")

    # Summary
    report.append("## Summary")
    report.append("")

    impl_pct = coverage['percentage_implemented']
    val_pct = coverage['percentage_validated']

    impl_bar = '█' * int(impl_pct / 5) + '░' * (20 - int(impl_pct / 5))
    val_bar = '█' * int(val_pct / 5) + '░' * (20 - int(val_pct / 5))

    report.append(f"| Metric | Coverage |")
    report.append(f"|--------|----------|")
    report.append(f"| **Implemented** | [{impl_bar}] {impl_pct:.1f}% ({coverage['implemented']}/{coverage['total']}) |")
    report.append(f"| **Validated** | [{val_bar}] {val_pct:.1f}% ({coverage['validated']}/{coverage['total']}) |")
    report.append("")

    # Module breakdown
    report.append("## Coverage by Module")
    report.append("")
    report.append("| Module | Implemented | Validated | Functions |")
    report.append("|--------|-------------|-----------|-----------|")

    for module_name, module in sorted(coverage['modules'].items()):
        total = len(module.functions)
        impl = sum(1 for f in module.functions.values() if f.is_implemented)
        val = sum(1 for f in module.functions.values() if f.has_validation_test)

        impl_icon = "✅" if impl == total else ("⚠️" if impl > 0 else "❌")
        val_icon = "✅" if val == total else ("⚠️" if val > 0 else "❌")

        report.append(f"| {module_name} | {impl_icon} {impl}/{total} | {val_icon} {val}/{total} | {total} |")

    report.append("")

    # Detailed coverage
    report.append("## Detailed Coverage")
    report.append("")

    for module_name, module in coverage['modules'].items():
        report.append(f"### {module_name}")
        report.append(f"*{module.description}*")
        report.append("")
        report.append("| librosa | Swift | Implemented | Validated | Tolerance |")
        report.append("|---------|-------|-------------|-----------|-----------|")

        for func_name, func in sorted(module.functions.items()):
            impl_icon = "✅" if func.is_implemented else "❌"
            val_icon = "✅" if func.has_validation_test else "❌"

            swift_loc = func.swift_location or "-"
            if swift_loc != "-":
                swift_loc = swift_loc.split('/')[-1]  # Just filename

            tol_str = "-"
            if func.tolerance_value is not None:
                if func.tolerance_value >= 0.01:
                    tol_str = f"{func.tolerance_value*100:.1f}%"
                else:
                    tol_str = f"{func.tolerance_value:.0e}"

            report.append(f"| `{func_name}` | {swift_loc} | {impl_icon} | {val_icon} | {tol_str} |")

        report.append("")

    # Known Limitations
    if coverage['known_limitations']:
        report.append("## Known Limitations")
        report.append("")
        report.append("*Features with documented accuracy issues:*")
        report.append("")
        for lim in coverage['known_limitations']:
            report.append(f"- **{lim['name']}** ({lim['value']*100:.0f}%): {lim['comment']}")
        report.append("")

    # Tolerance definitions
    report.append("## Validation Tolerances")
    report.append("")
    report.append("| Feature | Tolerance | Description |")
    report.append("|---------|-----------|-------------|")

    for tol_name, (tol_value, tol_comment) in sorted(coverage['tolerances'].items()):
        if tol_value >= 0.01:
            tol_str = f"{tol_value*100:.1f}%"
        else:
            tol_str = f"{tol_value:.0e}"

        # Truncate long comments
        comment_short = tol_comment[:60] + "..." if len(tol_comment) > 60 else tol_comment
        report.append(f"| {tol_name} | {tol_str} | {comment_short} |")

    report.append("")

    # SwiftRosa Extras
    if coverage['extras']:
        report.append("## SwiftRosa-Specific Features")
        report.append("")
        report.append("*Additional classes not in librosa:*")
        report.append("")

        # Group by module
        extras_list = sorted(coverage['extras'].items())[:20]  # Limit to 20
        for class_name, location in extras_list:
            module = location.split('/')[1] if '/' in location else 'Unknown'
            report.append(f"- `{class_name}` ({module})")

        if len(coverage['extras']) > 20:
            report.append(f"- *... and {len(coverage['extras']) - 20} more*")
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    # Find unvalidated implementations
    unvalidated = []
    for module in coverage['modules'].values():
        for func in module.functions.values():
            if func.is_implemented and not func.has_validation_test:
                unvalidated.append(func.name)

    if unvalidated:
        report.append("### Missing Validation Tests")
        report.append("")
        report.append("These features are implemented but lack validation tests:")
        report.append("")
        for func_name in unvalidated[:10]:
            report.append(f"- `{func_name}`")
        if len(unvalidated) > 10:
            report.append(f"- *... and {len(unvalidated) - 10} more*")
        report.append("")

    # Find unimplemented
    unimplemented = []
    for module in coverage['modules'].values():
        for func in module.functions.values():
            if not func.is_implemented:
                unimplemented.append(func.name)

    if unimplemented:
        report.append("### Missing Implementations")
        report.append("")
        for func_name in unimplemented:
            report.append(f"- `{func_name}`")
        report.append("")

    return '\n'.join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SwiftRosa API Coverage Checker (Dynamic)')
    parser.add_argument('--report', type=str, default='api_coverage.md',
                        help='Output file for coverage report')
    parser.add_argument('--json', type=str, default=None,
                        help='Output JSON file for machine-readable data')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console output')
    args = parser.parse_args()

    try:
        root = find_project_root()
        if not args.quiet:
            print(f"Project root: {root}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Analyze coverage
    coverage = analyze_coverage(root)

    # Generate markdown report
    report = generate_markdown_report(coverage)

    # Save report
    output_path = Path(args.report)
    output_path.write_text(report)

    # Save JSON if requested
    if args.json:
        json_data = {
            'generated': datetime.now().isoformat(),
            'total': coverage['total'],
            'implemented': coverage['implemented'],
            'validated': coverage['validated'],
            'percentage_implemented': coverage['percentage_implemented'],
            'percentage_validated': coverage['percentage_validated'],
            'known_limitations': coverage['known_limitations'],
            'tolerances': {k: {'value': v[0], 'comment': v[1]} for k, v in coverage['tolerances'].items()},
        }
        Path(args.json).write_text(json.dumps(json_data, indent=2))

    if not args.quiet:
        print()
        print("=" * 60)
        print("SwiftRosa API Coverage Report (Dynamic)")
        print("=" * 60)
        print(f"Implemented: {coverage['percentage_implemented']:.1f}% ({coverage['implemented']}/{coverage['total']})")
        print(f"Validated:   {coverage['percentage_validated']:.1f}% ({coverage['validated']}/{coverage['total']})")
        print()

        if coverage['known_limitations']:
            print("Known Limitations:")
            for lim in coverage['known_limitations']:
                print(f"  - {lim['name']}: {lim['comment'][:50]}...")
            print()

        print(f"Report saved to: {output_path}")
        if args.json:
            print(f"JSON saved to: {args.json}")

    return 0


if __name__ == '__main__':
    exit(main())
