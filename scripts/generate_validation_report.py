#!/usr/bin/env python3
"""
SwiftRosa Validation Report Generator

This script generates a comprehensive validation report comparing SwiftRosa
against librosa, including:
- API coverage summary
- Test results analysis
- Tolerance tier validation
- Recommendations

Usage:
    python generate_validation_report.py [--output ValidationReport.md]
"""

import argparse
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# =============================================================================
# TOLERANCE TIER DEFINITIONS (mirrors ValidationTolerances.swift)
# =============================================================================

TOLERANCE_TIERS = {
    1: {
        'name': 'Exact Operations',
        'tolerance': 1e-6,
        'description': 'Pure mathematical formulas with no error accumulation',
        'functions': ['window', 'mel_scale', 'frequency_conversion']
    },
    2: {
        'name': 'FFT Operations',
        'tolerance': 1e-5,
        'description': 'Single FFT operation, minimal rounding',
        'functions': ['stft_magnitude', 'stft_phase']
    },
    3: {
        'name': 'Linear Operations',
        'tolerance': 1e-4,
        'description': 'Matrix operations, error grows with dimensions',
        'functions': [
            'istft_reconstruction', 'cqt_magnitude', 'vqt_magnitude',
            'mel_filterbank', 'mel_spectrogram', 'spectral_centroid',
            'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness',
            'spectral_flux', 'zero_crossing_rate', 'rms_energy', 'delta',
            'poly_features'
        ]
    },
    4: {
        'name': 'Log-domain Operations',
        'tolerance': 1e-3,
        'description': 'log/exp amplify small differences',
        'functions': [
            'mfcc', 'pcen', 'db_conversion', 'chromagram', 'tonnetz',
            'tempogram', 'spectral_contrast'
        ]
    },
    5: {
        'name': 'Iterative/Stochastic Operations',
        'tolerance': 0.10,  # 10%
        'description': 'Random initialization, convergence variance',
        'functions': [
            'griffin_lim', 'nmf', 'hpss', 'resample', 'phase_vocoder'
        ]
    },
    6: {
        'name': 'Detection Operations',
        'tolerance': None,  # Frame-based
        'description': 'Approximate by nature, measured in frames or F1 score',
        'functions': ['onset_detect', 'beat_track', 'tempo', 'pitch']
    }
}


# =============================================================================
# TEST RESULT PARSING
# =============================================================================

def run_tests(filter_pattern=None):
    """Run Swift tests and capture output."""
    cmd = ['swift', 'test']
    if filter_pattern:
        cmd.extend(['--filter', filter_pattern])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Test execution timed out"
    except Exception as e:
        return f"Error running tests: {e}"


def parse_test_results(output):
    """Parse test output to extract pass/fail information."""
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'failures': [],
        'summary_lines': []
    }

    # Match test case results
    test_pattern = re.compile(r"Test Case.*'(.*)'.*\((passed|failed)\)")
    failure_pattern = re.compile(r"error:.*XCT.*failed:.*\"(.*)\"")
    summary_pattern = re.compile(r"Executed (\d+) tests?, with (\d+) test(?:s)? skipped and (\d+) failure")

    for line in output.split('\n'):
        test_match = test_pattern.search(line)
        if test_match:
            test_name = test_match.group(1)
            status = test_match.group(2)
            results['total'] += 1
            if status == 'passed':
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append(test_name)

        summary_match = summary_pattern.search(line)
        if summary_match:
            results['summary_lines'].append(line)

    return results


def parse_validation_summary(output):
    """Parse the validation summary block from test output."""
    summary = {}
    in_summary = False
    current_category = None

    for line in output.split('\n'):
        if 'COMPREHENSIVE VALIDATION SUMMARY' in line:
            in_summary = True
            continue

        if in_summary:
            if line.startswith('=' * 20):
                break

            # Parse overall percentage
            overall_match = re.match(r'Overall: (\d+)/(\d+) tests passed \((\d+\.\d+)%\)', line)
            if overall_match:
                summary['overall'] = {
                    'passed': int(overall_match.group(1)),
                    'total': int(overall_match.group(2)),
                    'percentage': float(overall_match.group(3))
                }
                continue

            # Parse category results
            category_match = re.match(r'([✅❌]) (\w+): (\d+)/(\d+)', line)
            if category_match:
                status = '✅' if category_match.group(1) == '✅' else '❌'
                category = category_match.group(2)
                passed = int(category_match.group(3))
                total = int(category_match.group(4))
                summary[category] = {
                    'status': status,
                    'passed': passed,
                    'total': total,
                    'percentage': (passed / total * 100) if total > 0 else 0
                }
                current_category = category
                continue

            # Parse failures within category
            failure_match = re.match(r'   ⚠️ (\w+): (.+)', line)
            if failure_match and current_category:
                if 'failures' not in summary.get(current_category, {}):
                    summary[current_category]['failures'] = []
                summary[current_category]['failures'].append({
                    'signal': failure_match.group(1),
                    'message': failure_match.group(2)
                })

    return summary


# =============================================================================
# API COVERAGE
# =============================================================================

def get_api_coverage():
    """Get API coverage from check_api_coverage.py."""
    try:
        result = subprocess.run(
            ['python3', 'scripts/check_api_coverage.py', '--quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # The script saves to api_coverage.md, parse it
        coverage_file = Path('api_coverage.md')
        if coverage_file.exists():
            content = coverage_file.read_text()
            # Extract summary
            match = re.search(r'\*\*Overall Coverage: (\d+\.\d+)%\*\* \((\d+)/(\d+) functions\)', content)
            if match:
                return {
                    'percentage': float(match.group(1)),
                    'implemented': int(match.group(2)),
                    'total': int(match.group(3))
                }
    except Exception as e:
        print(f"Warning: Could not get API coverage: {e}")

    return {'percentage': 0, 'implemented': 0, 'total': 0}


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(test_output, api_coverage):
    """Generate markdown validation report."""
    test_results = parse_test_results(test_output)
    validation_summary = parse_validation_summary(test_output)

    report = []

    # Header
    report.append("# SwiftRosa Validation Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report validates SwiftRosa as a 1:1 port of librosa by verifying:")
    report.append("1. **API Parity**: All public librosa functions have Swift equivalents")
    report.append("2. **Numerical Accuracy**: Outputs match within justified tolerance levels")
    report.append("")

    # Key Metrics
    report.append("### Key Metrics")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| API Coverage | {api_coverage['percentage']:.1f}% ({api_coverage['implemented']}/{api_coverage['total']}) |")
    if validation_summary.get('overall'):
        report.append(f"| Validation Tests | {validation_summary['overall']['percentage']:.1f}% ({validation_summary['overall']['passed']}/{validation_summary['overall']['total']}) |")
    report.append(f"| Total Tests Run | {test_results['total']} |")
    report.append(f"| Tests Passed | {test_results['passed']} |")
    report.append(f"| Tests Failed | {test_results['failed']} |")
    report.append(f"| Tests Skipped | {test_results['skipped']} |")
    report.append("")

    # Tolerance Tier Validation
    report.append("## Tolerance Tier Validation")
    report.append("")
    report.append("SwiftRosa uses a 6-tier tolerance hierarchy based on operation type:")
    report.append("")

    for tier_num, tier_info in TOLERANCE_TIERS.items():
        tol_str = f"{tier_info['tolerance']:.0e}" if tier_info['tolerance'] else "Frame-based"
        report.append(f"### Tier {tier_num}: {tier_info['name']} ({tol_str})")
        report.append(f"*{tier_info['description']}*")
        report.append("")
        report.append("Functions: " + ", ".join(f"`{f}`" for f in tier_info['functions']))
        report.append("")

    # Validation Results by Category
    if validation_summary:
        report.append("## Validation Results by Category")
        report.append("")
        report.append("| Category | Status | Pass Rate | Notes |")
        report.append("|----------|--------|-----------|-------|")

        for category, data in sorted(validation_summary.items()):
            if category == 'overall':
                continue
            status = data.get('status', '❓')
            passed = data.get('passed', 0)
            total = data.get('total', 0)
            pct = data.get('percentage', 0)
            failures = data.get('failures', [])
            notes = f"{len(failures)} failures" if failures else "All passed"
            report.append(f"| {category} | {status} | {pct:.0f}% ({passed}/{total}) | {notes} |")

        report.append("")

    # Known Issues
    report.append("## Known Issues and Discrepancies")
    report.append("")

    if validation_summary:
        issues = []
        for category, data in validation_summary.items():
            if category == 'overall':
                continue
            for failure in data.get('failures', []):
                issues.append({
                    'category': category,
                    'signal': failure['signal'],
                    'message': failure['message']
                })

        if issues:
            report.append("| Category | Signal | Issue |")
            report.append("|----------|--------|-------|")
            for issue in issues[:20]:  # Limit to 20
                msg = issue['message'][:50] + "..." if len(issue['message']) > 50 else issue['message']
                report.append(f"| {issue['category']} | {issue['signal']} | {msg} |")
            if len(issues) > 20:
                report.append(f"| ... | ... | ({len(issues) - 20} more) |")
            report.append("")
        else:
            report.append("No significant discrepancies detected.")
            report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### High Priority")
    report.append("")
    if api_coverage['percentage'] < 100:
        missing = api_coverage['total'] - api_coverage['implemented']
        report.append(f"1. **Complete API Coverage**: Implement {missing} remaining functions")
    if test_results['failed'] > 0:
        report.append(f"2. **Fix Failing Tests**: {test_results['failed']} tests need attention")
    report.append("")

    report.append("### Validation Methodology")
    report.append("")
    report.append("This validation uses:")
    report.append("- **Reference Data**: Generated from librosa 0.11.0 with identical test signals")
    report.append("- **Test Signals**: Sine waves, chirps, white noise, impulses, click trains, AM-modulated")
    report.append("- **Tolerance Hierarchy**: 6 tiers from 1e-6 (exact) to frame-based (detection)")
    report.append("- **Metrics**: Max absolute error, relative error, correlation, F1 score")
    report.append("")

    # Conclusion
    report.append("## Conclusion")
    report.append("")
    overall_status = "✅" if (api_coverage['percentage'] >= 75 and
                             validation_summary.get('overall', {}).get('percentage', 0) >= 75) else "⚠️"
    report.append(f"{overall_status} **SwiftRosa provides a {'high-fidelity' if overall_status == '✅' else 'partial'} "
                  f"port of librosa** with {api_coverage['percentage']:.0f}% API coverage and "
                  f"{validation_summary.get('overall', {}).get('percentage', 0):.0f}% validation accuracy.")
    report.append("")

    return '\n'.join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate SwiftRosa Validation Report')
    parser.add_argument('--output', type=str, default='ValidationReport.md',
                        help='Output file for validation report')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip running tests (use cached output)')
    parser.add_argument('--filter', type=str, default='Validation',
                        help='Test filter pattern')
    args = parser.parse_args()

    print("=" * 60)
    print("SwiftRosa Validation Report Generator")
    print("=" * 60)

    # Get API coverage
    print("\n1. Checking API coverage...")
    api_coverage = get_api_coverage()
    print(f"   API Coverage: {api_coverage['percentage']:.1f}%")

    # Run tests
    if args.skip_tests:
        print("\n2. Skipping tests (using cached output)...")
        test_output = ""
    else:
        print(f"\n2. Running tests (filter: {args.filter})...")
        test_output = run_tests(args.filter)
        print("   Tests completed.")

    # Generate report
    print("\n3. Generating report...")
    report = generate_report(test_output, api_coverage)

    # Save report
    output_path = Path(args.output)
    output_path.write_text(report)
    print(f"   Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"API Coverage: {api_coverage['percentage']:.1f}%")
    results = parse_test_results(test_output)
    print(f"Tests: {results['passed']}/{results['total']} passed, {results['failed']} failed")
    print(f"Report: {output_path}")


if __name__ == '__main__':
    main()
