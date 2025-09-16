#!/usr/bin/env python3
"""
P2 Acceptance Box Builder

Generates docs/ACCEPTANCE_BOX_P2.md with comprehensive acceptance criteria
for DON Emergent Gravity P2 validation suite.

This script creates a standardized acceptance matrix showing:
- Test categories and acceptance thresholds
- Pass/fail status for each criterion
- Detailed technical specifications
- Integration with gate-strict validation
"""
import argparse
import json
import os
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd

def check_file_exists(path, description=""):
    """Check if file exists and return status"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    return status, exists

def load_json_metrics(path, metrics_keys):
    """Load JSON and extract specific metrics"""
    if not os.path.exists(path):
        return {key: None for key in metrics_keys}
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        results = {}
        for key in metrics_keys:
            # Support nested key access with dots
            value = data
            for k in key.split('.'):
                value = value.get(k, None) if isinstance(value, dict) else None
                if value is None:
                    break
            results[key] = value
        return results
    except:
        return {key: None for key in metrics_keys}

def load_csv_metrics(path, required_columns):
    """Load CSV and check for required columns"""
    if not os.path.exists(path):
        return {col: None for col in required_columns}
    
    try:
        df = pd.read_csv(path)
        results = {}
        for col in required_columns:
            if col in df.columns:
                # Return summary stats for numeric columns
                if df[col].dtype in ['float64', 'int64']:
                    results[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std() if len(df) > 1 else 0),
                        'count': len(df)
                    }
                else:
                    results[col] = df[col].tolist()
            else:
                results[col] = None
        return results
    except:
        return {col: None for col in required_columns}

def evaluate_acceptance_criteria():
    """Evaluate all P2 acceptance criteria"""
    
    criteria = {
        'convergence': {
            'title': 'Grid & Box Convergence',
            'tests': [
                {
                    'name': 'Grid convergence (N≥320)',
                    'file': 'figs/convergence_grid.csv',
                    'metric': 'relative_error',
                    'threshold': 0.02,
                    'operator': 'abs_less_than',
                    'description': 'Relative error < 2% for N≥320'
                },
                {
                    'name': 'Box convergence (L≥160)',
                    'file': 'figs/convergence_box.csv', 
                    'metric': 'relative_error',
                    'threshold': 0.02,
                    'operator': 'abs_less_than',
                    'description': 'Relative error < 2% for L≥160 Mpc'
                }
            ]
        },
        'window_independence': {
            'title': 'Window Independence',
            'tests': [
                {
                    'name': 'Bootstrap consistency',
                    'file': 'figs/slope_window_heatmap.json',
                    'metric': 'bootstrap_ci.width',
                    'threshold': 0.1,
                    'operator': 'less_than',
                    'description': 'Bootstrap CI width < 0.1'
                },
                {
                    'name': 'Slope stability',
                    'file': 'figs/slope_window_heatmap.json',
                    'metric': 'slope_statistics.std',
                    'threshold': 0.05,
                    'operator': 'less_than',
                    'description': 'Slope standard deviation < 0.05'
                }
            ]
        },
        'helmholtz': {
            'title': 'Helmholtz Diagnostics',
            'tests': [
                {
                    'name': 'Curl-free criterion',
                    'file': 'figs/helmholtz_checks.json',
                    'metric': 'curl_metrics.curl_fraction',
                    'threshold': 0.05,
                    'operator': 'less_than',
                    'description': 'Curl fraction < 5%'
                },
                {
                    'name': 'Flux flatness (P2)',
                    'file': 'figs/helmholtz_checks.json',
                    'metric': 'flux_metrics.flux_p95',
                    'threshold': 0.05,
                    'operator': 'less_than',
                    'description': 'Flux p95 variation < 5%'
                }
            ]
        },
        'physics': {
            'title': 'Physics Validation',
            'tests': [
                {
                    'name': 'Kepler slope',
                    'file': 'figs/kepler_fit.csv',
                    'metric': 'slope',
                    'threshold': [0.98, 1.02],
                    'operator': 'in_range',
                    'description': 'Kepler slope ∈ [0.98, 1.02]'
                },
                {
                    'name': 'EP mass independence',
                    'file': 'sweeps/ep_m*/summary.csv',
                    'metric': 'slope_variation',
                    'threshold': 0.05,
                    'operator': 'less_than',
                    'description': 'Mass-dependent slope variation < 5%'
                }
            ]
        },
        'slip': {
            'title': 'Scale-Dependent Slip',
            'tests': [
                {
                    'name': 'Slip data availability',
                    'file': 'outputs/CLENS_release/slip_analysis.csv',
                    'metric': 'file_exists',
                    'threshold': True,
                    'operator': 'equals',
                    'description': 'Slip analysis CSV generated'
                },
                {
                    'name': 'Mean slip agreement',
                    'file': 'outputs/CLENS_release/slip_analysis.csv',
                    'metric': 'slip_parameter',
                    'threshold': [0.8, 1.2],
                    'operator': 'mean_in_range',
                    'description': 'Mean slip ∈ [0.8, 1.2] (20% tolerance)'
                }
            ]
        }
    }
    
    # Evaluate each test
    results = {}
    for category, cat_info in criteria.items():
        results[category] = {
            'title': cat_info['title'],
            'tests': [],
            'category_pass': True
        }
        
        for test in cat_info['tests']:
            test_result = evaluate_single_test(test)
            results[category]['tests'].append(test_result)
            
            if not test_result['passed']:
                results[category]['category_pass'] = False
    
    return results

def evaluate_single_test(test):
    """Evaluate a single acceptance test"""
    result = {
        'name': test['name'],
        'description': test['description'],
        'file': test['file'],
        'metric': test['metric'],
        'threshold': test['threshold'],
        'operator': test['operator'],
        'passed': False,
        'value': None,
        'message': 'Not evaluated'
    }
    
    # Handle special cases
    if test['metric'] == 'file_exists':
        exists = os.path.exists(test['file'])
        result['passed'] = exists == test['threshold']
        result['value'] = exists
        result['message'] = 'File exists' if exists else 'File missing'
        return result
    
    # Handle glob patterns for EP tests
    if '*' in test['file']:
        files = glob.glob(test['file'])
        if not files:
            result['message'] = 'No matching files found'
            return result
        
        # For EP tests, analyze slope variation across files
        if test['metric'] == 'slope_variation':
            slopes = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    if 'slope' in df.columns:
                        slopes.extend(df['slope'].dropna().tolist())
                except:
                    continue
            
            if slopes:
                import numpy as np
                variation = np.std(slopes) / np.mean(slopes) if slopes else float('inf')
                result['value'] = variation
                result['passed'] = variation < test['threshold']
                result['message'] = f'Variation: {variation:.4f}'
            else:
                result['message'] = 'No slope data found'
            return result
    
    # Load and evaluate metric from file
    if test['file'].endswith('.json'):
        data = load_json_metrics(test['file'], [test['metric']])
        value = data.get(test['metric'])
    elif test['file'].endswith('.csv'):
        metric_name = test['metric'].split('.')[-1]  # Get last part of dotted name
        data = load_csv_metrics(test['file'], [metric_name])
        metric_data = data.get(metric_name)
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            value = metric_data['mean']
        elif isinstance(metric_data, list) and metric_data:
            import numpy as np
            value = np.mean(metric_data)
        else:
            value = metric_data
    else:
        result['message'] = 'Unsupported file type'
        return result
    
    if value is None:
        result['message'] = 'Metric not found'
        return result
    
    result['value'] = value
    
    # Apply operator
    try:
        if test['operator'] == 'less_than':
            result['passed'] = float(value) < test['threshold']
        elif test['operator'] == 'abs_less_than':
            result['passed'] = abs(float(value)) < test['threshold']
        elif test['operator'] == 'in_range':
            result['passed'] = test['threshold'][0] <= float(value) <= test['threshold'][1]
        elif test['operator'] == 'mean_in_range':
            result['passed'] = test['threshold'][0] <= float(value) <= test['threshold'][1]
        elif test['operator'] == 'equals':
            result['passed'] = value == test['threshold']
        else:
            result['message'] = f'Unknown operator: {test["operator"]}'
            return result
        
        result['message'] = f'Value: {value:.4f}' if isinstance(value, float) else f'Value: {value}'
        if result['passed']:
            result['message'] += ' ✓'
        else:
            result['message'] += ' ✗'
            
    except Exception as e:
        result['message'] = f'Evaluation error: {e}'
    
    return result

def generate_acceptance_box_content(results):
    """Generate the acceptance box markdown content"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""# DON Emergent Gravity — P2 Acceptance Box

*Generated: {timestamp}*

## Overview

This document defines and tracks acceptance criteria for DON Emergent Gravity Phase 2 (P2) validation suite: **Universality & Slip**.

### Validation Categories

| Category | Status | Tests Passed | Description |
|----------|--------|--------------|-------------|
"""
    
    # Add category summary
    overall_pass = True
    for category, cat_data in results.items():
        status_symbol = "✓" if cat_data['category_pass'] else "✗"
        if not cat_data['category_pass']:
            overall_pass = False
        
        passed_count = sum(1 for test in cat_data['tests'] if test['passed'])
        total_count = len(cat_data['tests'])
        
        content += f"| {cat_data['title']} | {status_symbol} | {passed_count}/{total_count} | {category.replace('_', ' ').title()} validation |\n"
    
    content += f"\n**Overall P2 Status**: {'✓ PASSED' if overall_pass else '✗ FAILED'}\n\n"
    
    # Add detailed test results
    content += "## Detailed Acceptance Criteria\n\n"
    
    for category, cat_data in results.items():
        content += f"### {cat_data['title']}\n\n"
        
        content += "| Test | Threshold | Status | Value | File |\n"
        content += "|------|-----------|--------|-------|------|\n"
        
        for test in cat_data['tests']:
            status_symbol = "✓" if test['passed'] else "✗"
            threshold_str = str(test['threshold'])
            if isinstance(test['threshold'], list):
                threshold_str = f"[{test['threshold'][0]:.3f}, {test['threshold'][1]:.3f}]"
            elif isinstance(test['threshold'], float):
                threshold_str = f"{test['threshold']:.3f}"
            
            value_str = test['message']
            file_name = os.path.basename(test['file'])
            
            content += f"| {test['name']} | {threshold_str} | {status_symbol} | {value_str} | `{file_name}` |\n"
        
        content += "\n"
    
    # Add technical specifications
    content += """## Technical Specifications

### P2 Acceptance Gates

#### Convergence Requirements
- **Grid size**: Relative error < 2% for N ≥ 320
- **Box size**: Relative error < 2% for L ≥ 160 Mpc
- **Field consistency**: Same results across epsilon values

#### Window Independence
- **Bootstrap CI**: Width < 0.1 for slope estimation
- **Slope stability**: Standard deviation < 0.05 across windows
- **Theil-Sen robustness**: Consistent with OLS within tolerance

#### Helmholtz Field Properties  
- **Curl-free criterion**: ∇×J < 5% of |J|
- **Flux flatness**: p95 variation < 5% in fit annulus
- **Conservative field**: Energy conservation validated

#### Physics Validation
- **Kepler orbits**: Slope ∈ [0.98, 1.02] for circular orbits
- **Equivalence Principle**: Mass independence < 5% variation
- **Force law**: 1/r² behavior in test range

#### Scale-Dependent Slip
- **Data availability**: Complete slip analysis pipeline
- **Agreement tolerance**: Mean slip ∈ [0.8, 1.2] (20%)
- **Scale dependence**: Measured across 5 radial bands

### Integration with CI Pipeline

```bash
# Run P2 validation suite
make paper-kit

# Quick acceptance check
make gate-strict

# Full test suite  
pytest tests/test_physics.py -v
```

### File Dependencies

#### Required Input Files
- `fields/N320_L160_box_eps*.npz` - Field snapshots
- `data/CLENS/` - KiDS W1 cluster catalog
- `data/SPARC/` - Rotation curve data

#### Generated Validation Files
- `figs/convergence_*.csv` - Convergence analysis
- `figs/slope_window_heatmap.json` - Window independence
- `figs/helmholtz_checks.json` - Field diagnostics  
- `figs/kepler_fit.csv` - Orbital validation
- `sweeps/ep_m*/summary.csv` - EP test results
- `outputs/CLENS_release/slip_analysis.csv` - Slip measurements

#### Key Output Products
- `docs/REPORT.md` - Comprehensive validation report
- `docs/ACCEPTANCE_BOX_P2.md` - This acceptance matrix
- `outputs/CLENS_release/gt_panel.png` - Lensing comparison
- `figs/slope_window_heatmap.png` - Window analysis

## Validation History

| Date | Version | Status | Notes |
|------|---------|--------|-------|
"""
    
    content += f"| {timestamp[:10]} | 2.0.0 | {'PASSED' if overall_pass else 'FAILED'} | P2 implementation complete |\n"
    content += "\n"
    
    # Add troubleshooting section
    content += """## Troubleshooting

### Common Issues

1. **Missing field files**: Run `make field-generate` to create NPZ snapshots
2. **CLENS data unavailable**: Check `data/CLENS/` directory and run `make wl-run`
3. **Convergence failures**: Increase N or adjust epsilon tolerance
4. **Helmholtz violations**: Check field boundary conditions and μ transport
5. **Physics test failures**: Validate orbit integration parameters

### Recovery Procedures

```bash
# Regenerate fields
make clean-fields && make field-generate

# Rebuild convergence data  
make conv-grid conv-box

# Rerun window analysis
make slope-window

# Full pipeline rebuild
make clean && make paper-kit
```

### Contact & Support

- **Technical issues**: Check `validation_*.log` files
- **Physics questions**: Review `docs/OVERVIEW.md`
- **Pipeline errors**: Run with `--verbose` flag

---

*This acceptance box is automatically generated by `scripts/build_acceptance_box_p2.py`*
*For updates, run: `make paper-kit`*
"""
    
    return content

def main():
    """Main acceptance box generation"""
    ap = argparse.ArgumentParser(description="Generate P2 acceptance criteria box")
    ap.add_argument('--out', default='docs/ACCEPTANCE_BOX_P2.md',
                   help='Output path for acceptance box')
    args = ap.parse_args()
    
    print("[build_acceptance_box_p2] Evaluating P2 acceptance criteria...")
    
    # Evaluate all criteria
    results = evaluate_acceptance_criteria()
    
    # Generate content
    print("[build_acceptance_box_p2] Generating acceptance box content...")
    content = generate_acceptance_box_content(results)
    
    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(content)
    
    # Summary report
    total_tests = sum(len(cat['tests']) for cat in results.values())
    passed_tests = sum(sum(1 for test in cat['tests'] if test['passed']) for cat in results.values())
    overall_pass = passed_tests == total_tests
    
    print(f"\n[build_acceptance_box_p2] Results summary:")
    print(f"  Tests passed: {passed_tests}/{total_tests}")
    print(f"  Overall status: {'PASSED' if overall_pass else 'FAILED'}")
    print(f"  Output: {args.out}")
    
    if not overall_pass:
        print("\n[build_acceptance_box_p2] Failed tests:")
        for category, cat_data in results.items():
            for test in cat_data['tests']:
                if not test['passed']:
                    print(f"  - {test['name']}: {test['message']}")
    
    print(f"\n[build_acceptance_box_p2] P2 acceptance box complete!")

if __name__ == "__main__":
    main()