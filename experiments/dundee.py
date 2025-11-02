#!/usr/bin/env python3
"""
Python equivalent of dundee.r
Fits mixed-effects models to compute Psychometric Predictive Power (PPP)
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def scale_columns(df, columns):
    """Standardize columns (mean=0, std=1)"""
    scaler = StandardScaler()
    for col in columns:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    return df


def filter_data(data):
    """Apply exclusion criteria from Appendix C"""
    subdata = data.copy()
    
    # Filter conditions (using boolean False, not string 'False')
    subdata = subdata[subdata['time'] > 0]
    subdata = subdata[subdata['has_num'] == False]
    subdata = subdata[subdata['has_num_prev_1'] == False]
    subdata = subdata[subdata['has_punct'] == False]
    subdata = subdata[subdata['has_punct_prev_1'] == False]
    subdata = subdata[subdata['is_first'] == False]
    subdata = subdata[subdata['is_last'] == False]
    subdata = subdata[subdata['pos'] != 'CD']
    
    return subdata


def fit_models(data):
    """
    Fit baseline and test mixed-effects models
    
    Baseline: time ~ freq * length + freq_prev * length_prev + controls + random effects
    Test: Baseline + surprisal features
    """
    
    # Baseline model (Eq. 4 in paper)
    baseline_formula = (
        'time ~ log_gmean_freq * length + '
        'log_gmean_freq_prev_1 * length_prev_1 + '
        'screenN + lineN + segmentN'
    )
    
    # Test model with surprisal
    test_formula = (
        'time ~ surprisals_sum + surprisals_sum_prev_1 + surprisals_sum_prev_2 + '
        'log_gmean_freq * length + '
        'log_gmean_freq_prev_1 * length_prev_1 + '
        'screenN + lineN + segmentN'
    )
    
    print("  Fitting baseline model...")
    # Fit baseline model with random effects for article and subject
    baseline_model = smf.mixedlm(
        baseline_formula,
        data,
        groups=data['article'],
        re_formula='1',
        vc_formula={'subj_id': '0 + C(subj_id)'}
    )
    baseline_result = baseline_model.fit(method='lbfgs', maxiter=1000, reml=False)
    
    print("  Fitting test model with surprisal...")
    # Fit test model
    test_model = smf.mixedlm(
        test_formula,
        data,
        groups=data['article'],
        re_formula='1',
        vc_formula={'subj_id': '0 + C(subj_id)'}
    )
    test_result = test_model.fit(method='lbfgs', maxiter=1000, reml=False)
    
    return baseline_result, test_result


def compute_metrics(baseline_result, test_result, n_obs):
    """
    Compute PPP and related metrics
    
    PPP = (logLik_test - logLik_baseline) / N
    """
    base_loglik = baseline_result.llf
    test_loglik = test_result.llf
    
    # Per-observation log-likelihood
    base_loglik_per_obs = base_loglik / n_obs
    test_loglik_per_obs = test_loglik / n_obs
    
    # PPP: difference in log-likelihood per observation
    delta_loglik = test_loglik_per_obs - base_loglik_per_obs
    
    # Likelihood ratio test
    lr_stat = 2 * (test_loglik - base_loglik)
    df_diff = len(test_result.params) - len(baseline_result.params)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    
    return {
        'base_loglik': base_loglik,
        'test_loglik': test_loglik,
        'base_loglik_per_obs': base_loglik_per_obs,
        'test_loglik_per_obs': test_loglik_per_obs,
        'delta_loglik': delta_loglik,
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'n_obs': n_obs
    }


def save_results(metrics, residuals_df, output_dir):
    """Save likelihood metrics and residuals"""
    
    # Save likelihood.txt
    likelihood_path = os.path.join(output_dir, 'likelihood.txt')
    with open(likelihood_path, 'w') as f:
        f.write(f"linear_fit_logLik: {metrics['test_loglik_per_obs']}\n")
        f.write(f"delta_linear_fit_logLik: {metrics['delta_loglik']}\n")
        f.write(f"delta_linear_fit_chi_p: {metrics['p_value']}\n")
        f.write(f"\n# Additional metrics:\n")
        f.write(f"base_loglik_total: {metrics['base_loglik']}\n")
        f.write(f"test_loglik_total: {metrics['test_loglik']}\n")
        f.write(f"lr_statistic: {metrics['lr_statistic']}\n")
        f.write(f"n_observations: {metrics['n_obs']}\n")
    
    print(f"  Saved: {likelihood_path}")
    
    # Save residuals.txt (CSV format for compatibility)
    residuals_path = os.path.join(output_dir, 'residuals.txt')
    residuals_df.to_csv(residuals_path, index=False)
    print(f"  Saved: {residuals_path}")


def process_scores_file(scores_file, eye_data, base_dir):
    """Process a single scores.csv file"""
    
    print(f"\nProcessing: {scores_file}")
    
    # Load surprisal scores
    scores_data = pd.read_csv(scores_file, sep='\t')
    print(f"  Scores rows: {len(scores_data)}")
    print(f"  Eye data rows: {len(eye_data)}")
    
    # Combine with eye-tracking data
    if len(scores_data) != len(eye_data):
        print(f"  WARNING: Row count mismatch! {len(scores_data)} vs {len(eye_data)}")
        return
    
    data = pd.concat([scores_data, eye_data], axis=1)
    
    # Scale numeric columns
    columns_to_scale = [
        'screenN', 'lineN', 'segmentN', 'length', 'length_prev_1',
        'log_gmean_freq', 'log_gmean_freq_prev_1',
        'surprisals_sum', 'surprisals_sum_prev_1', 
        'surprisals_sum_prev_2', 'surprisals_sum_prev_3'
    ]
    data = scale_columns(data, columns_to_scale)
    
    # Keep raw surprisal for reference
    data['surprisals_sum_raw'] = scores_data['surprisals_sum'].copy()
    
    # Apply filters
    print("  Applying exclusion criteria...")
    subdata = filter_data(data)
    print(f"  Filtered rows: {len(subdata)}")
    
    if len(subdata) == 0:
        print("  ERROR: No data after filtering!")
        return
    
    # Fit models
    try:
        baseline_result, test_result = fit_models(subdata)
    except Exception as e:
        print(f"  ERROR fitting models: {e}")
        return
    
    # Compute metrics
    metrics = compute_metrics(baseline_result, test_result, len(subdata))
    
    # Compute residuals
    residuals = test_result.resid ** 2
    residuals_df = subdata.copy()
    residuals_df.insert(0, 'residual', residuals)
    
    # Determine output directory
    rel_path = os.path.relpath(scores_file, base_dir)
    output_dir = os.path.join(base_dir, os.path.dirname(rel_path))
    
    # Save results
    save_results(metrics, residuals_df, output_dir)
    
    print(f"  PPP (delta_loglik): {metrics['delta_loglik']:.6f}")
    print(f"  p-value: {metrics['p_value']:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description='Fit mixed-effects models to compute PPP (Python equivalent of dundee.r)'
    )
    parser.add_argument(
        'base_dir',
        help='Base directory containing scores.csv files (e.g., surprisals/DC/)'
    )
    parser.add_argument(
        '--eye-data',
        default='data/DC/all.txt.annotation.filtered.csv',
        help='Path to eye-tracking data CSV'
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load eye-tracking data
    print(f"Loading eye-tracking data: {args.eye_data}")
    eye_data = pd.read_csv(args.eye_data, sep='\t', quotechar='"')
    print(f"Eye-tracking data: {len(eye_data)} rows")
    
    # Find all scores.csv files
    pattern = os.path.join(args.base_dir, '**/scores.csv')
    scores_files = glob.glob(pattern, recursive=True)
    
    if not scores_files:
        print(f"No scores.csv files found in {args.base_dir}")
        return
    
    print(f"\nFound {len(scores_files)} scores.csv files")
    
    # Process each file
    for scores_file in sorted(scores_files):
        try:
            process_scores_file(scores_file, eye_data, args.base_dir)
        except Exception as e:
            print(f"ERROR processing {scores_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nDone!")


if __name__ == '__main__':
    main()
