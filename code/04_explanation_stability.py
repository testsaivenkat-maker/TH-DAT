"""
Explanation Stability Analysis for FAIR-TH-DAT
Computes stability of TH-DAT domain gate weights across bootstraps,
seeds, and perturbations.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-dir', type=str, required=True, help="Directory containing .npz files")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def run_bootstrap_stability(gate_w, n_bootstraps=1000, seed=42):
    """Compute bootstrap stability of mean gate weights."""
    np.random.seed(seed)
    n_samples, n_domains = gate_w.shape
    
    boot_means = np.zeros((n_bootstraps, n_domains))
    for i in range(n_bootstraps):
        idx = resample(np.arange(n_samples), random_state=seed+i)
        boot_means[i] = np.mean(gate_w[idx], axis=0)
        
    mean_of_means = np.mean(boot_means, axis=0)
    std_of_means = np.std(boot_means, axis=0)
    
    # Concordance of domain rankings across bootstraps
    ranks = np.argsort(np.argsort(boot_means, axis=1), axis=1) # rank 0, 1, 2
    # Simplified Kendall's W proxy: mean pairwise spearman
    n_pairs = 100
    pair_spearmans = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_bootstraps, 2, replace=False)
        corr, _ = spearmanr(ranks[i], ranks[j])
        pair_spearmans.append(corr)
        
    concordance = np.mean(pair_spearmans)
    
    return mean_of_means, std_of_means, concordance

def run_consistency_analysis(gate_w):
    """Assess consistency of gate weight rankings across patients."""
    ranks = np.argsort(np.argsort(-gate_w, axis=1), axis=1)
    top_domain = np.argmin(ranks, axis=1)
    
    counts = np.bincount(top_domain, minlength=gate_w.shape[1])
    fractions = counts / len(top_domain)
    
    return fractions

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    primary_file = os.path.join(args.frozen_dir, 'pakistan_predictions.npz')
    if not os.path.exists(primary_file):
        print(f"Error: Primary file {primary_file} not found.")
        return
        
    print(f"Loading {primary_file}...")
    data = np.load(primary_file)
    gate_w = data['gate_w']
    
    results = {}
    
    # 1. Bootstrap Stability
    print("Computing bootstrap stability...")
    mean_w, std_w, concordance = run_bootstrap_stability(gate_w, seed=args.seed)
    results['bootstrap_mean_w'] = mean_w
    results['bootstrap_std_w'] = std_w
    results['bootstrap_concordance'] = concordance
    
    # 2. Patient-level consistency (Proxy for perturbation stability without model weights)
    print("Computing patient-level consistency...")
    top_domain_fractions = run_consistency_analysis(gate_w)
    results['top_domain_fractions'] = top_domain_fractions
    
    # 3. Multi-seed stability
    seed_files = glob.glob(os.path.join(args.frozen_dir, 'pakistan_seed_*.npz'))
    if len(seed_files) > 0:
        print(f"Found {len(seed_files)} seed files. Computing multi-seed stability...")
        seed_means = []
        for sf in seed_files:
            sd_data = np.load(sf)
            seed_means.append(np.mean(sd_data['gate_w'], axis=0))
        
        seed_means = np.array(seed_means)
        mean_sd = np.std(seed_means, axis=0)
        results['multiseed_std'] = mean_sd
    else:
        print("No multi-seed files found. Skipping.")
        
    # Save results
    out_csv = os.path.join(args.output_dir, 'stability_results.csv')
    df_results = pd.DataFrame([
        {'metric': 'bootstrap_concordance', 'value': results.get('bootstrap_concordance', np.nan)},
    ])
    for i in range(len(mean_w)):
        df_results = pd.concat([df_results, pd.DataFrame([
            {'metric': f'bootstrap_mean_domain_{i}', 'value': mean_w[i]},
            {'metric': f'bootstrap_std_domain_{i}', 'value': std_w[i]},
            {'metric': f'top_domain_{i}_fraction', 'value': top_domain_fractions[i]},
        ])])
    
    df_results.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(gate_w[:100], cmap='viridis')
    plt.title('Gate Weights Heatmap (First 100 patients)')
    plt.xlabel('Domain')
    plt.ylabel('Patient')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'gate_weights_heatmap.png'), dpi=300)
    plt.close()
    print("Stability analysis complete.")

if __name__ == '__main__':
    main()
