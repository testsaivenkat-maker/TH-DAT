import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

# Insert local path for utils imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.metrics import bootstrap_ci, expected_calibration_error
    from utils.plotting import apply_sr_style, save_figure
except ImportError:
    # Fallback for self-contained execution
    def bootstrap_ci(y_true, y_pred, metric_func, n_resamples=2000, random_state=42, **kwargs):
        rng = np.random.RandomState(random_state)
        values = []
        n = len(y_true)
        for _ in range(n_resamples):
            indices = rng.choice(n, size=n, replace=True)
            if len(np.unique(y_true[indices])) < 2: continue
            values.append(metric_func(y_true[indices], y_pred[indices], **kwargs))
        if not values: return np.nan, np.nan, np.nan
        return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)
        
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        binned_y_prob = np.digitize(y_prob, np.linspace(0, 1, n_bins+1)) - 1
        bin_counts = np.bincount(binned_y_prob, minlength=n_bins)
        bin_weights = bin_counts / len(y_prob)
        ece = np.sum(bin_weights[bin_counts > 0] * np.abs(prob_true - prob_pred))
        return ece
        
    def apply_sr_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def save_figure(fig, path, dpi=300):
        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{path}.jpg", dpi=dpi, bbox_inches='tight')

def maximum_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) == 0: return np.nan
    return np.max(np.abs(prob_true - prob_pred))

def hosmer_lemeshow_test(y_true, y_prob, g=10):
    """Simple HL test implementation."""
    from scipy.stats import chi2
    pihat = y_prob
    g = min(g, len(np.unique(y_prob)))
    try:
        qs = np.quantile(pihat, np.linspace(0, 1, g+1))
        qs[0] = -1e-5
        bins = pd.cut(pihat, qs)
        obs = pd.crosstab(bins, y_true)
        exp_1 = [np.sum(pihat[bins == b]) for b in obs.index]
        exp_0 = [np.sum(1 - pihat[bins == b]) for b in obs.index]
        hl = np.sum((obs[1] - exp_1)**2 / exp_1) + np.sum((obs[0] - exp_0)**2 / exp_0)
        p_val = 1 - chi2.cdf(hl, g - 2)
        return hl, p_val
    except:
        return np.nan, np.nan

def calib_slope_intercept(y_true, y_prob):
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(y_prob / (1 - y_prob))
    lr = LogisticRegression(penalty=None).fit(logit_p.reshape(-1, 1), y_true)
    return lr.coef_[0][0], lr.intercept_[0]

def main():
    parser = argparse.ArgumentParser(description="Calibration Analysis")
    parser.add_argument("--frozen-dir", type=str, default="../data/frozen", help="Path to frozen data")
    parser.add_argument("--output-dir", type=str, default="../outputs", help="Path to output directory")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    print("Loading predictions...")
    models = {}
    try:
        # Expected to contain TH-DAT and baselines (RF, XGB, LR)
        th_data = np.load(os.path.join(args.frozen_dir, "pakistan_predictions.npz"))
        models['TH-DAT'] = {
            'y_true': th_data['y_true'],
            'y_prob': th_data['y_prob'],
            'splits': th_data['splits']
        }
        
        baseline_data = np.load(os.path.join(args.frozen_dir, "baseline_predictions.npz"))
        for model in ['RF', 'XGB', 'LR']:
            models[model] = {
                'y_true': baseline_data[f'{model}_y_true'],
                'y_prob': baseline_data[f'{model}_y_prob'],
                'splits': baseline_data['splits']
            }
    except FileNotFoundError:
        print("Warning: Data files missing. Using synthetic data.")
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        splits = np.random.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2])
        models = {
            'TH-DAT': {'y_true': y_true, 'y_prob': np.random.beta(2, 3, n), 'splits': splits},
            'RF': {'y_true': y_true, 'y_prob': np.random.beta(1.5, 3, n), 'splits': splits},
            'XGB': {'y_true': y_true, 'y_prob': np.random.beta(1, 4, n), 'splits': splits},
            'LR': {'y_true': y_true, 'y_prob': np.random.beta(2.5, 2.5, n), 'splits': splits}
        }

    results = []
    
    apply_sr_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, data in models.items():
        print(f"Processing {name}...")
        y_true_all = data['y_true']
        y_prob_all = data['y_prob']
        splits = data['splits']
        
        val_mask = splits == 1
        test_mask = splits == 2
        
        # Platt Scaling
        lr = LogisticRegression(penalty=None)
        lr.fit(y_prob_all[val_mask].reshape(-1, 1), y_true_all[val_mask])
        y_prob_calibrated = lr.predict_proba(y_prob_all[test_mask].reshape(-1, 1))[:, 1]
        
        y_t = y_true_all[test_mask]
        
        for variant, y_p in [("Uncalibrated", y_prob_all[test_mask]), ("Platt Scaled", y_prob_calibrated)]:
            # Metrics
            brier, brier_l, brier_u = bootstrap_ci(y_t, y_p, brier_score_loss)
            ece, ece_l, ece_u = bootstrap_ci(y_t, y_p, expected_calibration_error)
            mce, mce_l, mce_u = bootstrap_ci(y_t, y_p, maximum_calibration_error)
            
            slope, intercept = calib_slope_intercept(y_t, y_p)
            hl_stat, hl_p = hosmer_lemeshow_test(y_t, y_p)
            
            results.append({
                "Model": name,
                "Variant": variant,
                "Brier": f"{brier:.4f} [{brier_l:.4f}, {brier_u:.4f}]",
                "ECE": f"{ece:.4f} [{ece_l:.4f}, {ece_u:.4f}]",
                "MCE": f"{mce:.4f} [{mce_l:.4f}, {mce_u:.4f}]",
                "Slope": slope,
                "Intercept": intercept,
                "HL_Stat": hl_stat,
                "HL_pvalue": hl_p
            })
            
            if variant == "Uncalibrated":
                # Plot reliability curve
                prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10)
                ax.plot(prob_pred, prob_true, marker='s', label=f'{name} (ECE={ece:.3f})')
                
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax.set_title("Reliability Diagram (Test Set)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    
    out_fig = os.path.join(args.output_dir, "figures", "reliability_diagram")
    save_figure(fig, out_fig)
    print(f"Saved reliability diagram to {out_fig}.png/.jpg")
    
    df_results = pd.DataFrame(results)
    print("\nCalibration Metrics:")
    print(df_results.to_string(index=False))
    
    out_csv = os.path.join(args.output_dir, "tables", "calibration_metrics.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

if __name__ == "__main__":
    main()
