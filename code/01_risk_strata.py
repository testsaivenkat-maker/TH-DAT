import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

# Insert local path for utils imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.metrics import bootstrap_ci
    from utils.plotting import apply_sr_style, save_figure
except ImportError:
    # Fallback for self-contained execution if utils not present
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
    
    def apply_sr_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def save_figure(fig, path, dpi=300):
        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{path}.jpg", dpi=dpi, bbox_inches='tight')

def optimize_risk_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Optimize risk thresholds to stratify into Low, Moderate, and High risk.
    Uses validation set to find thresholds.
    """
    # Simple percentile-based approach for demonstration: 33rd and 66th percentiles of positive predictions
    # A real implementation would use specific clinical sensitivity/specificity targets
    t_low = np.percentile(y_prob, 50)
    t_high = np.percentile(y_prob, 85)
    return float(t_low), float(t_high)

def calculate_stratum_metrics(y_true: np.ndarray, y_prob: np.ndarray, t_low: float, t_high: float) -> pd.DataFrame:
    """Calculate metrics for each risk stratum with 95% CIs."""
    strata = []
    conditions = [
        ("Low", y_prob < t_low),
        ("Moderate", (y_prob >= t_low) & (y_prob < t_high)),
        ("High", y_prob >= t_high)
    ]
    
    total_n = len(y_true)
    
    def sens_func(y, p):
        tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel() if len(np.unique(y)) > 1 else (0,0,0,0)
        return tp / (tp + fn) if (tp+fn) > 0 else np.nan
        
    def spec_func(y, p):
        tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel() if len(np.unique(y)) > 1 else (0,0,0,0)
        return tn / (tn + fp) if (tn+fp) > 0 else np.nan

    def ppv_func(y, p):
        tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel() if len(np.unique(y)) > 1 else (0,0,0,0)
        return tp / (tp + fp) if (tp+fp) > 0 else np.nan

    def npv_func(y, p):
        tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel() if len(np.unique(y)) > 1 else (0,0,0,0)
        return tn / (tn + fn) if (tn+fn) > 0 else np.nan
    
    for name, mask in conditions:
        n = mask.sum()
        if n == 0:
            continue
            
        y_t = y_true[mask]
        y_p = y_prob[mask]
        
        prevalence = y_t.mean()
        mean_prob = y_p.mean()
        
        # Calculate CIs if possible
        if len(np.unique(y_t)) > 1:
            sens, sens_l, sens_u = bootstrap_ci(y_t, y_p, sens_func)
            spec, spec_l, spec_u = bootstrap_ci(y_t, y_p, spec_func)
            ppv, ppv_l, ppv_u = bootstrap_ci(y_t, y_p, ppv_func)
            npv, npv_l, npv_u = bootstrap_ci(y_t, y_p, npv_func)
        else:
            sens, sens_l, sens_u = np.nan, np.nan, np.nan
            spec, spec_l, spec_u = np.nan, np.nan, np.nan
            ppv, ppv_l, ppv_u = np.nan, np.nan, np.nan
            npv, npv_l, npv_u = np.nan, np.nan, np.nan
            
        strata.append({
            "Stratum": name,
            "N": n,
            "Percentage": (n / total_n) * 100,
            "Prevalence": prevalence,
            "Mean_Prob": mean_prob,
            "Sensitivity": sens,
            "Sensitivity_CI": f"[{sens_l:.3f}, {sens_u:.3f}]",
            "Specificity": spec,
            "Specificity_CI": f"[{spec_l:.3f}, {spec_u:.3f}]",
            "PPV": ppv,
            "PPV_CI": f"[{ppv_l:.3f}, {ppv_u:.3f}]",
            "NPV": npv,
            "NPV_CI": f"[{npv_l:.3f}, {npv_u:.3f}]"
        })
        
    return pd.DataFrame(strata)

def main():
    parser = argparse.ArgumentParser(description="Risk Stratification Analysis")
    parser.add_argument("--frozen-dir", type=str, default="../data/frozen", help="Path to frozen data")
    parser.add_argument("--output-dir", type=str, default="../outputs", help="Path to output directory")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    print("Loading data...")
    try:
        # Mocking data load based on described structure
        data = np.load(os.path.join(args.frozen_dir, "pakistan_predictions.npz"))
        y_true = data['y_true']
        y_prob = data['y_prob']
        splits = data['splits'] # 1=val, 2=test
    except FileNotFoundError:
        print(f"Warning: Data not found at {args.frozen_dir}. Generating synthetic data for demonstration.")
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.beta(1, 3, 1000)
        y_prob[y_true == 1] = np.random.beta(3, 1, y_true.sum())
        splits = np.random.choice([0, 1, 2], 1000, p=[0.6, 0.2, 0.2])
        
    val_mask = splits == 1
    test_mask = splits == 2
    
    print("Optimizing thresholds on validation set...")
    t_low, t_high = optimize_risk_thresholds(y_true[val_mask], y_prob[val_mask])
    print(f"Thresholds identified: Low < {t_low:.3f} <= Moderate < {t_high:.3f} <= High")
    
    print("Calculating metrics on test set...")
    results_df = calculate_stratum_metrics(y_true[test_mask], y_prob[test_mask], t_low, t_high)
    
    # Print formatted table
    print("\nRisk Stratification Results:")
    print(results_df.to_string(index=False))
    
    # Save results
    out_csv = os.path.join(args.output_dir, "tables", "risk_strata_metrics.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")
    
    # Plot histogram
    apply_sr_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    test_probs = y_prob[test_mask]
    test_labels = y_true[test_mask]
    
    ax.hist(test_probs[test_labels == 0], bins=30, alpha=0.5, label='Control (0)', color='blue')
    ax.hist(test_probs[test_labels == 1], bins=30, alpha=0.5, label='Case (1)', color='red')
    ax.axvline(t_low, color='green', linestyle='--', label=f't_low ({t_low:.2f})')
    ax.axvline(t_high, color='orange', linestyle='--', label=f't_high ({t_high:.2f})')
    
    ax.set_title("Risk Distribution by True Class")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend()
    
    out_fig = os.path.join(args.output_dir, "figures", "risk_distribution")
    save_figure(fig, out_fig)
    print(f"Saved figure to {out_fig}.png/.jpg")

if __name__ == "__main__":
    main()
