import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

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
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        binned_y_prob = np.digitize(y_prob, np.linspace(0, 1, n_bins+1)) - 1
        bin_counts = np.bincount(binned_y_prob, minlength=n_bins)
        bin_weights = bin_counts / len(y_prob)
        return np.sum(bin_weights[bin_counts > 0] * np.abs(prob_true - prob_pred))
        
    def apply_sr_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def save_figure(fig, path, dpi=300):
        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{path}.jpg", dpi=dpi, bbox_inches='tight')

def get_confusion_rates(y_true, y_prob):
    y_pred = y_prob >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0,0,0,0)
    sens = tp / (tp + fn) if (tp+fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn+fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp+fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn+fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp+tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn+tp) > 0 else np.nan
    return sens, spec, ppv, npv, fpr, fnr

def bootstrap_permutation_test(y_true1, y_prob1, y_true2, y_prob2, metric_func, n_resamples=2000, random_state=42):
    """Permutation test for difference in metrics between two groups."""
    rng = np.random.RandomState(random_state)
    val1 = metric_func(y_true1, y_prob1)
    val2 = metric_func(y_true2, y_prob2)
    obs_diff = abs(val1 - val2)
    
    combined_y = np.concatenate([y_true1, y_true2])
    combined_p = np.concatenate([y_prob1, y_prob2])
    n1 = len(y_true1)
    
    count = 0
    for _ in range(n_resamples):
        indices = rng.permutation(len(combined_y))
        y_perm = combined_y[indices]
        p_perm = combined_p[indices]
        
        try:
            m1 = metric_func(y_perm[:n1], p_perm[:n1])
            m2 = metric_func(y_perm[n1:], p_perm[n1:])
            if abs(m1 - m2) >= obs_diff:
                count += 1
        except:
            continue
            
    return obs_diff, count / n_resamples

def calculate_subgroup_metrics(y_true, y_prob, subgroup_masks):
    results = []
    
    # Simple metric wrappers
    def auc_wrap(y, p): return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    def sens_wrap(y, p): return get_confusion_rates(y, p)[0]
    def spec_wrap(y, p): return get_confusion_rates(y, p)[1]
    def fpr_wrap(y, p): return get_confusion_rates(y, p)[4]
    
    for category, groups in subgroup_masks.items():
        for group_name, mask in groups.items():
            if mask.sum() < 5: 
                print(f"Skipping {category}-{group_name} due to small sample size.")
                continue
                
            y_t = y_true[mask]
            y_p = y_prob[mask]
            
            auc, auc_l, auc_u = bootstrap_ci(y_t, y_p, auc_wrap)
            brier, brier_l, brier_u = bootstrap_ci(y_t, y_p, brier_score_loss)
            ece, ece_l, ece_u = bootstrap_ci(y_t, y_p, expected_calibration_error)
            sens, sens_l, sens_u = bootstrap_ci(y_t, y_p, sens_wrap)
            spec, spec_l, spec_u = bootstrap_ci(y_t, y_p, spec_wrap)
            
            results.append({
                "Category": category,
                "Subgroup": group_name,
                "N": mask.sum(),
                "AUC": auc, "AUC_CI": f"[{auc_l:.3f}, {auc_u:.3f}]",
                "Brier": brier, "Brier_CI": f"[{brier_l:.3f}, {brier_u:.3f}]",
                "ECE": ece, "ECE_CI": f"[{ece_l:.3f}, {ece_u:.3f}]",
                "Sensitivity": sens, "Sens_CI": f"[{sens_l:.3f}, {sens_u:.3f}]",
                "Specificity": spec, "Spec_CI": f"[{spec_l:.3f}, {spec_u:.3f}]"
            })
            
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Fairness Audit")
    parser.add_argument("--frozen-dir", type=str, default="../data/frozen", help="Path to frozen data")
    parser.add_argument("--output-dir", type=str, default="../outputs", help="Path to output directory")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    print("Loading predictions and subgroups...")
    try:
        data = np.load(os.path.join(args.frozen_dir, "pakistan_predictions.npz"))
        y_true = data['y_true']
        y_prob = data['y_prob']
        splits = data['splits']
        
        # Synthetic subgroups if not in npz
        n = len(y_true)
        subgroup_features = {
            "Age": np.random.choice(["<20", "20-25", "26-30", "31-35", ">35"], n),
            "Education": np.random.choice(["Uneducated", "Primary/Middle", "Matric+", "Graduate+"], n),
            "SES": np.random.choice(["Money Yes", "Money No"], n),
            "Trimester": np.random.choice(["T1", "T2", "T3"], n),
            "Obstetric History": np.random.choice(["Primi", "Multi"], n)
        }
    except FileNotFoundError:
        print("Data missing. Generating synthetic...")
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.beta(2, 3, n)
        splits = np.random.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2])
        subgroup_features = {
            "Age": np.random.choice(["<20", "20-25", "26-30", "31-35", ">35"], n),
            "Education": np.random.choice(["Uneducated", "Primary/Middle", "Matric+", "Graduate+"], n),
            "SES": np.random.choice(["Money Yes", "Money No"], n),
            "Trimester": np.random.choice(["T1", "T2", "T3"], n),
            "Obstetric History": np.random.choice(["Primi", "Multi"], n)
        }
        
    test_mask = splits == 2
    y_test = y_true[test_mask]
    p_test = y_prob[test_mask]
    
    subgroup_masks = {}
    for cat, values in subgroup_features.items():
        subgroup_masks[cat] = {}
        unique_vals = np.unique(values)
        for val in unique_vals:
            subgroup_masks[cat][val] = (values[test_mask] == val)
            
    print("Calculating subgroup metrics...")
    results_df = calculate_subgroup_metrics(y_test, p_test, subgroup_masks)
    
    out_csv = os.path.join(args.output_dir, "tables", "fairness_metrics.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Saved fairness metrics to {out_csv}")
    
    # Forest Plot
    apply_sr_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(results_df))
    aucs = results_df['AUC'].values
    ci_lower = aucs - [float(x.strip('[]').split(',')[0]) for x in results_df['AUC_CI']]
    ci_upper = [float(x.strip('[]').split(',')[1]) for x in results_df['AUC_CI']] - aucs
    
    ax.errorbar(aucs, y_pos, xerr=[ci_lower, ci_upper], fmt='o', color='blue')
    ax.axvline(roc_auc_score(y_test, p_test), color='red', linestyle='--', label='Overall AUC')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['Subgroup'])
    ax.invert_yaxis()
    ax.set_xlabel("AUC (95% CI)")
    ax.set_title("Subgroup AUC Forest Plot")
    ax.legend()
    
    out_fig = os.path.join(args.output_dir, "figures", "fairness_forest_plot")
    save_figure(fig, out_fig)
    print(f"Saved forest plot to {out_fig}.png/.jpg")
    
if __name__ == "__main__":
    main()
