#!/usr/bin/env python3
"""
FAIR-TH-DAT Paper 2 — Script 07: Clinical Utility Analysis
Decision Curve Analysis, Net Benefit, Number Needed to Screen.
Runs on CPU using pre-extracted .npz files.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import (net_benefit, net_benefit_curve, treat_all_net_benefit,
                           bootstrap_ci, risk_strata_metrics, optimize_risk_thresholds)
from utils.data import load_pakistan_data, load_baseline_predictions
from utils.plotting import apply_sr_style, decision_curve_plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def number_needed_to_screen(y_true, y_prob, threshold):
    """NNS = 1 / (Prevalence × Sensitivity) at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    total_screened = y_pred.sum()
    if tp == 0:
        return np.inf
    return total_screened / tp


def main():
    parser = argparse.ArgumentParser(description='FAIR-TH-DAT Clinical Utility')
    parser.add_argument('--frozen-dir', default='../data/frozen')
    parser.add_argument('--output-dir', default='../outputs')
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, 'figures')
    tbl_dir = os.path.join(args.output_dir, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # Load data
    print("[07] Loading data...")
    pak = load_pakistan_data(args.frozen_dir)
    test_mask = pak['test_mask']
    y_te = pak['labels'][test_mask]
    p_te = pak['probs'][test_mask]

    try:
        bl = load_baseline_predictions(args.frozen_dir)
        has_baselines = True
    except FileNotFoundError:
        has_baselines = False
        print("[WARN] No baseline predictions found")

    # Get risk thresholds from validation set
    val_mask = pak['val_mask']
    y_val = pak['labels'][val_mask]
    p_val = pak['probs'][val_mask]
    t_low, t_high = optimize_risk_thresholds(y_val, p_val)
    print(f"[07] Risk thresholds: t_low={t_low:.2f}, t_high={t_high:.2f}")

    # ─── Decision Curve Analysis ───
    print("[07] Computing decision curves...")
    thresholds = np.arange(0.01, 0.99, 0.01)
    models_dca = {}

    # TH-DAT
    _, thdat_nb = net_benefit_curve(y_te, p_te, thresholds)
    models_dca['TH-DAT'] = (thresholds, thdat_nb)

    # Baselines
    if has_baselines:
        for name, key in [('Random Forest', 'rf_probs'),
                           ('XGBoost', 'xgb_probs'),
                           ('Logistic Reg.', 'lr_probs')]:
            if key in bl:
                _, nb = net_benefit_curve(bl['test_labels'], bl[key], thresholds)
                models_dca[name] = (thresholds, nb)

    # Plot
    dca_path = os.path.join(fig_dir, 'fig7_decision_curves.png')
    decision_curve_plot(models_dca, y_te, dca_path)

    # ─── Net Benefit at Optimal Threshold ───
    print("[07] Net benefit analysis...")
    results = []
    optimal_threshold = t_high  # Use high-risk threshold as decision point

    nb_thdat = net_benefit(y_te, p_te, optimal_threshold)
    nb_ta = treat_all_net_benefit(y_te, optimal_threshold)
    results.append({
        'Model': 'TH-DAT',
        'Net_Benefit': nb_thdat,
        'NB_vs_TreatAll': nb_thdat - nb_ta,
        'NB_vs_TreatNone': nb_thdat,
    })

    if has_baselines:
        for name, key in [('RF', 'rf_probs'), ('XGB', 'xgb_probs'), ('LR', 'lr_probs')]:
            if key in bl:
                nb = net_benefit(bl['test_labels'], bl[key], optimal_threshold)
                results.append({
                    'Model': name,
                    'Net_Benefit': nb,
                    'NB_vs_TreatAll': nb - nb_ta,
                    'NB_vs_TreatNone': nb,
                })

    results.append({'Model': 'Treat All', 'Net_Benefit': nb_ta,
                    'NB_vs_TreatAll': 0, 'NB_vs_TreatNone': nb_ta})
    results.append({'Model': 'Treat None', 'Net_Benefit': 0,
                    'NB_vs_TreatAll': -nb_ta, 'NB_vs_TreatNone': 0})

    nb_df = pd.DataFrame(results)
    nb_path = os.path.join(tbl_dir, 'clinical_utility_net_benefit.csv')
    nb_df.to_csv(nb_path, index=False)
    print(f"[07] Saved {nb_path}")
    print(nb_df.to_string(index=False))

    # ─── Number Needed to Screen ───
    print("\n[07] Number Needed to Screen...")
    nns_results = []

    for name in ['Low', 'Moderate', 'High']:
        if name == 'Low':
            mask = p_te < t_low
            thresh = t_low
        elif name == 'Moderate':
            mask = (p_te >= t_low) & (p_te < t_high)
            thresh = (t_low + t_high) / 2
        else:
            mask = p_te >= t_high
            thresh = t_high

        n = mask.sum()
        if n > 0 and y_te[mask].sum() > 0:
            nns = n / y_te[mask].sum()
        else:
            nns = np.inf

        nns_results.append({
            'Risk_Stratum': name,
            'N': int(n),
            'True_Cases': int(y_te[mask].sum()),
            'NNS': round(nns, 2) if np.isfinite(nns) else 'N/A',
        })

    nns_df = pd.DataFrame(nns_results)
    nns_path = os.path.join(tbl_dir, 'clinical_utility_nns.csv')
    nns_df.to_csv(nns_path, index=False)
    print(f"[07] Saved {nns_path}")
    print(nns_df.to_string(index=False))

    # ─── Two-stage screening simulation ───
    print("\n[07] Two-stage screening simulation...")
    prevalence = y_te.mean()
    n_total = len(y_te)
    n_depressed = y_te.sum()

    # Stage 1: EPDS screen (assumed 80% sensitivity, 60% specificity)
    epds_sens, epds_spec = 0.80, 0.60
    n_epds_positive = int(n_depressed * epds_sens + (n_total - n_depressed) * (1 - epds_spec))
    n_epds_true_pos = int(n_depressed * epds_sens)

    # Stage 2: TH-DAT on EPDS positives
    # Using test set performance as proxy
    thdat_sens = (y_te[p_te >= t_high] == 1).sum() / max(y_te.sum(), 1) if y_te.sum() > 0 else 0
    thdat_spec = (y_te[p_te < t_high] == 0).sum() / max((y_te == 0).sum(), 1) if (y_te == 0).sum() > 0 else 0

    n_thdat_positive = int(n_epds_true_pos * thdat_sens + (n_epds_positive - n_epds_true_pos) * (1 - thdat_spec))
    referral_reduction = (1 - n_thdat_positive / max(n_epds_positive, 1)) * 100

    screening_results = {
        'Total_Patients': n_total,
        'Prevalence': f'{prevalence:.1%}',
        'Stage1_EPDS_Positives': n_epds_positive,
        'Stage2_THDAT_Referrals': n_thdat_positive,
        'Referral_Reduction': f'{referral_reduction:.1f}%',
        'THDAT_Sensitivity_at_HighRisk': f'{thdat_sens:.3f}',
        'THDAT_Specificity_at_HighRisk': f'{thdat_spec:.3f}',
    }

    screen_df = pd.DataFrame([screening_results])
    screen_path = os.path.join(tbl_dir, 'clinical_utility_screening.csv')
    screen_df.to_csv(screen_path, index=False)
    print(f"[07] Saved {screen_path}")
    print(f"\n  Two-stage screening simulation:")
    for k, v in screening_results.items():
        print(f"    {k}: {v}")

    print("\n[07] Clinical utility analysis COMPLETE")


if __name__ == '__main__':
    main()
