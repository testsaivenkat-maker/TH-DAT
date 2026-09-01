"""Metrics module for FAIR-TH-DAT Paper 2.
Calibration, fairness, clinical utility, and risk stratification metrics.
All metrics support bootstrap 95% CIs.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             confusion_matrix, cohen_kappa_score, log_loss)
from scipy.special import expit, logit
from scipy.optimize import minimize
from typing import Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════

def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn: Callable, n_boot: int = 2000,
                 seed: int = 42, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for any metric.

    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities
        metric_fn: Function(y_true, y_score) -> float
        n_boot: Number of bootstrap iterations
        seed: Random seed
        ci: Confidence level (0.95 = 95%)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_score)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(s):
                scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue
    if len(scores) < 10:
        return point, np.nan, np.nan
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return point, lo, hi


# ═══════════════════════════════════════════════════
# CALIBRATION METRICS
# ═══════════════════════════════════════════════════

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower is better, 0 = perfect)."""
    return brier_score_loss(y_true, y_prob)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (n_bin / total) * abs(acc - conf)
    return ece


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Maximum Calibration Error (MCE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce = max(mce, abs(acc - conf))
    return mce


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray
                                 ) -> Tuple[float, float]:
    """Logistic recalibration slope and intercept.

    Fits: logit(P_cal) = a + b * logit(P_orig)
    Perfect calibration: a=0, b=1.

    Returns:
        (slope, intercept)
    """
    eps = 1e-7
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(p_clipped)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(logits.reshape(-1, 1), y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute data for reliability diagram.

    Returns:
        bin_centers: midpoint of each bin
        bin_means: observed frequency (accuracy) in each bin
        bin_counts: number of samples in each bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    centers, means, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else \
               (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        centers.append((lo + hi) / 2)
        means.append(y_true[mask].mean())
        counts.append(n_bin)
    return np.array(centers), np.array(means), np.array(counts)


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray,
                          n_groups: int = 10) -> Tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test.

    Returns:
        (chi2_statistic, p_value)
    """
    from scipy.stats import chi2
    order = np.argsort(y_prob)
    groups = np.array_split(order, n_groups)
    chi2_stat = 0.0
    for g in groups:
        n_g = len(g)
        if n_g == 0:
            continue
        obs = y_true[g].sum()
        exp = y_prob[g].sum()
        exp_neg = n_g - exp
        if exp > 0:
            chi2_stat += (obs - exp) ** 2 / exp
        if exp_neg > 0:
            chi2_stat += ((n_g - obs) - exp_neg) ** 2 / exp_neg
    p_val = 1 - chi2.cdf(chi2_stat, n_groups - 2)
    return chi2_stat, p_val


# ═══════════════════════════════════════════════════
# CLINICAL UTILITY
# ═══════════════════════════════════════════════════

def net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                threshold: float) -> float:
    """Net benefit at a given decision threshold.

    NB = (TP/N) - (FP/N) * (pt / (1 - pt))
    """
    n = len(y_true)
    if n == 0 or threshold <= 0 or threshold >= 1:
        return 0.0
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / n - fp / n * (threshold / (1 - threshold))


def net_benefit_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      thresholds: np.ndarray = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Net benefit across a range of thresholds.

    Returns:
        thresholds, net_benefits
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    nbs = [net_benefit(y_true, y_prob, t) for t in thresholds]
    return thresholds, np.array(nbs)


def treat_all_net_benefit(y_true: np.ndarray, threshold: float) -> float:
    """Net benefit of treating all patients."""
    prevalence = y_true.mean()
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


# ═══════════════════════════════════════════════════
# FAIRNESS METRICS
# ═══════════════════════════════════════════════════

def subgroup_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float = 0.5) -> dict:
    """Compute comprehensive metrics for a subgroup.

    Returns dict with: n, prevalence, auc, sensitivity, specificity,
                       ppv, npv, fpr, fnr, brier, ece, accuracy, f1
    """
    n = len(y_true)
    if n < 5:
        return {'n': n, 'prevalence': np.nan}

    y_pred = (y_prob >= threshold).astype(int)
    result = {'n': n, 'prevalence': float(y_true.mean())}

    try:
        result['auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        result['auc'] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    result['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    result['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    result['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    result['npv'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    result['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    result['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    result['accuracy'] = float(accuracy_score(y_true, y_pred))
    result['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    result['brier'] = float(brier_score_loss(y_true, y_prob))
    result['ece'] = float(expected_calibration_error(y_true, y_prob))

    return result


def equalized_odds_gap(y_true: np.ndarray, y_pred: np.ndarray,
                        group_labels: np.ndarray) -> Dict[str, float]:
    """Equalized odds gap: max |FPR_i - FPR_j| and max |FNR_i - FNR_j|.

    Returns dict with fpr_gap, fnr_gap, max_gap
    """
    groups = np.unique(group_labels)
    fprs, fnrs = [], []
    for g in groups:
        mask = group_labels == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 5:
            continue
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        fnrs.append(fn / (fn + tp) if (fn + tp) > 0 else 0)

    fpr_gap = max(fprs) - min(fprs) if fprs else 0
    fnr_gap = max(fnrs) - min(fnrs) if fnrs else 0
    return {'fpr_gap': fpr_gap, 'fnr_gap': fnr_gap,
            'max_gap': max(fpr_gap, fnr_gap)}


def predictive_parity_gap(y_true: np.ndarray, y_pred: np.ndarray,
                           group_labels: np.ndarray) -> float:
    """Max |PPV_i - PPV_j| across groups."""
    groups = np.unique(group_labels)
    ppvs = []
    for g in groups:
        mask = group_labels == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yp == 1) & (yt == 1)).sum()
        pp = (yp == 1).sum()
        if pp > 0:
            ppvs.append(tp / pp)
    return max(ppvs) - min(ppvs) if len(ppvs) >= 2 else 0.0


def calibration_gap(y_true: np.ndarray, y_prob: np.ndarray,
                     group_labels: np.ndarray) -> float:
    """Max |ECE_i - ECE_j| across groups."""
    groups = np.unique(group_labels)
    eces = []
    for g in groups:
        mask = group_labels == g
        if mask.sum() < 10:
            continue
        eces.append(expected_calibration_error(y_true[mask], y_prob[mask]))
    return max(eces) - min(eces) if len(eces) >= 2 else 0.0


# ═══════════════════════════════════════════════════
# RISK STRATA
# ═══════════════════════════════════════════════════

def optimize_risk_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                              sens_constraint: float = 0.90,
                              npv_constraint: float = 0.90
                              ) -> Tuple[float, float]:
    """Optimize Low/High risk thresholds on validation set.

    Low threshold: maximize NPV while NPV >= npv_constraint
    High threshold: maximize sensitivity while sens >= sens_constraint

    Returns:
        (t_low, t_high)
    """
    best_score = -1
    best_tl, best_th = 0.3, 0.7
    for tl in np.arange(0.10, 0.50, 0.01):
        for th in np.arange(tl + 0.10, 0.90, 0.01):
            # High-risk group
            high_mask = y_prob >= th
            if high_mask.sum() < 5:
                continue
            sens_high = y_true[high_mask].sum() / max(y_true.sum(), 1)

            # Low-risk group
            low_mask = y_prob < tl
            if low_mask.sum() < 5:
                continue
            npv_low = (1 - y_true[low_mask]).sum() / max(low_mask.sum(), 1)

            if sens_high >= sens_constraint and npv_low >= npv_constraint:
                score = sens_high + npv_low  # maximize both
                if score > best_score:
                    best_score = score
                    best_tl, best_th = tl, th
    return best_tl, best_th


def risk_strata_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                         t_low: float, t_high: float) -> dict:
    """Compute metrics for each risk stratum.

    Returns dict with Low/Moderate/High strata metrics.
    """
    strata = {
        'Low':      y_prob < t_low,
        'Moderate': (y_prob >= t_low) & (y_prob < t_high),
        'High':     y_prob >= t_high,
    }
    results = {}
    for name, mask in strata.items():
        n = mask.sum()
        if n == 0:
            results[name] = {'n': 0}
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        results[name] = {
            'n': int(n),
            'pct': float(n / len(y_true) * 100),
            'prevalence': float(yt.mean()),
            'mean_prob': float(yp.mean()),
            'brier': float(brier_score_loss(yt, yp)) if n > 1 else np.nan,
            'ece': float(expected_calibration_error(yt, yp)) if n > 10 else np.nan,
        }
    # Overall sensitivity/specificity of binary High vs Not-High
    y_pred_high = (y_prob >= t_high).astype(int)
    cm = confusion_matrix(y_true, y_pred_high, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        results['High']['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        results['High']['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        results['High']['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        results['High']['npv'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # Low-risk NPV
    y_pred_low = (y_prob < t_low).astype(int)
    low_mask = y_prob < t_low
    if low_mask.sum() > 0:
        results['Low']['npv'] = float((1 - y_true[low_mask]).mean())
        results['Low']['fnr_in_low'] = float(y_true[low_mask].mean())

    return results


def cohens_kappa(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Cohen's kappa for category agreement."""
    return float(cohen_kappa_score(labels1, labels2))
