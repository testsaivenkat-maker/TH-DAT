"""Plotting utilities for FAIR-TH-DAT Paper 2.
Scientific Reports style, 300 DPI, publication-quality figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

SR_STYLE = {
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLORS = {
    'thdat': '#2C73D2',
    'rf': '#FF6B6B',
    'xgb': '#FFA726',
    'lr': '#66BB6A',
    'tabtrans': '#AB47BC',
    'demo': '#4A86C8',
    'obst': '#E6A020',
    'psyc': '#6AAF50',
    'low': '#66BB6A',
    'moderate': '#FFA726',
    'high': '#FF6B6B',
}


def apply_sr_style():
    plt.rcParams.update(SR_STYLE)


def save_figure(fig, name: str, output_dir: str = '../outputs/figures'):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'{name}.png')
    jpg_path = os.path.join(output_dir, f'{name}.jpg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(jpg_path, dpi=300, bbox_inches='tight', facecolor='white',
                format='jpeg', pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f"[FIG] Saved {png_path}")
    return png_path


def reliability_plot(models_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     output_path: str):
    """Reliability diagram comparing multiple models.

    Args:
        models_data: {model_name: (bin_centers, bin_means, bin_counts)}
    """
    apply_sr_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.05})

    colors = list(COLORS.values())
    for i, (name, (centers, means, counts)) in enumerate(models_data.items()):
        c = colors[i % len(colors)]
        ax1.plot(centers, means, 'o-', color=c, label=name, markersize=4, linewidth=1.5)
        ax2.bar(centers, counts, width=0.08, alpha=0.5, color=c, label=name)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.set_ylabel('Observed frequency')
    ax1.set_title('Reliability Diagram')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Count')

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def fairness_forest_plot(subgroup_results: Dict[str, Dict[str, dict]],
                          metric: str = 'auc',
                          output_path: str = 'fairness_forest.png'):
    """Forest plot of subgroup metrics with CIs.

    Args:
        subgroup_results: {attribute: {group: {metric: (point, lo, hi)}}}
    """
    apply_sr_style()
    labels, points, lows, highs = [], [], [], []

    for attr, groups in subgroup_results.items():
        labels.append(f'--- {attr} ---')
        points.append(np.nan)
        lows.append(np.nan)
        highs.append(np.nan)
        for group, metrics in groups.items():
            if metric in metrics:
                val = metrics[metric]
                if isinstance(val, tuple) and len(val) == 3:
                    labels.append(f'  {group} (n={metrics.get("n", "?")})')
                    points.append(val[0])
                    lows.append(val[1])
                    highs.append(val[2])

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.3)))
    y_pos = np.arange(len(labels))

    for i in range(len(labels)):
        if np.isnan(points[i]):
            ax.text(0.5, y_pos[i], labels[i], fontweight='bold', fontsize=9,
                   va='center', transform=ax.get_yaxis_transform())
        else:
            xerr = [[points[i] - lows[i]], [highs[i] - points[i]]]
            ax.errorbar(points[i], y_pos[i], xerr=xerr, fmt='o',
                       color=COLORS['thdat'], markersize=5, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f'Subgroup {metric.upper()} with 95% CI')
    ax.axvline(x=np.nanmean(points), color='gray', linestyle='--', alpha=0.5)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def decision_curve_plot(models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         y_true: np.ndarray, output_path: str):
    """Decision curve analysis plot.

    Args:
        models_data: {model_name: (thresholds, net_benefits)}
        y_true: true labels for treat-all line
    """
    apply_sr_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    prevalence = y_true.mean()
    thresholds = np.arange(0.01, 0.99, 0.01)
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all, 'k:', label='Treat All', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', label='Treat None', linewidth=1)

    colors = list(COLORS.values())
    for i, (name, (ts, nbs)) in enumerate(models_data.items()):
        ax.plot(ts, nbs, color=colors[i % len(colors)], label=name, linewidth=1.5)

    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, max(prevalence * 1.2, 0.1))

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def trustworthiness_radar(scores: Dict[str, float], output_path: str):
    """Spider/radar diagram for trustworthiness dimensions."""
    apply_sr_style()
    labels = list(scores.keys())
    values = list(scores.values())
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color=COLORS['thdat'], alpha=0.25)
    ax.plot(angles, values, color=COLORS['thdat'], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('FAIR-TH-DAT Trustworthiness Assessment', fontsize=12, pad=20)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def phenotype_umap_plot(embeddings_2d: np.ndarray, cluster_labels: np.ndarray,
                         y_true: np.ndarray, output_path: str):
    """UMAP scatter colored by cluster with depression prevalence annotation."""
    apply_sr_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
    cmap = plt.cm.Set2
    for i, cl in enumerate(unique_labels):
        mask = cluster_labels == cl
        prev = y_true[mask].mean()
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[cmap(i)], s=5, alpha=0.5,
                   label=f'P{cl+1} (n={mask.sum()}, prev={prev:.1%})')

    noise = cluster_labels < 0
    if noise.sum() > 0:
        ax1.scatter(embeddings_2d[noise, 0], embeddings_2d[noise, 1],
                   c='lightgray', s=3, alpha=0.3, label='Noise')

    ax1.set_title('Risk Phenotypes (UMAP)')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(fontsize=7, markerscale=3)

    # Depression status overlay
    colors = ['#66BB6A', '#FF6B6B']
    for label, color, name in zip([0, 1], colors, ['Not Depressed', 'Depressed']):
        mask = y_true == label
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=color, s=5, alpha=0.4, label=name)
    ax2.set_title('Depression Status Overlay')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(fontsize=8, markerscale=3)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def phenotype_radar_chart(phenotype_profiles: Dict[str, Dict[str, float]],
                           output_path: str):
    """Radar chart comparing phenotype profiles."""
    apply_sr_style()
    labels = list(list(phenotype_profiles.values())[0].keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.cm.Set2

    for i, (pheno_name, profile) in enumerate(phenotype_profiles.items()):
        values = [profile[l] for l in labels]
        values += values[:1]
        ax.plot(angles, values, color=cmap(i), linewidth=2, label=pheno_name)
        ax.fill(angles, values, color=cmap(i), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title('Risk Phenotype Characterization', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def robustness_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       xlabel: str, output_path: str):
    """Plot robustness degradation curves (AUC/ECE vs missingness/noise)."""
    apply_sr_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for name, (x_vals, metrics) in results.items():
        if 'auc' in name.lower():
            ax1.plot(x_vals, metrics, 'o-', label=name, markersize=4)
        elif 'ece' in name.lower() or 'brier' in name.lower():
            ax2.plot(x_vals, metrics, 's-', label=name, markersize=4)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('Discrimination Robustness')
    ax1.legend(fontsize=8)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('ECE / Brier')
    ax2.set_title('Calibration Robustness')
    ax2.legend(fontsize=8)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")


def stability_heatmap(data: np.ndarray, row_labels: List[str],
                       col_labels: List[str], output_path: str,
                       title: str = 'Explanation Stability'):
    """Heatmap for explanation stability across seeds/methods."""
    apply_sr_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[FIG] Saved {output_path}")
