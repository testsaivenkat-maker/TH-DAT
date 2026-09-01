"""Data loading utilities for FAIR-TH-DAT Paper 2.
Loads pre-extracted .npz files produced by 00_freeze_and_extract.py.
"""
import numpy as np
import os
from typing import Dict, Optional, Tuple

def load_pakistan_data(frozen_dir: str = '../data/frozen') -> dict:
    """Load Pakistan predictions, embeddings, and metadata from frozen .npz.

    Returns dict with keys:
        probs, labels, splits, combined, gate_w, tri_w, domain_stack,
        proj, features, raw_features, trimester_ids, feature_names,
        domain_indices, train_mask, val_mask, test_mask
    """
    path = os.path.join(frozen_dir, 'pakistan_predictions.npz')
    data = dict(np.load(path, allow_pickle=True))
    # Convenience masks
    data['train_mask'] = data['splits'] == 0
    data['val_mask']   = data['splits'] == 1
    data['test_mask']  = data['splits'] == 2
    # Convert feature_names from array to list of strings
    if 'feature_names' in data:
        data['feature_names'] = list(data['feature_names'])
    if 'domain_indices' in data:
        data['domain_indices'] = list(data['domain_indices'])
    return data


def load_uganda_data(frozen_dir: str = '../data/frozen') -> dict:
    """Load Uganda predictions and metadata from frozen .npz."""
    path = os.path.join(frozen_dir, 'uganda_predictions.npz')
    data = dict(np.load(path, allow_pickle=True))
    if 'feature_names' in data:
        data['feature_names'] = list(data['feature_names'])
    return data


def load_baseline_predictions(frozen_dir: str = '../data/frozen') -> dict:
    """Load baseline model predictions on test set.

    Returns dict with keys: rf_probs, xgb_probs, lr_probs, test_labels
    """
    path = os.path.join(frozen_dir, 'baseline_predictions.npz')
    return dict(np.load(path, allow_pickle=True))


def load_multiseed_data(frozen_dir: str = '../data/frozen') -> dict:
    """Load predictions from multiple training seeds for stability analysis.

    Returns dict mapping seed -> prediction dict
    """
    seeds = {}
    for f in sorted(os.listdir(frozen_dir)):
        if f.startswith('pakistan_seed_') and f.endswith('.npz'):
            seed = int(f.replace('pakistan_seed_', '').replace('.npz', ''))
            seeds[seed] = dict(np.load(os.path.join(frozen_dir, f), allow_pickle=True))
    return seeds


def get_subgroups(raw_features: np.ndarray, feature_names: list,
                  trimester_ids: np.ndarray = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Derive subgroup boolean masks for fairness audit.

    Args:
        raw_features: (N, num_feat) unscaled features (label-encoded)
        feature_names: list of feature column names
        trimester_ids: (N,) trimester IDs (0,1,2) if available

    Returns:
        Dict mapping subgroup_attribute -> {group_name: boolean_mask}
        Example: {'Age': {'<20': mask, '20-25': mask, ...}, 'Trimester': {...}}
    """
    N = raw_features.shape[0]
    fn = [f.strip() for f in feature_names]  # strip trailing spaces
    subgroups = {}

    # --- Age groups ---
    if 'Age' in fn:
        age_col = raw_features[:, fn.index('Age')]
        subgroups['Age'] = {
            '<20':   age_col < 20,
            '20-25': (age_col >= 20) & (age_col <= 25),
            '26-30': (age_col >= 26) & (age_col <= 30),
            '31-35': (age_col >= 31) & (age_col <= 35),
            '>35':   age_col > 35,
        }

    # --- Female Education ---
    if 'Female Education' in fn:
        edu_col = raw_features[:, fn.index('Female Education')]
        # LabelEncoded values - group by numeric ranges
        # 0=Graduation, 1=Intermediate, 2=Matric, 3=Middle, 4=Post-Graduation, 5=Primary, 6=Uneducated
        # We group: Uneducated(6), Primary/Middle(3,5), Matric+(1,2), Graduate+(0,4)
        subgroups['Education'] = {
            'Uneducated':    edu_col == 6,
            'Primary/Middle': np.isin(edu_col, [3, 5]),
            'Matric+':       np.isin(edu_col, [1, 2]),
            'Graduate+':     np.isin(edu_col, [0, 4]),
        }

    # --- Socioeconomic Status (Money) ---
    money_names = ['Sufficient Money for Basic Needs', 'Sufficient Money']
    for mn in money_names:
        if mn in fn:
            money_col = raw_features[:, fn.index(mn)]
            subgroups['SES (Money)'] = {
                'Sufficient':     money_col == 1,  # Yes
                'Not Sufficient': money_col == 0,  # No
            }
            break

    # --- Trimester ---
    if trimester_ids is not None:
        subgroups['Trimester'] = {
            'T1': trimester_ids == 0,
            'T2': trimester_ids == 1,
            'T3': trimester_ids == 2,
        }

    # --- Obstetric History (Gravida) ---
    if 'Gravida' in fn:
        grav_col = raw_features[:, fn.index('Gravida')]
        subgroups['Obstetric History'] = {
            'Primigravida':  grav_col == 1,  # LabelEncoded
            'Multigravida':  grav_col == 0,
        }

    # --- Family System ---
    if 'Family System' in fn:
        fam_col = raw_features[:, fn.index('Family System')]
        subgroups['Family System'] = {
            'Joint':   fam_col == 0,
            'Nuclear': fam_col == 1,
        }

    # --- Working Status ---
    if 'Working Status' in fn:
        work_col = raw_features[:, fn.index('Working Status')]
        subgroups['Working Status'] = {
            'Housewife':    work_col == 0,
            'Working Lady': work_col == 1,
        }

    # Filter out empty groups
    for attr in list(subgroups.keys()):
        subgroups[attr] = {k: v for k, v in subgroups[attr].items() if v.sum() > 0}
        if not subgroups[attr]:
            del subgroups[attr]

    return subgroups
