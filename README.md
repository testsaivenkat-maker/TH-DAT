# FAIR-TH-DAT: Trustworthiness Assessment of TH-DAT

Eight-dimensional post-hoc trustworthiness evaluation of the Trimester-Aware Hierarchical Domain-Attention Transformer (TH-DAT) for antenatal depression risk stratification.

## Structure

- data/frozen/ - Frozen model predictions (.npz) and weights (.pt)
- outputs/figures/ - Original evaluation figures (9 PNGs)
- outputs/figures_v2/ - Extended figures: ROC, PR, confusion matrices, ablation, error analysis (6 PNGs)
- outputs/tables_v2/ - All evaluation metrics as CSV files
- paper/ - Manuscript PDF
- code/ - TH-DAT training and evaluation scripts

## Reproducibility

All results are reproducible with seed=42. Python 3.14, PyTorch, scikit-learn, matplotlib required.

## Key Results (Pakistan Test Set, n=2,102)

| Model | AUC | F1 | MCC | PR-AUC |
|-------|-----|----|-----|--------|
| TH-DAT | 0.714 | 0.788 | 0.310 | 0.802 |
| Random Forest | 0.955 | 0.916 | 0.748 | 0.973 |
| XGBoost | 0.883 | 0.890 | 0.669 | 0.902 |

Clinical Trustworthiness Index (CTI) = 0.778

## Citation

Under review at Scientific Reports (2026).

## License

MIT
