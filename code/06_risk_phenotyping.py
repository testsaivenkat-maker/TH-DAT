"""
Risk Phenotyping for FAIR-TH-DAT
Identifies distinct patient risk profiles using combined embeddings,
UMAP dimensionality reduction, and clustering (HDBSCAN/KMeans).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("umap-learn not installed. Dimensionality reduction will use PCA instead.")
    from sklearn.decomposition import PCA

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    warnings.warn("hdbscan not installed. Skipping HDBSCAN clustering.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-dir', type=str, required=True, help="Directory containing .npz files")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--t-low', type=float, default=0.2, help="Low risk threshold")
    parser.add_argument('--t-high', type=float, default=0.5, help="High risk threshold")
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def reduce_dimensions(embeddings, n_components=10, seed=42):
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=seed)
        return reducer.fit_transform(embeddings)
    else:
        pca = PCA(n_components=n_components, random_state=seed)
        return pca.fit_transform(embeddings)

def cluster_embeddings(reduced_emb, seed=42):
    clusters = {}
    
    # HDBSCAN
    if HAS_HDBSCAN:
        hdb = hdbscan.HDBSCAN(min_cluster_size=50)
        clusters['hdbscan'] = hdb.fit_predict(reduced_emb)
        
    # K-Means
    best_k = 3
    best_sil = -1
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        labels = km.fit_predict(reduced_emb)
        sil = silhouette_score(reduced_emb, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            clusters['kmeans'] = labels
            
    print(f"Selected K={best_k} for K-Means (Silhouette={best_sil:.3f})")
    
    # GMM
    gmm = GaussianMixture(n_components=best_k, random_state=seed)
    clusters['gmm'] = gmm.fit_predict(reduced_emb)
    
    return clusters

def characterize_clusters(data, labels, t_low, t_high):
    df_list = []
    y = data['y']
    y_prob = data['y_prob']
    gate_w = data['gate_w']
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = (labels == lbl)
        n_cluster = mask.sum()
        pct = n_cluster / len(labels) * 100
        
        y_c = y[mask]
        y_prob_c = y_prob[mask]
        gate_w_c = gate_w[mask]
        
        dep_prev = np.mean(y_c)
        mean_prob = np.mean(y_prob_c)
        
        risk_low = np.mean(y_prob_c < t_low)
        risk_med = np.mean((y_prob_c >= t_low) & (y_prob_c < t_high))
        risk_high = np.mean(y_prob_c >= t_high)
        
        mean_gw = np.mean(gate_w_c, axis=0)
        
        df_list.append({
            'cluster': lbl,
            'size': n_cluster,
            'percent': pct,
            'depression_prev': dep_prev,
            'mean_pred_prob': mean_prob,
            'pct_low_risk': risk_low,
            'pct_med_risk': risk_med,
            'pct_high_risk': risk_high,
            'gw_domain_0': mean_gw[0] if len(mean_gw)>0 else np.nan,
            'gw_domain_1': mean_gw[1] if len(mean_gw)>1 else np.nan,
            'gw_domain_2': mean_gw[2] if len(mean_gw)>2 else np.nan,
        })
        
    return pd.DataFrame(df_list)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    primary_file = os.path.join(args.frozen_dir, 'pakistan_predictions.npz')
    if not os.path.exists(primary_file):
        print(f"Error: Primary file {primary_file} not found.")
        return
        
    print(f"Loading {primary_file}...")
    data = np.load(primary_file)
    
    if 'z_f_stack' in data:
        embeddings = data['z_f_stack']
        # Reshape if necessary, assume (N, domains, dim) -> (N, domains*dim)
        if len(embeddings.shape) == 3:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
    else:
        print("Error: embeddings (z_f_stack) not found in npz. Using gate weights as proxy embeddings.")
        embeddings = data['gate_w']
        
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Reducing dimensions for clustering...")
    emb_10d = reduce_dimensions(embeddings, n_components=10, seed=args.seed)
    
    print("Clustering...")
    clusters = cluster_embeddings(emb_10d, seed=args.seed)
    
    main_labels = clusters.get('kmeans', next(iter(clusters.values())))
    
    print("Characterizing phenotypes...")
    df_char = characterize_clusters(data, main_labels, args.t_low, args.t_high)
    out_csv = os.path.join(args.output_dir, 'phenotype_characterization.csv')
    df_char.to_csv(out_csv, index=False)
    print(f"Saved characterization to {out_csv}")
    
    # 2D reduction for visualization
    print("Reducing to 2D for visualization...")
    emb_2d = reduce_dimensions(embeddings, n_components=2, seed=args.seed)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=main_labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Patient Risk Phenotypes (2D Projection)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'phenotypes_2d.png'), dpi=300)
    plt.close()
    
    print("Phenotyping complete.")

if __name__ == '__main__':
    main()
