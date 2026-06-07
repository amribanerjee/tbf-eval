import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def calculate_within_cluster_bcm(shap_matrix, cluster_labels, k):
    cluster_bcm_scores = {}
    for cluster_id in range(k):
        mask = (cluster_labels == cluster_id)
        matrix = shap_matrix[mask]
        if matrix.shape[0] < 2:
            cluster_bcm_scores[cluster_id] = 1.0
            continue
        sim_matrix = cosine_similarity(matrix)
        indices = np.triu_indices(sim_matrix.shape[0], k=1)
        cluster_bcm_scores[cluster_id] = np.mean(sim_matrix[indices]) if len(indices[0]) > 0 else 1.0
    return cluster_bcm_scores

def run_clustering_pipeline(chosen_k=3):
    shap_path = "tbf/models/shap_fingerprints.csv"
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"Missing SHAP fingerprints file at: {shap_path}")

    df = pd.read_csv(shap_path)
    
    feature_cols = [
        "total_steps", "mean_action_length", "max_action_length",
        "file_search_count", "file_view_count", "file_edit_count",
        "test_execution_count", "action_entropy", "consecutive_repetition_max",
        "unique_action_ratio", "error_flag_count", "step_velocity"
    ]
    
    X = df[feature_cols].to_numpy()

    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        print(f"k = {k} | WCSS (Inertia): {km.inertia_:.4f}")

    final_km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df["cluster_label"] = final_km.fit_predict(X)

    cluster_bcm = calculate_within_cluster_bcm(X, df["cluster_label"].to_numpy(), chosen_k)
    
    for cluster_id in range(chosen_k):
        c_group = df[df["cluster_label"] == cluster_id]
        success_rate = c_group["resolved"].mean()
        
        print(f"Cluster {cluster_id} | Size: {len(c_group)} | Success Rate: {success_rate:.4f} | Within-Cluster BCM: {cluster_bcm[cluster_id]:.4f}")
        
        mean_shaps = c_group[feature_cols].mean()
        for feat, val in mean_shaps.items():
            print(f"  {feat}: {val:.4f}")
            
        comp = c_group["agent_system"].value_counts(normalize=True)
        for agent, pct in comp.items():
            print(f"  {agent}: {pct*100:.2f}%")
            
    df.to_csv("tbf/models/clustered_fingerprints.csv", index=False)

if __name__ == "__main__":
    run_clustering_pipeline(chosen_k=3)
