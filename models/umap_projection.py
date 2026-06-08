import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

def run_umap_pipeline():
    clustered_path = "tbf/models/clustered_fingerprints.csv"
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(f"Missing cluster dataset at: {clustered_path}. Run clustering.py first.")

    df = pd.read_csv(clustered_path)
    
    feature_cols = [
        "total_steps", "mean_action_length", "max_action_length",
        "file_search_count", "file_view_count", "file_edit_count",
        "test_execution_count", "action_entropy", "consecutive_repetition_max",
        "unique_action_ratio", "error_flag_count", "step_velocity"
    ]
    
    X = df[feature_cols].to_numpy()
    labels = df["cluster_label"].to_numpy()

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    
    X_umap = reducer.fit_transform(X)
    
    np.save("tbf/models/umap_2d_projection.npy", X_umap)

    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(
        X_umap[:, 0], 
        X_umap[:, 1], 
        c=labels, 
        cmap="viridis", 
        s=4, 
        alpha=0.6
    )
    plt.colorbar(scatter, label="K-Means Cluster Label")
    plt.title("2D UMAP Projection of Agent Behavior SHAP Fingerprints")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plot_output_path = "figures/umap_cluster_plot.png"
    plt.savefig(plot_output_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run_umap_pipeline()
