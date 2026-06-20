import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

def run_umap_pipeline():
    clustered_path = "tbf/models/clustered_fingerprints.csv"
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(f"Missing cluster dataset at: {clustered_path}")

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

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    fig, ax = plt.subplots(figsize=(7, 6))
    
    scatter = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=labels,
        cmap="viridis",
        s=3,
        alpha=0.5,
        rasterized=True
    )
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("K-Means Cluster Label", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title("2D UMAP Projection of High-Dimensional Agent Action Trajectories", fontsize=12, pad=12)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.2)
    
    ax.set_axisbelow(True)

    plot_output_dir = "figures"
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    plot_output_path = os.path.join(plot_output_dir, "umap_cluster_plot.pdf")
    plt.savefig(plot_output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_umap_pipeline()
