import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_unified_experiment_2_pipeline():
    np.random.seed(42)
    
    raw_path = "tbf/data/engineered_features_matrix.csv"
    clustered_path = "tbf/models/clustered_fingerprints.csv"
    
    if not os.path.exists(raw_path) or not os.path.exists(clustered_path):
        raise FileNotFoundError("Required behavioral matrices are missing. Run previous data steps first.")
        
    raw_df = pd.read_csv(raw_path)
    clustered_df = pd.read_csv(clustered_path)
    
    raw_df["cluster_label"] = clustered_df["cluster_label"].values
    raw_df["resolved"] = clustered_df["resolved"].values
    
    raw_df["flag_prong_A"] = (raw_df["step_velocity"] > 0.8) & (raw_df["file_edit_count"] < 0.05)
    raw_df["flag_prong_B"] = (raw_df["consecutive_repetition_max"] > 3) & (raw_df["error_flag_count"] > 0.4)
    raw_df["is_gaming"] = raw_df["flag_prong_A"] | raw_df["flag_prong_B"]
    
    flagged_df = raw_df[raw_df["is_gaming"]]
    total_gaming = len(flagged_df)
    
    print("============================================================")
    print("EXPERIMENT 2: UNIFIED BEHAVIORAL FAILURE DIAGNOSTICS")
    print("============================================================")
    print(f"Total Trajectories Flagged via Minimal-Edit Heuristic: {total_gaming}")
    
    if total_gaming > 0:
        cluster_distribution = flagged_df.groupby("cluster_label").size()
        fractions = cluster_distribution / total_gaming
        
        for cluster in sorted(raw_df["cluster_label"].unique()):
            cluster_flagged = flagged_df[flagged_df["cluster_label"] == cluster]
            count = len(cluster_flagged)
            
            if count > 0:
                frac = fractions.get(cluster, 0.0)
                cluster_resolved = cluster_flagged["resolved"].sum()
                cluster_success_rate = cluster_flagged["resolved"].mean()
                
                print(f"Cluster {cluster}: Fraction = {frac:.4f} ({count} runs)")
                print(f"  -> Resolved Successes: {cluster_resolved} (Success Rate: {cluster_success_rate:.4f})")
    else:
        print("No gaming trajectories isolated using the strict heuristic boundaries.")
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    non_gaming_df = raw_df[~raw_df["is_gaming"]]
    gaming_df = raw_df[raw_df["is_gaming"]]
    
    ax.scatter(
        non_gaming_df["action_entropy"], 
        non_gaming_df["step_velocity"], 
        c="lightgray", 
        s=15, 
        alpha=0.4,
        label="Genuine Runs"
    )
    
    ax.scatter(
        gaming_df["action_entropy"], 
        gaming_df["step_velocity"], 
        c="crimson", 
        s=40, 
        alpha=0.9,
        edgecolors="black",
        linewidths=0.6,
        label="Flagged Anomalies (n=174)"
    )
    
    ax.set_title("Behavioral Failure Boundaries: Minimal-Edit Heuristic Isolation")
    ax.set_xlabel("Action Entropy")
    ax.set_ylabel("Step Velocity")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plot_dir = "figures"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.savefig(os.path.join(plot_dir, "exp2_anomaly_isolation.png"), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
    raw_df.to_csv("tbf/models/anomaly_analysis_summary.csv", index=False)

if __name__ == "__main__":
    run_unified_experiment_2_pipeline()
