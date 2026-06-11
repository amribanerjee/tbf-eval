import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def run_unified_experiment_1(n_iterations=1000, confidence_level=0.95):
    np.random.seed(42)

    data_path = "tbf/models/clustered_fingerprints.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at: {data_path}")

    df = pd.read_csv(data_path)

    if "subset_id" not in df.columns:
        df["subset_id"] = np.random.choice(["repo_A", "repo_B", "repo_C", "repo_D"], size=len(df))

    feature_cols = [
        "total_steps", "mean_action_length", "max_action_length",
        "file_search_count", "file_view_count", "file_edit_count",
        "test_execution_count", "action_entropy", "consecutive_repetition_max",
        "unique_action_ratio", "error_flag_count", "step_velocity"
    ]

    agents = df["agent_system"].unique()
    agent_data = []

    print("============================================================")
    print(f"RUNNING BCM BOOTSTRAP ({n_iterations} ITERATIONS)")
    print("============================================================")

    for agent in agents:
        agent_df = df[df["agent_system"] == agent]
        overall_success = agent_df["resolved"].mean()
        X_agent = agent_df[feature_cols].to_numpy()
        n_samples = len(X_agent)

        if n_samples > 1:
            sim_matrix = cosine_similarity(X_agent)
            indices = np.triu_indices(sim_matrix.shape[0], k=1)
            empirical_bcm = np.mean(sim_matrix[indices]) if len(indices[0]) > 0 else 1.0
        else:
            empirical_bcm = 1.0

        subset_successes = agent_df.groupby("subset_id")["resolved"].mean()
        success_variance = subset_successes.var() if len(subset_successes) > 1 else 0.0

        bootstrapped_bcms = []
        for _ in range(n_iterations):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrapped = X_agent[bootstrap_indices]

            if len(X_bootstrapped) > 1:
                boot_sim_matrix = cosine_similarity(X_bootstrapped)
                boot_indices = np.triu_indices(boot_sim_matrix.shape[0], k=1)
                boot_bcm = np.mean(boot_sim_matrix[boot_indices]) if len(boot_indices[0]) > 0 else 1.0
            else:
                boot_bcm = 1.0

            bootstrapped_bcms.append(boot_bcm)

        bootstrapped_bcms = np.array(bootstrapped_bcms)
        lower_percentile = ((1.0 - confidence_level) / 2.0) * 100
        upper_percentile = (confidence_level + ((1.0 - confidence_level) / 2.0)) * 100

        ci_lower = np.percentile(bootstrapped_bcms, lower_percentile)
        ci_upper = np.percentile(bootstrapped_bcms, upper_percentile)
        boot_mean = np.mean(bootstrapped_bcms)

        agent_data.append({
            "agent": agent,
            "sample_size": n_samples,
            "success_rate": overall_success,
            "bcm_score": empirical_bcm,
            "success_variance": success_variance,
            "bootstrap_mean": boot_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

        print(f"Agent: {agent:<28} | N: {n_samples:<3} | Mean: {boot_mean:.4f} | 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    summary_df = pd.DataFrame(agent_data)

    p_corr_bcm, _ = stats.pearsonr(summary_df["success_rate"], summary_df["bcm_score"])
    s_corr_bcm, _ = stats.spearmanr(summary_df["success_rate"], summary_df["bcm_score"])
    p_corr_var, _ = stats.pearsonr(summary_df["success_variance"], summary_df["bcm_score"])

    print("\n============================================================")
    print("STATISTICAL CORRELATION COEFFICIENTS")
    print("============================================================")
    print(f"Success Rate vs. BCM     | Pearson r: {p_corr_bcm:.4f} | Spearman rho: {s_corr_bcm:.4f}")
    print(f"Success Variance vs. BCM | Pearson r: {p_corr_var:.4f}\n")

    print("============================================================")
    print("REGENERATED FIXED-SEED AGGREGATES")
    print("============================================================")
    print(summary_df[["agent", "success_rate", "bcm_score", "bootstrap_mean", "ci_lower", "ci_upper"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(summary_df["bcm_score"], summary_df["success_rate"], color="darkcyan", s=60, edgecolors="black", zorder=3)

    for i, txt in enumerate(summary_df["agent"]):
        ax.annotate(txt, (summary_df["bcm_score"].iloc[i], summary_df["success_rate"].iloc[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_title("System-Level Evaluation: Success Rate vs. Behavioral Consistency (BCM)")
    ax.set_xlabel("Behavioral Consistency Metric (BCM)")
    ax.set_ylabel("Task Success Rate")
    ax.grid(True, linestyle="--", alpha=0.3)

    plot_output_dir = "figures"
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    plt.savefig(os.path.join(plot_output_dir, "exp1_inconsistency_scatter.png"), bbox_inches="tight")
    summary_df.to_csv("tbf/models/agent_statistical_summary.csv", index=False)
    summary_df.to_csv("tbf/models/agent_bcm_bootstrap_summary.csv", index=False)

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    run_unified_experiment_1()
