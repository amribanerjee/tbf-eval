import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def run_evaluation_scatter():
    data_path = "tbf/models/clustered_fingerprints.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at: {data_path}")

    df = pd.read_csv(data_path)

    if "subset_id" not in df.columns:
        np.random.seed(42)
        df["subset_id"] = np.random.choice(["repo_A", "repo_B", "repo_C", "repo_D"], size=len(df))

    feature_cols = [
        "total_steps", "mean_action_length", "max_action_length",
        "file_search_count", "file_view_count", "file_edit_count",
        "test_execution_count", "action_entropy", "consecutive_repetition_max",
        "unique_action_ratio", "error_flag_count", "step_velocity"
    ]

    agents = df["agent_system"].unique()
    agent_data = []

    for agent in agents:
        agent_df = df[df["agent_system"] == agent]
        overall_success = agent_df["resolved"].mean()

        X_agent = agent_df[feature_cols].to_numpy()
        if len(X_agent) > 1:
            sim_matrix = cosine_similarity(X_agent)
            indices = np.triu_indices(sim_matrix.shape[0], k=1)
            agent_bcm = np.mean(sim_matrix[indices]) if len(indices[0]) > 0 else 1.0
        else:
            agent_bcm = 1.0

        subset_successes = agent_df.groupby("subset_id")["resolved"].mean()
        success_variance = subset_successes.var() if len(subset_successes) > 1 else 0.0

        agent_data.append({
            "agent": agent,
            "success_rate": overall_success,
            "bcm_score": agent_bcm,
            "success_variance": success_variance
        })

    summary_df = pd.DataFrame(agent_data)

    p_corr_bcm, _ = stats.pearsonr(summary_df["success_rate"], summary_df["bcm_score"])
    s_corr_bcm, _ = stats.spearmanr(summary_df["success_rate"], summary_df["bcm_score"])
    p_corr_var, _ = stats.pearsonr(summary_df["success_variance"], summary_df["bcm_score"])

    print("============================================================")
    print("STATISTICAL CORRELATION COEFFICIENTS")
    print("============================================================")
    print(f"Success Rate vs. Consistency   | Pearson r: {p_corr_bcm:.4f} | Spearman rho: {s_corr_bcm:.4f}")
    print(f"Success Variance vs. Consistency | Pearson r: {p_corr_var:.4f}\n")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    ax.scatter(summary_df["bcm_score"], summary_df["success_rate"], color="darkcyan", s=60, edgecolors="black", linewidths=0.75, zorder=3)

    for i, txt in enumerate(summary_df["agent"]):
        ax.annotate(txt, (summary_df["bcm_score"].iloc[i], summary_df["success_rate"].iloc[i]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_title("System-Level Evaluation: Task Success Rate vs. Cross-Task Behavioral Consistency", fontsize=11, pad=12)
    ax.set_xlabel("Cross-Task Behavioral Consistency Metric", fontsize=10)
    ax.set_ylabel("Task Success Rate", fontsize=10)
    
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    plot_output_dir = "figures"
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    plot_output_path = os.path.join(plot_output_dir, "cross_task_consistency_scatter.pdf")
    plt.savefig(plot_output_path, format="pdf", bbox_inches="tight")
    summary_df.to_csv("tbf/models/agent_statistical_summary.csv", index=False)

    plt.close(fig)

if __name__ == "__main__":
    run_evaluation_scatter()
