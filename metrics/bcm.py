import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_bcm(matrix):
    if matrix.shape[0] < 2:
        return 1.0
    sim_matrix = cosine_similarity(matrix)
    indices = np.triu_indices(sim_matrix.shape[0], k=1)
    if len(indices[0]) == 0:
        return 1.0
    return np.mean(sim_matrix[indices])

def run_bcm_pipeline():
    single_fingerprint = np.array([0.5, -0.2, 0.1, 0.9])
    synthetic_matrix = np.vstack([single_fingerprint] * 3)
    synthetic_bcm = calculate_bcm(synthetic_matrix)

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

    results = []
    for (system, outcome), group in df.groupby(["agent_system", "resolved"]):
        if len(group) > 1:
            matrix = group[feature_cols].to_numpy()
            score = calculate_bcm(matrix)
            results.append({
                "Agent System": system,
                "Outcome": f"Failure (0)" if outcome == 0 else f"Success (1)",
                "BCM Score": round(score, 4),
                "Sample Size": len(group)
            })

    bcm_df = pd.DataFrame(results)
    
    if not bcm_df.empty:
        bcm_df.sort_values(by=["Agent System", "Outcome"], ascending=[True, True], inplace=True)
        bcm_df.reset_index(drop=True, inplace=True)
    
    output_dir = "tbf/models"
    os.makedirs(output_dir, exist_ok=True)
    bcm_df.to_csv(os.path.join(output_dir, "bcm_results.csv"), index=False)

    print("============================================================")
    print("BCM COMPARISON TABLE")
    print("============================================================")
    print(bcm_df.to_string(index=False))

if __name__ == "__main__":
    run_bcm_pipeline()
