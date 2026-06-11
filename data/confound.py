import os
import pandas as pd
import numpy as np

def run_comprehensive_confounder_audit():
    raw_path = "tbf/data/engineered_features_matrix.csv"
    clustered_path = "tbf/models/clustered_fingerprints.csv"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Required matrix file missing at: {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    if os.path.exists(clustered_path):
        c_df = pd.read_csv(clustered_path)
        df["cluster_label"] = c_df["cluster_label"].values
    else:
        df["cluster_label"] = 0

    agents = sorted(list(df["agent_system"].unique()))
    task_maps = {agent: set(df[df["agent_system"] == agent]["instance_id"].unique()) for agent in agents}
    
    def compute_bcm_score(subset):
        if len(subset) < 2:
            return 0.0
        val = 1.0 - subset["action_entropy"].std()
        return float(np.clip(val, 0.0, 1.0))

    global_bcm_ref = {
        "claude-3-5-sonnet-20241022": 0.8327,
        "claude-3-7-sonnet-20250219": 0.7657,
        "gpt-4o-2024-08-06": 0.8143,
        "swe-agent-llama-8b": 0.0862,
        "swe-agent-llama-70b": 0.0649,  
        "swe-agent-llama-405b": 0.0709
    }

    print("============================================================")
    print("1. DYNAMIC PAIRWISE SHARED-TASK COUNT MATRIX (6x6)")
    print("============================================================")
    matrix_data = []
    for a1 in agents:
        row = {}
        for a2 in agents:
            overlap = len(task_maps[a1].intersection(task_maps[a2]))
            row[a2] = overlap
        matrix_data.append(row)
        
    overlap_matrix = pd.DataFrame(matrix_data, index=agents, columns=agents)
    print(overlap_matrix.to_string())
    
    print("\n============================================================")
    print("2. TARGETED PAIRWISE DIFFICULTY CONTROL DIAGNOSTICS")
    print("============================================================")
    target_pairs = [
        ("gpt-4o-2024-08-06", "swe-agent-llama-405b"),
        ("claude-3-5-sonnet-20241022", "swe-agent-llama-405b"),
        ("claude-3-7-sonnet-20250219", "swe-agent-llama-405b")
    ]
    
    for p1, p2 in target_pairs:
        if p1 in task_maps and p2 in task_maps:
            shared = task_maps[p1].intersection(task_maps[p2])
            print(f"Intersection for [{p1}] vs [{p2}]:")
            print(f"  -> Shared Task Count = {len(shared)}")
            
            if len(shared) > 0:
                shared_df = df[df["instance_id"].isin(shared)]
                p1_shared = shared_df[shared_df["agent_system"] == p1]
                p2_shared = shared_df[shared_df["agent_system"] == p2]
                
                print(f"  -> Controlled BCM {p1} (N={len(p1_shared)}): {compute_bcm_score(p1_shared):.4f}")
                print(f"  -> Controlled BCM {p2} (N={len(p2_shared)}): {compute_bcm_score(p2_shared):.4f}")
            else:
                print("  -> Cannot recompute cross-agent BCM due to zero shared tasks.")
            print("-" * 60)
            
    print("\n============================================================")
    print("3. WITHIN-TASK BCM CONTROL (STRICT COGNITIVE CONTROL)")
    print("============================================================")
    for agent in agents:
        agent_df = df[df["agent_system"] == agent]
        counts = agent_df["instance_id"].value_counts()
        valid_tasks = counts[counts >= 3].index
        
        task_scores = []
        for tid in valid_tasks:
            t_subset = agent_df[agent_df["instance_id"] == tid]
            task_scores.append(compute_bcm_score(t_subset))
            
        avg_within_task = np.mean(task_scores) if len(task_scores) > 0 else 0.0
        g_bcm = global_bcm_ref.get(agent, 0.0)
        
        print(f"Agent: {agent}")
        print(f"  -> Tasks with >=3 attempts: {len(valid_tasks)}")
        print(f"  -> Global Baseline BCM:    {g_bcm:.4f}")
        print(f"  -> Avg Within-Task BCM:    {avg_within_task:.4f}")
        
    print("\n============================================================")
    print("4. REAL-TIME TRAJECTORY SOURCE SHEET INVENTORY")
    print("============================================================")
    print(f"Total entries parsed in active workspace matrix: {len(df)}")
    for agent in agents:
        a_count = len(df[df["agent_system"] == agent])
        u_tasks = len(task_maps[agent])
        print(f"  -> {agent:<30} | Trajectories: {a_count:<6} | Tasks: {u_tasks}")
    print("============================================================")

if __name__ == "__main__":
    run_comprehensive_confounder_audit()
