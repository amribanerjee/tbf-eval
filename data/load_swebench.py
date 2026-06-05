import os
import json
import pandas as pd
from datasets import load_dataset

def process_swe_smith_subset():
    print("Loading SWE-smith-trajectories database layer...")
    try:
        dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    except Exception:
        dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="train")
        
    compiled_records = []

    for traj in dataset:
        instance_id = traj.get("instance_id", "")
        agent_system = traj.get("model", "swe_smith_claude_3.7")

        success_status = traj.get("resolved", False)
        binary_outcome = 1 if success_status in [True, "True", 1] else 0

        messages_field = traj.get("messages", "")
        action_sequence = []

        try:
            messages = json.loads(messages_field) if isinstance(messages_field, str) else messages_field
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("message_type") == "action":
                        content = msg.get("content", "")
                        if content:
                            action_sequence.append(content)
        except Exception:
            pass

        compiled_records.append({
            "instance_id": instance_id,
            "agent_system": agent_system,
            "total_steps": len(action_sequence),
            "resolved": binary_outcome,
            "raw_trajectory_sequence": json.dumps(action_sequence)
        })
    
    return pd.DataFrame(compiled_records)

def process_nebius_subset():
    print("Loading nebius/SWE-agent-trajectories database layer...")
    dataset = load_dataset("nebius/SWE-agent-trajectories", split="train")
    df_nebius = pd.DataFrame(dataset)

    df_nebius['resolved_outcome'] = df_nebius['target'].apply(lambda x: 1 if x in [True, "True", "True\n", 1] else 0)

    breakdown = df_nebius.groupby('model_name').size().reset_index(name='counts')
    valid_systems = breakdown[breakdown['counts'] >= 500]['model_name'].tolist()
    
    print(f"Systems meeting minimum volume threshold (>= 500): {valid_systems}")
    selected_systems = valid_systems[:4]
    print(f"Selected model systems for multi-agent profile: {selected_systems}")

    compiled_records = []
    for sys in selected_systems:
        df_sub = df_nebius[df_nebius['model_name'] == sys]
        sample_size = min(2000, len(df_sub))
        df_sampled = df_sub.sample(n=sample_size, random_state=42)

        for _, row in df_sampled.iterrows():
            instance_id = row.get("instance_id", "")
            agent_system = row['model_name']
            binary_outcome = int(row['resolved_outcome'])
            trajectory_data = row['trajectory']
            action_sequence = []

            try:
                messages = json.loads(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get("text") if msg.get("text") is not None else msg.get("system_prompt", "")
                            content_str = str(content).strip()
                            if content_str and content_str != "None":
                                action_sequence.append(content_str)
                        elif isinstance(msg, str):
                            action_sequence.append(msg)
            except Exception:
                pass

            compiled_records.append({
                "instance_id": instance_id,
                "agent_system": agent_system,
                "total_steps": len(action_sequence),
                "resolved": binary_outcome,
                "raw_trajectory_sequence": json.dumps(action_sequence)
            })

    return pd.DataFrame(compiled_records)

def run_dataset_generation_pipeline():
    os.makedirs("tbf/data", exist_ok=True)

    df_smith = process_swe_smith_subset()
    df_smith_sampled = df_smith.sample(n=min(4000, len(df_smith)), random_state=42)

    df_nebius_sampled = process_nebius_subset()

    print("\nMerging subsets into final combined multi-agent dataframe...")
    combined_df = pd.concat([df_smith_sampled, df_nebius_sampled], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\nFinal Multi-Agent System Count Profile:")
    print(combined_df['agent_system'].value_counts())

    output_path = "tbf/data/raw_behavioral_dataframe.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nProcessing finished. Combined raw behavioral matrix stored at: {output_path}")

if __name__ == "__main__":
    run_dataset_generation_pipeline()
