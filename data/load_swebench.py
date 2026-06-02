import pandas as pd
import json
import os
from datasets import load_dataset

def batch_process_to_dataframe():
    dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    compiled_records = []
    
    os.makedirs("tbf/data", exist_ok=True)
    
    for traj in dataset:
        instance_id = traj.get("instance_id", "")
        agent_system = traj.get("model", "")
        
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
        
    df = pd.DataFrame(compiled_records)
    output_path = "tbf/data/raw_behavioral_dataframe.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} trajectories.")
    print(f"Saved dataset to: {output_path}")
    
    return df

if __name__ == "__main__":
    batch_process_to_dataframe()
