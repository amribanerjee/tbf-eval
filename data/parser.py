import json
from datasets import load_dataset

def manual_trajectory_inspector(num_to_test=10):
    dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    
    print(f"Total historical traces loaded: {len(dataset)}")
    print(f"Inspecting the first {num_to_test} action sequences manually:\n" + "="*60)
    
    for idx in range(min(num_to_test, len(dataset))):
        traj = dataset[idx]
        instance_id = traj.get("instance_id", f"unknown_id_{idx}")
        agent_system = traj.get("model", "unknown_model")
        
        messages_field = traj.get("messages", "")
        action_sequence = []
        
        try:
            messages = json.loads(messages_field) if isinstance(messages_field, str) else messages_field
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("message_type") == "action":
                        content = msg.get("content", "").strip()
                        if content:
                            action_sequence.append(content)
        except Exception:
            pass
                    
        print(f"\n[{idx + 1}] Target Patch: {instance_id} | Core Model: {agent_system}")
        print(f"    Total Sequential Turns: {len(action_sequence)}")
        print(f"    Raw Action Order: {action_sequence[:3]}")
        if len(action_sequence) > 3:
            print("    ...")

if __name__ == "__main__":
    manual_trajectory_inspector()
