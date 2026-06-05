import json
from datasets import load_dataset

def manual_nebius_inspector(num_to_test=10):
    dataset = load_dataset("nebius/SWE-agent-trajectories", split="train")

    print(f"Total historical traces loaded: {len(dataset)}")
    print(f"Inspecting the first {num_to_test} action sequences manually:\n" + "="*60)

    for idx in range(min(num_to_test, len(dataset))):
        traj = dataset[idx]
        instance_id = traj.get("instance_id", f"unknown_id_{idx}")
        agent_system = traj.get("model_name", "unknown_model")
        trajectory_data = traj.get("trajectory", "")
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

        print(f"\n[{idx + 1}] Target ID: {instance_id} | Core Model: {agent_system}")
        print(f"    Total Sequential Turns: {len(action_sequence)}")
        print(f"    Raw Action Order: {action_sequence[:3]}")
        if len(action_sequence) > 3:
            print("    ...")

if __name__ == "__main__":
    manual_nebius_inspector()
