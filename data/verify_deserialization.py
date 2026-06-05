import json
from datasets import load_dataset

def test_nebius_deserialization():
    dataset = load_dataset("nebius/SWE-agent-trajectories", split="train")
    first_record = dataset[0]
    trajectory_data = first_record.get("trajectory", "")

    try:
        messages = json.loads(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        print(f"Successfully deserialized full array. Total elements: {len(messages)}")

        for idx, msg in enumerate(messages[:5]):
            print(f"\n--- Message {idx} ---")
            if isinstance(msg, dict):
                print(f"Role: {msg.get('role')}")
                content = msg.get("text") if msg.get("text") is not None else msg.get("system_prompt", "")
                print(f"Content: {str(content)[:300]}")
            else:
                print(f"Raw Element Content: {str(msg)[:300]}")
    except Exception as e:
        print(f"Global array deserialization failed: {e}")
        print(f"Start of raw trajectory data: {str(trajectory_data)[:300]}")

if __name__ == "__main__":
    test_nebius_deserialization()
