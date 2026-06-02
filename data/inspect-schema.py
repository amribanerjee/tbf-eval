from datasets import load_dataset
import json

dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
print(f"Total trajectories available: {len(dataset)}")

first_record = dataset[0]
print("\nAvailable fields in the schema:")
for key in first_record.keys():
    print(f"- {key}")
    
print("\nSample record contents:")
for key, value in first_record.items():
    val_str = str(value)
    print(f"{key}: {val_str[:200]}..." if len(val_str) > 200 else f"{key}: {value}")

dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
first_record = dataset[0]
messages_field = first_record.get("messages", "")

try:
    messages = json.loads(messages_field) if isinstance(messages_field, str) else messages_field
    print(f"Successfully deserialized full array. Total messages: {len(messages)}")
    
    for idx, msg in enumerate(messages[:5]):
        print(f"\n--- Message {idx} ---")
        print(f"Role: {msg.get('role')}")
        print(f"Message Type: {msg.get('message_type')}")
        print(f"Content: {str(msg.get('content', ''))[:300]}")
except Exception as e:
    print(f"Global array deserialization failed: {e}")
    print(f"Start of raw messages data: {str(messages_field)[:300]}")
