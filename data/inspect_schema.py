from datasets import load_dataset

def check_nebius_schema():
    print("Loading nebius/SWE-agent-trajectories dataset from Hugging Face...")
    dataset = load_dataset("nebius/SWE-agent-trajectories", split="train")
    print(f"Total trajectories available: {len(dataset)}")

    print("\nAvailable fields in the schema:")
    for key in dataset.features.keys():
        print(f"- {key}")

    print("\nSample record contents:")
    first_record = dataset[0]
    for key, value in first_record.items():
        val_str = str(value)
        print(f"{key}: {val_str[:200]}..." if len(val_str) > 200 else f"{key}: {value}")

if __name__ == "__main__":
    check_nebius_schema()
