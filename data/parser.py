import json
import glob

def manual_trajectory_inspector(data_dir='tbf/data', num_to_test=10):
    json_files = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    all_extracted_trajectories = []
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'instances' in data:
                data = data['instances']
            all_extracted_trajectories.extend(data)
            
    print(f"Total historical traces loaded across systems: {len(all_extracted_trajectories)}")
    print(f"Inspecting the first {num_to_test} action sequences manually:\n" + "="*60)
    
    for idx, traj in enumerate(all_extracted_trajectories[:num_to_test]):
        instance_id = traj.get('instance_id', f'unknown_id_{idx}')
        agent_system = traj.get('model', 'unknown_agent')
        history = traj.get('history', [])
        
        action_sequence = [step.get('action', '').strip() for step in history if isinstance(step, dict)]
        
        print(f"\n[{idx + 1}] Target Patch: {instance_id} | Core Model: {agent_system}")
        print(f"    Total Sequential Turns: {len(action_sequence)}")
        print(f"    Raw Action Order: {action_sequence}")

if __name__ == "__main__":
    manual_trajectory_inspector()
