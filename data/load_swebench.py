import pandas as pd
import json
import glob

def batch_process_to_dataframe(data_dir='tbf/data'):
    json_files = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    compiled_records = []
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'instances' in data:
                data = data['instances']
                
            for traj in data:
                instance_id = traj.get('instance_id', '')
                agent_system = traj.get('model', '')
                
                success_status = traj.get('success', traj.get('resolved', False))
                binary_outcome = 1 if success_status in [True, 'True', 1, 'success'] else 0
                
                history = traj.get('history', [])
                action_sequence = [step.get('action', '') for step in history if isinstance(step, dict)]
                
                compiled_records.append({
                    'instance_id': instance_id,
                    'agent_system': agent_system,
                    'total_steps': len(action_sequence),
                    'resolved': binary_outcome,
                    'raw_trajectory_sequence': json.dumps(action_sequence)
                })
                
    df = pd.DataFrame(compiled_records)
    output_path = 'tbf/data/raw_behavioral_dataframe.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} trajectories.")
    print(f"Saved dataset to: {output_path}")
    
    return df

if __name__ == "__main__":
    batch_process_to_dataframe()
