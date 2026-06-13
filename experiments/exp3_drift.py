import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

fp = pd.read_csv('tbf/models/shap_fingerprints.csv')
feat_df = pd.read_csv('tbf/data/engineered_features_matrix.csv')

assert len(feat_df) == len(fp), f'row mismatch {len(feat_df)} vs {len(fp)}'

fp['instance_id'] = feat_df['instance_id'].values
fp['agent_system'] = feat_df['agent_system'].values
fp['resolved'] = feat_df['resolved'].values

feat = ['total_steps','mean_action_length','max_action_length','file_search_count','file_view_count','file_edit_count','test_execution_count','action_entropy','consecutive_repetition_max','unique_action_ratio','error_flag_count','step_velocity']

def bcm(sub):
    X = sub[feat].to_numpy()
    norms = np.linalg.norm(X, axis=1)
    X = X[norms > 1e-9]
    if len(X) < 2: return np.nan
    S = cosine_similarity(X)
    iu = np.triu_indices(len(X), k=1)
    return float(S[iu].mean())

task_resolution = fp.groupby('instance_id')['resolved'].mean().to_frame('difficulty_score')

q33 = task_resolution['difficulty_score'].quantile(1/3)
q66 = task_resolution['difficulty_score'].quantile(2/3)

def assign_bin(score):
    if score <= q33:
        return 'hard'
    elif score <= q66:
        return 'medium'
    else:
        return 'easy'

task_resolution['bin'] = task_resolution['difficulty_score'].apply(assign_bin)

fp = fp.merge(task_resolution[['bin']], left_on='instance_id', right_index=True)

print("==================================================================")
print("EXPERIMENT 3: BEHAVIORAL DRIFT AND SUCCESS RATE BY DIFFICULTY BIN")
print("==================================================================")
print(f"Bin Thresholds: Easy (>{round(q66,4)}), Medium ({round(q33,4)} to {round(q66,4)}), Hard (<={round(q33,4)})\n")

bins_order = ['easy', 'medium', 'hard']

for a in sorted(fp['agent_system'].unique()):
    print(f"Agent: {a}")
    agent_sub = fp[fp['agent_system'] == a]
    
    for b in bins_order:
        bin_sub = agent_sub[agent_sub['bin'] == b]
        n_traj = len(bin_sub)
        
        if n_traj > 0:
            bin_bcm = bcm(bin_sub)
            bin_success = bin_sub['resolved'].mean()
            print(f"  -> {b.upper():<6} | Trajectories: {n_traj:<5} | BCM: {round(bin_bcm, 4):<7} | Success Rate: {round(bin_success, 4)}")
        else:
            print(f"  -> {b.upper():<6} | Trajectories: 0     | BCM: NaN     | Success Rate: NaN")
    print("-" * 66)
