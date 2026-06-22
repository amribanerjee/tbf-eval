import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

fp = pd.read_csv('tbf/models/shap_fingerprints.csv')
feat = pd.read_csv('tbf/data/engineered_features_matrix.csv')
assert len(fp) == len(feat)
if 'agent_system' not in fp.columns: fp['agent_system'] = feat['agent_system'].values
if 'instance_id' not in fp.columns: fp['instance_id'] = feat['instance_id'].values

cols = ['total_steps','mean_action_length','max_action_length','file_search_count','file_view_count','file_edit_count','test_execution_count','action_entropy','consecutive_repetition_max','unique_action_ratio','error_flag_count','step_velocity']

def bcm(f):
    X = f[cols].to_numpy()
    X = X[np.linalg.norm(X, axis=1) > 1e-9]
    if len(X) < 2: return np.nan
    S = cosine_similarity(X)
    return float(S[np.triu_indices(len(X), k=1)].mean())

for agent in sorted(fp['agent_system'].unique()):
    a = fp[fp['agent_system']==agent]
    vc = a['instance_id'].value_counts()
    tasks = vc[vc>=3].index
    scores = [bcm(a[a['instance_id']==t]) for t in tasks]
    scores = [s for s in scores if not np.isnan(s)]
    wt = float(np.mean(scores)) if scores else float('nan')
    print(f'{agent:<30} | tasks(N>=3): {len(tasks):<4} | global: {bcm(a):.4f} | within-task: {wt:.4f}')
