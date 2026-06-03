import json
import re
import numpy as np
import pandas as pd
import os

def extract_features(raw_trajectory_str):
    features = {}
    
    try:
        actions = json.loads(raw_trajectory_str)
    except Exception:
        actions = []
        
    total_actions = len(actions)
    features['total_steps'] = total_actions
    
    if total_actions == 0:
        features['mean_action_length'] = 0.0
        features['max_action_length'] = 0.0
        features['file_search_count'] = 0
        features['file_view_count'] = 0
        features['file_edit_count'] = 0
        features['test_execution_count'] = 0
        features['action_entropy'] = 0.0
        features['consecutive_repetition_max'] = 0
        features['unique_action_ratio'] = 0.0
        features['error_flag_count'] = 0
        features['step_velocity'] = 0.0
        return features

    lengths = [len(a) for a in actions]
    features['mean_action_length'] = float(np.mean(lengths))
    features['max_action_length'] = float(np.max(lengths))
    
    search_patterns = [r'find', r'search', r'grep', r'locate', r'ls']
    view_patterns = [r'view', r'cat', r'open', r'read', r'display']
    edit_patterns = [r'edit', r'write', r'modify', r'patch', r'sed']
    test_patterns = [r'test', r'pytest', r'run', r'execute']
    
    searches, views, edits, tests = 0, 0, 0, 0
    action_categories = []
    
    for a in actions:
        a_low = a.lower()
        category = 'other'
        if any(re.search(p, a_low) for p in search_patterns):
            searches += 1
            category = 'search'
        if any(re.search(p, a_low) for p in view_patterns):
            views += 1
            category = 'view'
        if any(re.search(p, a_low) for p in edit_patterns):
            edits += 1
            category = 'edit'
        if any(re.search(p, a_low) for p in test_patterns):
            tests += 1
            category = 'test'
        action_categories.append(category)
            
    features['file_search_count'] = searches
    features['file_view_count'] = views
    features['file_edit_count'] = edits
    features['test_execution_count'] = tests
    
    _, counts = np.unique(actions, return_counts=True)
    probs = counts / total_actions
    features['action_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-9)))
    
    max_rep = 1
    current_rep = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i-1]:
            current_rep += 1
            if current_rep > max_rep:
                max_rep = current_rep
        else:
            current_rep = 1
    features['consecutive_repetition_max'] = max_rep
    features['unique_action_ratio'] = float(len(counts) / total_actions)
    
    error_patterns = [r'error', r'fail', r'exception', r'traceback', r'invalid']
    errors = sum(1 for a in actions if any(re.search(p, a.lower()) for p in error_patterns))
    features['error_flag_count'] = errors
    
    transitions = 0
    for i in range(1, len(action_categories)):
        if action_categories[i] != action_categories[i-1]:
            transitions += 1
    features['step_velocity'] = float(transitions / total_actions)
    
    return features

def batch_extract_pipeline(csv_path='tbf/data/raw_behavioral_dataframe.csv'):
    df = pd.read_csv(csv_path)
    feature_list = []
    
    print("Beginning feature extraction over full matrix rows...")
    for idx, row in df.iterrows():
        raw_str = row['raw_trajectory_sequence']
        feats = extract_features(raw_str)
        feats['instance_id'] = row['instance_id']
        feats['agent_system'] = row['agent_system']
        feats['resolved'] = row['resolved']
        feature_list.append(feats)
        
    features_df = pd.DataFrame(feature_list)
    
    for col in features_df.select_dtypes(include=[np.number]).columns:
        features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
        if features_df[col].isnull().any():
            features_df[col] = features_df[col].fillna(0.0)
            
    out_dir = 'tbf/data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'engineered_features_matrix.csv')
    features_df.to_csv(out_path, index=False)
    
    print(f"Pipeline complete. Formatted matrix saved to: {out_path}")
    return features_df

if __name__ == "__main__":
    batch_extract_pipeline()
