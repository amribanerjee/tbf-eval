import json
import re
import os
import numpy as np
import pandas as pd

def extract_features(raw_trajectory_str):
    features = {}
    try:
        messages = json.loads(raw_trajectory_str)
        if not isinstance(messages, list):
            messages = [messages]
    except Exception:
        messages = []

    total_messages = len(messages)
    features['total_steps'] = total_messages

    if total_messages == 0:
        features['mean_action_length'] = 0.0
        features['max_action_length'] = 0.0
        features['file_search_count'] = 0.0
        features['file_view_count'] = 0.0
        features['file_edit_count'] = 0.0
        features['test_execution_count'] = 0.0
        features['action_entropy'] = 0.0
        features['consecutive_repetition_max'] = 0
        features['unique_action_ratio'] = 0.0
        features['error_flag_count'] = 0.0
        features['step_velocity'] = 0.0
        return features

    lengths = [len(str(m)) for m in messages]
    features['mean_action_length'] = float(np.mean(lengths))
    features['max_action_length'] = float(np.max(lengths))

    search_patterns = [
        r'find\b.*\b(?:file|class|method|function|dir|folder)',
        r'search\b.*\b(?:directory|codebase|repo|file|text|grep)',
        r'locate\b.*\b(?:class|file|module|widgets)',
        r'\bls\b', r'\bgrep\b'
    ]
    view_patterns = [
        r'(?:view|read|examine|open|check|inspect|look\s+at)\b.*\b(?:file|content|code|source|widgets\.py|\.py|\.json|\.md)',
        r'\bcat\b'
    ]
    edit_patterns = [
        r'(?:edit|write|modify|patch|change|update|fix|replace|insert|save)\b.*\b(?:file|code|line|method|class|widgets\.py|\.py)',
        r'\bsed\b', r'<<<<<<<'
    ]
    test_patterns = [
        r'(?:run|execute|try|check|perform)\b.*\b(?:test|pytest|suite|unittest|verification)',
        r'\btest\b.*\b(?:fail|pass|run|output)'
    ]

    searches, views, edits, tests = 0, 0, 0, 0
    action_categories = []

    for m in messages:
        m_low = str(m).lower()
        category = 'other'

        is_edit = any(re.search(p, m_low) for p in edit_patterns)
        is_test = any(re.search(p, m_low) for p in test_patterns)
        is_view = any(re.search(p, m_low) for p in view_patterns)
        is_search = any(re.search(p, m_low) for p in search_patterns)

        if is_edit:
            edits += 1
            category = 'edit'
        elif is_test:
            tests += 1
            category = 'test'
        elif is_view:
            views += 1
            category = 'view'
        elif is_search:
            searches += 1
            category = 'search'

        action_categories.append(category)

    features['file_search_count'] = float(searches / total_messages)
    features['file_view_count'] = float(views / total_messages)
    features['file_edit_count'] = float(edits / total_messages)
    features['test_execution_count'] = float(tests / total_messages)

    message_strings = [str(m) for m in messages]
    _, counts = np.unique(message_strings, return_counts=True)
    probs = counts / total_messages
    features['action_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-9)))
    features['unique_action_ratio'] = float(len(counts) / total_messages)

    max_rep = 1
    current_rep = 1
    for i in range(1, len(message_strings)):
        if message_strings[i] == message_strings[i-1]:
            current_rep += 1
            if current_rep > max_rep:
                max_rep = current_rep
        else:
            current_rep = 1
    features['consecutive_repetition_max'] = max_rep

    error_patterns = [
        r'\b(?:name|type|value|key|index|attribute|zero-division|syntax|indentation)error\b',
        r'traceback', r'assert.*fail', r'failed\s+\d+\s+test', r'exception\b',
        r'issue\s+with\s+the\s+`?\w+`?\s+class'
    ]
    errors = sum(1 for m in messages if any(re.search(p, str(m).lower()) for p in error_patterns))
    features['error_flag_count'] = float(errors / total_messages)

    transitions = 0
    for i in range(1, len(action_categories)):
        if action_categories[i] != action_categories[i-1]:
            transitions += 1
    features['step_velocity'] = float(transitions / total_messages)

    return features

def batch_extract_pipeline():
    data_path = 'tbf/data/raw_behavioral_dataframe.csv'
    df = pd.read_csv(data_path)

    trajectory_col = None
    for col in df.columns:
        if col.lower() in ('raw_trajectory_sequence', 'trajectory', 'sequence', 'text'):
            trajectory_col = col
            break
    if trajectory_col is None:
        trajectory_col = df.columns[0]

    label_col = 'resolved'
    for col in df.columns:
        if col.lower() in ('resolved', 'label', 'target', 'success'):
            label_col = col
            break

    print("Extracting anchored structural features across data matrix...")
    feature_list = []
    for idx, row in df.iterrows():
        feats = extract_features(row[trajectory_col])
        feats['instance_id'] = row.get('instance_id', idx)
        feats['agent_system'] = row.get('agent_system', 'unknown')
        feats['resolved'] = row[label_col]
        feature_list.append(feats)

    features_df = pd.DataFrame(feature_list)

    for col in features_df.select_dtypes(include=[np.number]).columns:
        features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
        features_df[col] = features_df[col].fillna(0.0)

    out_dir = 'tbf/data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'engineered_features_matrix.csv')
    features_df.to_csv(out_path, index=False)
    print(f"Extraction complete! Fresh matrix saved to: {out_path}")

if __name__ == "__main__":
    batch_extract_pipeline()
