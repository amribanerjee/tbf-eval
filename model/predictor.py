import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def run_complete_pipeline():
    feature_matrix_path = "tbf/data/engineered_features_matrix.csv"
    if not os.path.exists(feature_matrix_path):
        raise FileNotFoundError(f"Missing feature matrix at: {feature_matrix_path}")

    df = pd.read_csv(feature_matrix_path)

    feature_cols = [
        "total_steps", "mean_action_length", "max_action_length",
        "file_search_count", "file_view_count", "file_edit_count",
        "test_execution_count", "action_entropy", "consecutive_repetition_max",
        "unique_action_ratio", "error_flag_count", "step_velocity"
    ]
    target_col = "resolved"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    oof_proba = np.zeros(len(df))
    oof_shap = np.zeros_like(X, dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        clf_lgb.fit(X_train, y_train)
        
        p_val = clf_lgb.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = p_val

        explainer = shap.TreeExplainer(clf_lgb)
        shap_vals = explainer.shap_values(X_val)

        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                shap_vals = shap_vals[1]

        oof_shap[val_idx] = shap_vals
        print(f"Fold {fold + 1} processing completed.")

    print("\n" + "="*60)
    print("GLOBAL PERFORMANCE RESULTS (PREDICTOR)")
    print("="*60)
    
    optimized_threshold = 0.370
    predictions = (oof_proba >= optimized_threshold).astype(int)
    
    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, oof_proba)
    
    print(f"Mean ROC-AUC (5-Fold): {roc_auc:.4f}")
    print(f"Optimized Threshold  : {optimized_threshold:.3f}\n")
    print(classification_report(y, predictions, target_names=["failure (0)", "success (1)"]))

    print("="*60)
    print("FIRST 5 ROWS OF OUT-OF-FOLD SHAP MATRIX")
    print("="*60)
    print(oof_shap[:5])

    output_dir = "tbf/models"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(output_dir, "oof_shap_matrix.npy"), oof_shap)
    
    shap_df = pd.DataFrame(oof_shap, columns=feature_cols)
    shap_df["resolved"] = y
    shap_df.to_csv(os.path.join(output_dir, "shap_fingerprints.csv"), index=False)
    print(f"\nSaved shap_fingerprints.csv successfully to {output_dir}")

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS: CONSECUTIVE_REPETITION_MAX")
    print("="*60)
    print(df["consecutive_repetition_max"].describe())

if __name__ == "__main__":
    run_complete_pipeline()
