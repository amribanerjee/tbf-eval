import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def run_prediction_pipeline():
    matrix_path = 'tbf/data/engineered_features_matrix.csv'
    if not os.path.exists(matrix_path):
        print(f"Error: Feature matrix not found at {matrix_path}. Run extract_features.py first.")
        return

    df = pd.read_csv(matrix_path)

    y = df['resolved'].astype(int).values
    neg_support = int((y == 0).sum())
    pos_support = int((y == 1).sum())
    scale_weight = neg_support / pos_support

    feature_cols = [
        'total_steps', 'mean_action_length', 'max_action_length',
        'file_search_count', 'file_view_count', 'file_edit_count',
        'test_execution_count', 'action_entropy', 'consecutive_repetition_max',
        'unique_action_ratio', 'error_flag_count', 'step_velocity'
    ]

    X = df[feature_cols].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    oof_proba = np.zeros(len(y))
    importances_accumulator = np.zeros(len(feature_cols))

    lgb_params = {
        "n_estimators": 1500,
        "learning_rate": 0.005,
        "num_leaves": 15,
        "max_depth": 4,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_samples": 30,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "scale_pos_weight": scale_weight,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print("Training LightGBM on verified structural behavioral dimensions...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        clf_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        p_val = clf_lgb.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = p_val

        fold_auc = roc_auc_score(y_val, p_val)
        fold_aucs.append(fold_auc)

        importances_accumulator += clf_lgb.booster_.feature_importance(importance_type="gain")
        print(f"  Fold {fold_idx}  |  Validation ROC-AUC: {fold_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    print(f"\nMean ROC-AUC (5-Fold): {mean_auc:.4f} (+/- {np.std(fold_aucs):.4f})")

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.3, 0.7, 41):
        preds = (oof_proba >= thresh).astype(int)
        rep = classification_report(y, preds, output_dict=True)
        f1_macro = rep["macro avg"]["f1-score"]
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_thresh = thresh

    print(f"Optimized Threshold: {best_thresh:.3f}\n")
    print(classification_report(y, (oof_proba >= best_thresh).astype(int), target_names=["failure (0)", "success (1)"], digits=4))

    importance_series = pd.Series(importances_accumulator / 5.0, index=feature_cols)
    normalized_importance = importance_series / importance_series.sum()
    normalized_importance = normalized_importance.sort_values(ascending=False)

    print("Top Feature Importances (Normalized Gain)")
    print("-" * 52)
    for feat, score in normalized_importance.items():
        print(f"  {feat:<35s}  {score:.6f}")

if __name__ == "__main__":
    run_prediction_pipeline()
