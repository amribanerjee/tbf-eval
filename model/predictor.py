import json
import math
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def _safe_parse_trajectory(raw_json: str) -> list:
    if not isinstance(raw_json, str) or not raw_json.strip():
        return []
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        stripped = raw_json.strip().strip("[]")
        if stripped:
            return [stripped]
        return []

def extract_trajectory_features(raw_json: str) -> dict:
    steps = _safe_parse_trajectory(raw_json)
    n_steps = len(steps)
    
    if n_steps == 0:
        return {
            "full_text": "",
            "mid_trajectory_text": "",
            "terminal_step_text": "",
            "meta_sequence_depth": 0.0,
            "meta_step_imbalance": 0.0,
            "trajectory_velocity": 0.0,
            "terminal_density_ratio": 0.0
        }
    
    full_text = " ".join(steps)
    terminal_step_text = steps[-1]
    mid_trajectory_text = " ".join(steps[:-1]) if n_steps > 1 else steps[0]
    
    step_lengths = [len(s.split()) for s in steps]
    mean_len = np.mean(step_lengths) if step_lengths else 1.0
    max_len = np.max(step_lengths) if step_lengths else 1.0
    
    velocity = 0.0
    if n_steps > 1:
        velocity = float(np.mean(np.diff(step_lengths)))
        
    density_ratio = float(step_lengths[-1] / mean_len if mean_len > 0 else 0.0)
    
    return {
        "full_text": full_text,
        "mid_trajectory_text": mid_trajectory_text,
        "terminal_step_text": terminal_step_text,
        "meta_sequence_depth": float(math.log1p(n_steps)),
        "meta_step_imbalance": float(max_len / mean_len if mean_len > 0 else 0.0),
        "trajectory_velocity": velocity,
        "terminal_density_ratio": density_ratio
    }

def run_pipeline():
    data_path = "tbf/data/raw_behavioral_dataframe.csv"
    df = pd.read_csv(data_path)

    trajectory_col = None
    label_col = None

    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith("[").any():
            trajectory_col = col
            break

    if trajectory_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                trajectory_col = col
                break

    for col in df.columns:
        if col.lower() in ("resolved", "label", "target", "success", "outcome"):
            label_col = col
            break

    y = df[label_col].astype(int).values
    neg_support = int((y == 0).sum())
    pos_support = int((y == 1).sum())
    scale_weight = neg_support / pos_support

    print(f"Dataset shape: {df.shape}")
    print("Extracting multi-domain features and building sublinear multi-tier text spaces...")
    
    records = [extract_trajectory_features(raw) for raw in df[trajectory_col]]
    df_features = pd.DataFrame(records)
    
    tfidf_full = TfidfVectorizer(max_features=45000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True, min_df=2)
    X_tfidf_full = tfidf_full.fit_transform(df_features["full_text"].astype(str))
    
    tfidf_mid = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=2)
    X_tfidf_mid = tfidf_mid.fit_transform(df_features["mid_trajectory_text"].astype(str))
    
    tfidf_term = TfidfVectorizer(max_features=25000, analyzer="char", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    X_tfidf_term = tfidf_term.fit_transform(df_features["terminal_step_text"].astype(str))
    
    chi2_selector = SelectKBest(chi2, k=120)
    X_chi2_sparse = chi2_selector.fit_transform(X_tfidf_full, y)
    chi2_indices = chi2_selector.get_support(indices=True)
    full_feature_names = tfidf_full.get_feature_names_out()
    chi2_cols = [f"high_signal_ngram_{full_feature_names[idx].replace(' ', '_')}" for idx in chi2_indices]
    df_chi2 = pd.DataFrame(X_chi2_sparse.toarray(), columns=chi2_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_full_proba = np.zeros(len(y))
    oof_mid_proba = np.zeros(len(y))
    oof_term_proba = np.zeros(len(y))
    
    print("Generating non-linear tiered neural meta-features...")
    for train_idx, val_idx in skf.split(X_tfidf_full, y):
        mlp_full = MLPClassifier(hidden_layer_sizes=(64, 16), activation="relu", early_stopping=True, alpha=0.03, random_state=42, max_iter=25)
        mlp_full.fit(X_tfidf_full[train_idx], y[train_idx])
        oof_full_proba[val_idx] = mlp_full.predict_proba(X_tfidf_full[val_idx])[:, 1]
        
        mlp_mid = MLPClassifier(hidden_layer_sizes=(32, 8), activation="relu", early_stopping=True, alpha=0.03, random_state=42, max_iter=25)
        mlp_mid.fit(X_tfidf_mid[train_idx], y[train_idx])
        oof_mid_proba[val_idx] = mlp_mid.predict_proba(X_tfidf_mid[val_idx])[:, 1]
        
        mlp_term = MLPClassifier(hidden_layer_sizes=(32, 8), activation="relu", early_stopping=True, alpha=0.03, random_state=42, max_iter=25)
        mlp_term.fit(X_tfidf_term[train_idx], y[train_idx])
        oof_term_proba[val_idx] = mlp_term.predict_proba(X_tfidf_term[val_idx])[:, 1]
        
    X_final_df = pd.DataFrame({
        "meta_sequence_depth": df_features["meta_sequence_depth"],
        "meta_step_imbalance": df_features["meta_step_imbalance"],
        "trajectory_velocity": df_features["trajectory_velocity"],
        "terminal_density_ratio": df_features["terminal_density_ratio"],
        "neural_full_text_proba": oof_full_proba,
        "neural_mid_text_proba": oof_mid_proba,
        "neural_terminal_text_proba": oof_term_proba,
        "meta_text_interaction": oof_full_proba * oof_term_proba
    })
    
    X_final_df = pd.concat([X_final_df, df_chi2], axis=1)
    feature_names = X_final_df.columns.tolist()
    X_arr = X_final_df.values
    
    fold_aucs = []
    oof_proba = np.zeros(len(y))
    importances_accumulator = np.zeros(len(feature_names))

    lgb_params = {
        "n_estimators": 3000,
        "learning_rate": 0.004,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.75,
        "colsample_bytree": 0.50,
        "min_child_samples": 20,
        "reg_alpha": 4.0,
        "reg_lambda": 15.0,
        "scale_pos_weight": scale_weight,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print("Running lightgbm integrated feature space booster...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y), start=1):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        clf_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
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
    
    importance_series = pd.Series(importances_accumulator / 5.0, index=feature_names)
    normalized_importance = importance_series / importance_series.sum()
    normalized_importance = normalized_importance.sort_values(ascending=False)
    
    print("Top Feature Importances (Normalized Gain)")
    print("-" * 52)
    for feat, score in normalized_importance.head(15).items():
        print(f"  {feat:<35s}  {score:.6f}")

if __name__ == "__main__":
    run_pipeline()  
