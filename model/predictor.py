import json
import math
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def extract_trajectory_segments(raw_json: str) -> dict:
    steps = _safe_parse_trajectory(raw_json)
    n_steps = len(steps)
    
    if n_steps == 0:
        return {
            "full_text": "",
            "terminal_step_text": "",
            "meta_sequence_depth": 0.0,
            "meta_step_imbalance": 0.0
        }
    
    full_text = " ".join(steps)
    terminal_step_text = steps[-1]
    
    step_lengths = [len(s.split()) for s in steps]
    mean_len = np.mean(step_lengths) if step_lengths else 1.0
    max_len = np.max(step_lengths) if step_lengths else 1.0
    
    return {
        "full_text": full_text,
        "terminal_step_text": terminal_step_text,
        "meta_sequence_depth": float(math.log1p(n_steps)),
        "meta_step_imbalance": float(max_len / mean_len if mean_len > 0 else 0.0)
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
    spw = neg_support / pos_support

    print(f"Trajectory column detected  : {trajectory_col}")
    print(f"Label column detected       : {label_col}")
    print(f"Dataset shape               : {df.shape}")
    print()
    print(f"Class distribution  -> 0 (failure): {neg_support}  |  1 (success): {pos_support}")
    print(f"scale_pos_weight    -> {spw:.4f}")
    print()

    print("Extracting geometric properties and isolating terminal execution states ...")
    records = []
    for raw in df[trajectory_col]:
        records.append(extract_trajectory_segments(raw))
    df_segments = pd.DataFrame(records)
    
    print("Fitting global text vector space (broad-spectrum n-grams) ...")
    tfidf_full = TfidfVectorizer(max_features=60000, ngram_range=(1, 4), stop_words="english", sublinear_tf=True)
    X_tfidf_full = tfidf_full.fit_transform(df_segments["full_text"].astype(str))
    
    print("Fitting terminal step text vector space (dense char n-grams) ...")
    tfidf_term = TfidfVectorizer(max_features=25000, analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    X_tfidf_term = tfidf_term.fit_transform(df_segments["terminal_step_text"].astype(str))
    
    print("Extracting selective Chi-Square n-grams across full trajectory spectrum ...")
    chi2_selector = SelectKBest(chi2, k=128)
    X_chi2_sparse = chi2_selector.fit_transform(X_tfidf_full, y)
    chi2_indices = chi2_selector.get_support(indices=True)
    full_feature_names = tfidf_full.get_feature_names_out()
    chi2_cols = [f"high_signal_ngram_{full_feature_names[idx].replace(' ', '_')}" for idx in chi2_indices]
    df_chi2 = pd.DataFrame(X_chi2_sparse.toarray(), columns=chi2_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Generating out-of-fold stacked predictions for multi-tier text modeling layers ...")
    oof_full_proba = np.zeros(len(y))
    oof_term_proba = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X_tfidf_full, y):
        text_clf_full = LogisticRegression(C=0.55, class_weight="balanced", random_state=42, max_iter=300)
        text_clf_full.fit(X_tfidf_full[train_idx], y[train_idx])
        oof_full_proba[val_idx] = text_clf_full.predict_proba(X_tfidf_full[val_idx])[:, 1]
        
        text_clf_term = LogisticRegression(C=0.45, class_weight="balanced", random_state=42, max_iter=300)
        text_clf_term.fit(X_tfidf_term[train_idx], y[train_idx])
        oof_term_proba[val_idx] = text_clf_term.predict_proba(X_tfidf_term[val_idx])[:, 1]
        
    df_meta_final = df_segments.drop(columns=["full_text", "terminal_step_text"])
    df_meta_final["global_text_stack_proba"] = oof_full_proba
    df_meta_final["terminal_step_semantic_anchor"] = oof_term_proba
    
    X_final = pd.concat([df_meta_final, df_chi2], axis=1)
    feature_names = X_final.columns.tolist()
    X_arr = X_final.values
    
    print(f"Multi-tier stacked feature matrix shape : {X_arr.shape}")
    print()

    fold_aucs = []
    oof_proba = np.zeros(len(y))

    xgb_params = dict(
        n_estimators=1600,
        learning_rate=0.007,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=12,
        gamma=0.5,
        reg_alpha=1.2,
        reg_lambda=7.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    rf_params = dict(
        n_estimators=700,
        max_depth=14,
        min_samples_split=14,
        min_samples_leaf=7,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    importances_accumulator = np.zeros(len(feature_names))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y), start=1):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf_xgb = xgb.XGBClassifier(**xgb_params)
        clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba_val_xgb = clf_xgb.predict_proba(X_val)[:, 1]
        importances_accumulator += clf_xgb.feature_importances_

        clf_rf = RandomForestClassifier(**rf_params)
        clf_rf.fit(X_train, y_train)
        proba_val_rf = clf_rf.predict_proba(X_val)[:, 1]

        p_xgb_cal = np.power(proba_val_xgb, 1.05)
        p_rf_cal = np.power(proba_val_rf, 0.95)
        
        proba_val = (0.58 * p_xgb_cal) + (0.42 * p_rf_cal)
        oof_proba[val_idx] = proba_val
        
        fold_auc = roc_auc_score(y_val, proba_val)
        fold_aucs.append(fold_auc)

        print(f"  Fold {fold_idx}  |  Validation ROC-AUC: {fold_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    print()
    print(f"Mean ROC-AUC (5-Fold)       : {mean_auc:.4f}  (+/- {std_auc:.4f})")
    print()

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.3, 0.7, 41):
        preds = (oof_proba >= thresh).astype(int)
        rep = classification_report(y, preds, output_dict=True)
        f1_macro = rep["macro avg"]["f1-score"]
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_thresh = thresh

    oof_preds = (oof_proba >= best_thresh).astype(int)
    print(f"Optimized Decision Threshold: {best_thresh:.3f}")
    print()
    print("Out-of-Fold Classification Report")
    print("-" * 52)
    print(
        classification_report(
            y,
            oof_preds,
            target_names=["failure (0)", "success (1)"],
            digits=4,
        )
    )

    avg_importances = importances_accumulator / 5.0
    importance_series = pd.Series(avg_importances, index=feature_names).sort_values(
        ascending=False
    )

    print("Top Base XGBoost Feature Importances (descending)")
    print("-" * 52)
    for feat, score in importance_series.head(20).items():
        print(f"  {feat:<40s}  {score:.6f}")
    print()

    if mean_auc < 0.65:
        print("WARNING: Mean ROC-AUC is below 0.65.")
    elif mean_auc > 0.95:
        print("WARNING: Mean ROC-AUC exceeds 0.95. Check for data leakage.")
    else:
        print(f"SUCCESS: Model performance is optimal (Mean AUC = {mean_auc:.4f}).")

if __name__ == "__main__":
    run_pipeline()
