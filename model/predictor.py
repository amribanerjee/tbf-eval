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
from sklearn.isotonic import IsotonicRegression
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
            "terminal_step_text": "",
            "meta_sequence_depth": 0.0,
            "meta_step_imbalance": 0.0,
            "trajectory_velocity": 0.0,
            "terminal_density_ratio": 0.0
        }

    full_text = " ".join(steps)
    terminal_step_text = steps[-1]

    step_lengths = [len(s.split()) for s in steps]
    mean_len = np.mean(step_lengths) if step_lengths else 1.0
    max_len = np.max(step_lengths) if step_lengths else 1.0

    velocity = 0.0
    if n_steps > 1:
        velocity = float(np.mean(np.diff(step_lengths)))

    density_ratio = float(step_lengths[-1] / mean_len if mean_len > 0 else 0.0)

    return {
        "full_text": full_text,
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
    spw = neg_support / pos_support

    print(f"Trajectory column detected  : {trajectory_col}")
    print(f"Label column detected       : {label_col}")
    print(f"Dataset shape               : {df.shape}")
    print()

    print("Extracting multi-domain trajectory metrics...")
    records = [extract_trajectory_features(raw) for raw in df[trajectory_col]]
    df_features = pd.DataFrame(records)

    print("Fitting global text vector space (broad-spectrum n-grams) ...")
    tfidf_full = TfidfVectorizer(max_features=65000, ngram_range=(1, 4), stop_words="english", sublinear_tf=True)
    X_tfidf_full = tfidf_full.fit_transform(df_features["full_text"].astype(str))

    print("Fitting terminal step text vector space (dense char n-grams) ...")
    tfidf_term = TfidfVectorizer(max_features=25000, analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    X_tfidf_term = tfidf_term.fit_transform(df_features["terminal_step_text"].astype(str))

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
        text_clf_full = LogisticRegression(C=0.60, class_weight="balanced", random_state=42, max_iter=300)
        text_clf_full.fit(X_tfidf_full[train_idx], y[train_idx])
        oof_full_proba[val_idx] = text_clf_full.predict_proba(X_tfidf_full[val_idx])[:, 1]

        text_clf_term = LogisticRegression(C=0.50, class_weight="balanced", random_state=42, max_iter=300)
        text_clf_term.fit(X_tfidf_term[train_idx], y[train_idx])
        oof_term_proba[val_idx] = text_clf_term.predict_proba(X_tfidf_term[val_idx])[:, 1]

    df_text_meta = pd.DataFrame({
        "global_text_stack_proba": oof_full_proba,
        "terminal_step_semantic_anchor": oof_term_proba
    })

    X_text_domain = pd.concat([df_text_meta, df_chi2], axis=1)
    text_feature_names = X_text_domain.columns.tolist()
    X_text_arr = X_text_domain.values

    X_structural_arr = df_features[["meta_sequence_depth", "meta_step_imbalance", "trajectory_velocity", "terminal_density_ratio"]].values

    fold_aucs = []
    oof_proba = np.zeros(len(y))
    importances_accumulator = np.zeros(len(text_feature_names))

    xgb_text_params = dict(
        n_estimators=2200,
        learning_rate=0.005,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=12,
        gamma=0.6,
        reg_alpha=1.5,
        reg_lambda=8.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    rf_text_params = dict(
        n_estimators=800,
        max_depth=14,
        min_samples_split=14,
        min_samples_leaf=7,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    xgb_struct_params = dict(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    print("Executing parallel track domain training and meta-blending...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_text_arr, y), start=1):
        X_text_train, X_text_val = X_text_arr[train_idx], X_text_arr[val_idx]
        X_struct_train, X_struct_val = X_structural_arr[train_idx], X_structural_arr[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf_xgb_text = xgb.XGBClassifier(**xgb_text_params)
        clf_xgb_text.fit(X_text_train, y_train, eval_set=[(X_text_val, y_val)], verbose=False)
        p_val_xgb_text = clf_xgb_text.predict_proba(X_text_val)[:, 1]
        importances_accumulator += clf_xgb_text.feature_importances_

        clf_rf_text = RandomForestClassifier(**rf_text_params)
        clf_rf_text.fit(X_text_train, y_train)
        p_val_rf_text = clf_rf_text.predict_proba(X_text_val)[:, 1]

        clf_xgb_struct = xgb.XGBClassifier(**xgb_struct_params)
        clf_xgb_struct.fit(X_struct_train, y_train, eval_set=[(X_struct_val, y_val)], verbose=False)
        p_val_xgb_struct = clf_xgb_struct.predict_proba(X_struct_val)[:, 1]

        p_xgb_text_cal = np.power(p_val_xgb_text, 1.05)
        p_rf_text_cal = np.power(p_val_rf_text, 0.95)
        p_text_blend = (0.58 * p_xgb_text_cal) + (0.42 * p_rf_text_cal)

        p_val_blend = (0.78 * p_text_blend) + (0.22 * p_val_xgb_struct)

        p_tr_xgb_text = clf_xgb_text.predict_proba(X_text_train)[:, 1]
        p_tr_rf_text = clf_rf_text.predict_proba(X_text_train)[:, 1]
        p_tr_struct = clf_xgb_struct.predict_proba(X_struct_train)[:, 1]

        p_tr_text_blend = (0.58 * np.power(p_tr_xgb_text, 1.05)) + (0.42 * np.power(p_tr_rf_text, 0.95))
        p_tr_blend = (0.78 * p_tr_text_blend) + (0.22 * p_tr_struct)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr_blend, y_train)
        proba_val = iso.predict(p_val_blend)

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
    importance_series = pd.Series(avg_importances, index=text_feature_names).sort_values(
        ascending=False
    )

    print("Top Core Text-Track Feature Importances (descending)")
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
