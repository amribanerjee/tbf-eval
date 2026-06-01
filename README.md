# tbf-eval
Trajectory Behavioral Fingerprinting for AI Agent Evaluation

# Trajectory Behavioral Fingerprinting (TBF)

A web-based framework for evaluating autonomous AI agents on SWE-bench by mapping sequential step trajectories to compact SHAP attribution vectors.

## 🎯 Target Venue
- **Conference:** Conference on Language Modeling (COLM 2026)
- **Track:** AI Measurement Science Workshop (AIMS)
- **Submission Deadline:** June 23, 2026

## 📁 Workspace Architecture
- `tbf/data/` : Raw SWE-bench/SWE-agent trajectory JSON files.
- `tbf/model/` : Trained XGBoost predictive models and SHAP explainer objects.
- `tbf/metrics/` : Core behavioral math scripts (Behavioral Consistency Metric & clustering algorithms).
- `tbf/experiments/` : Evaluation code running across agent systems and difficulty bounds.
- `tbf/figures/` : Generated plots, UMAP clusters, and SHAP distribution charts.

## 🛠️ Environment Stack
- **Runtime:** Google Colab (Python 3.10)
- **Key Libraries:** `xgboost`, `shap`, `scikit-learn`, `pandas`, `umap-learn`, `scipy`

## 🚀 Getting Started
1. Open the project notebook in Google Colab.
2. Execute the environment setup block to install runtime dependencies.
3. Drop raw trajectory logs into the `tbf/data/` directory via the left-hand files panel.
