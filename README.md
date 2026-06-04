# tbf-eval

Trajectory Behavioral Fingerprinting for AI Agent Evaluation

*Amritesh Banerjee, Pranil Raichura*

## Overview

tbf-eval is a framework for evaluating autonomous AI agents on SWE-bench by mapping sequential step trajectories to compact SHAP attribution vectors.

## Target Venue

- **Conference:** Conference on Language Modeling (COLM 2026)
- **Track:** AI Measurement Science Workshop (AIMS)
- **Submission Deadline:** June 23, 2026

## Repository Structure

- `tbf/data/` — Raw SWE-bench/SWE-agent trajectory JSON files
- `tbf/model/` — Trained XGBoost predictive models and SHAP explainer objects
- `tbf/metrics/` — Behavioral consistency metric and clustering algorithm scripts
- `tbf/experiments/` — Evaluation code across agent systems and difficulty tiers
- `tbf/figures/` — Generated plots, UMAP clusters, and SHAP distribution charts

## Setup

- **Runtime:** Google Colab (Python 3.10)
- **Dependencies:** `xgboost`, `shap`, `scikit-learn`, `pandas`, `umap-learn`, `scipy`

## Getting Started

1. Open the project notebook in Google Colab.
2. Run the environment setup cell to install dependencies.
3. Place raw trajectory logs in the `tbf/data/` directory using the left-hand file panel.
