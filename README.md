# tbf-eval
Trajectory Behavioral Fingerprinting for AI Agent Evaluation  
*Amritesh Banerjee, Pranil Raichura*

## Overview

tbf-eval is a framework for evaluating autonomous AI agents on SWE-bench by mapping sequential step trajectories to compact SHAP attribution vectors. It introduces the **Behavioral Consistency Metric (BCM)** — a cosine-similarity-based score measuring intra-agent fingerprint stability — and applies it across three experiments: bootstrapped inconsistency analysis, behavioral gaming detection, and difficulty-stratified drift.

## Target Venue

- **Conference:** Conference on Language Modeling (COLM 2026)
- **Track:** AI Measurement Science Workshop (AIMS)
- **Submission Deadline:** June 23, 2026

---

## Repository Structure

```
tbf-eval/
├── data/
│   ├── load_swebench.py            # Downloads and merges both HuggingFace datasets
│   ├── extract_features.py         # Extracts 12 behavioral features per trajectory
│   ├── parser.py                   # Trajectory JSON parsing utilities
│   ├── inspect_schema.py           # Schema inspection / sanity checks
│   ├── verify_deserialization.py   # Validates trajectory deserialization
│   └── confound.py                 # Confounder audit (shared-task overlap, BCM by agent)
├── model/
│   └── behavioral_predictor_pipeline.py   # 5-fold LightGBM + SHAP extraction
├── models/
│   └── umap_projection.py          # UMAP 2D projection of SHAP fingerprints
├── metrics/
│   ├── bcm.py                      # Behavioral Consistency Metric computation
│   └── clustering.py               # K-Means clustering over SHAP fingerprints
├── experiments/
│   ├── exp1_inconsistency.py       # Exp 1: BCM bootstrap + statistical correlation
│   ├── exp2_gaming.py              # Exp 2: Behavioral gaming / failure diagnostics
│   └── exp3_drift.py               # Exp 3: Difficulty-stratified behavioral drift
├── figures/
│   └── plot_all.py                 # Feature histograms and distribution plots
└── results/
    ├── figures/                    # Generated PNG plots (exp1–3, feature histograms)
    ├── files/                      # Intermediate CSVs and NumPy arrays
    └── *.txt                       # Captured stdout for each pipeline stage
```

---

## Installation

**Runtime:** Python 3.10 (tested on Google Colab and local environments)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/tbf-eval.git
cd tbf-eval
```

### 2. Install dependencies

```bash
pip install datasets pandas numpy scikit-learn lightgbm shap umap-learn scipy matplotlib seaborn
```

Or install from a requirements file (if provided):

```bash
pip install -r requirements.txt
```

### 3. Create the required data directory

```bash
mkdir -p tbf/data tbf/models
```

---

## Downloading the Datasets

The pipeline uses two publicly available trajectory datasets from HuggingFace. Both are downloaded automatically by `data/load_swebench.py`. No manual download is required, but a HuggingFace account and `huggingface_hub` authentication may be needed for gated datasets.

### Dataset 1 — SWE-smith-trajectories

- **Source:** [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
- **Split used:** `tool` (falls back to `train` if unavailable)
- **Agent:** `swe_smith_claude_3.7`
- **Sample size:** up to 4,000 trajectories (random seed 42)

### Dataset 2 — nebius/SWE-agent-trajectories

- **Source:** [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- **Split used:** `train`
- **Agents selected:** up to 4 model systems with ≥ 500 trajectories each
- **Sample size:** up to 2,000 trajectories per agent (random seed 42)

Both datasets are merged and shuffled into a single file: `tbf/data/raw_behavioral_dataframe.csv`

To authenticate with HuggingFace (if required):

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Reproducing All Results

Run each script in order. Every script reads from files written by the previous stage.

---

### Stage 0 — Verify Trajectory Deserialization

```bash
python data/verify_deserialization.py
```

Validates that trajectory JSON can be parsed correctly. Prints sample message fields.  
**Expected output:** `results/verify_deserialization_output.txt`

---

### Stage 1 — Download and Merge Datasets

```bash
python data/load_swebench.py
```

Downloads both HuggingFace datasets, merges them, and writes the combined behavioral dataframe.  
**Output:** `tbf/data/raw_behavioral_dataframe.csv`  
**Expected console output:**
```
Systems meeting minimum volume threshold (>= 500): ['swe-agent-llama-405b', 'swe-agent-llama-70b', 'swe-agent-llama-8b']
Final Multi-Agent System Count Profile:
  claude-3-7-sonnet-20250219    2938
  swe-agent-llama-70b           2000
  swe-agent-llama-8b            2000
  swe-agent-llama-405b          1191
  claude-3-5-sonnet-20241022     946
  gpt-4o-2024-08-06              116
```

---

### Stage 2 — Inspect Schema

```bash
python data/inspect_schema.py
```

Prints field names and a sample record from the raw dataframe for validation.  
**Expected output:** `results/inspect_schema_output.txt`

---

### Stage 3 — Extract Behavioral Features

```bash
python data/extract_features.py
```

Extracts 12 behavioral features from each trajectory's action sequence:

| Feature | Description |
|---|---|
| `total_steps` | Total number of messages in trajectory |
| `mean_action_length` | Mean character length of actions |
| `max_action_length` | Max character length of any action |
| `file_search_count` | Regex-matched file search actions |
| `file_view_count` | Regex-matched file view/read actions |
| `file_edit_count` | Regex-matched file edit/write actions |
| `test_execution_count` | Regex-matched test run actions |
| `action_entropy` | Shannon entropy over action categories |
| `consecutive_repetition_max` | Max run of consecutive identical action types |
| `unique_action_ratio` | Unique actions / total actions |
| `error_flag_count` | Count of error-pattern matches |
| `step_velocity` | Steps per unit of sequence length |

**Output:** `tbf/data/engineered_features_matrix.csv`

---

### Stage 4 — Train Predictor and Extract SHAP Fingerprints

```bash
python model/behavioral_predictor_pipeline.py
```

Trains a LightGBM classifier with 5-fold stratified cross-validation and extracts out-of-fold SHAP values as behavioral fingerprints.

**Expected console output:**
```
Fold 1 processing completed.
...
Fold 5 processing completed.

============================================================
GLOBAL PERFORMANCE RESULTS (PREDICTOR)
============================================================
Mean ROC-AUC (5-Fold): 0.6900
Optimized Threshold  : 0.370

              precision    recall  f1-score   support
 failure (0)     0.8260    0.6374    0.7196      6680
 success (1)     0.3999    0.6428    0.4931      2511
    accuracy                         0.6389      9191

TOP FEATURE IMPORTANCES (NORMALIZED GAIN):
  total_steps              0.5382
  mean_action_length       0.1164
  max_action_length        0.0914
  unique_action_ratio      0.0746
  test_execution_count     0.0492
```

**Outputs:**
- `tbf/models/shap_fingerprints.csv`
- `tbf/models/oof_proba.npy`
- `tbf/models/oof_shap_matrix.npy`

---

### Stage 5 — Compute Behavioral Consistency Metric (BCM)

```bash
python metrics/bcm.py
```

Computes per-agent, per-outcome BCM scores using pairwise cosine similarity over SHAP fingerprint matrices.

**Output:** `tbf/models/bcm_results.csv`  
**Expected output:** `results/bcm_output.txt`

---

### Stage 6 — Cluster SHAP Fingerprints

```bash
python metrics/clustering.py
```

Runs K-Means (k=2..8) and fits the final model at k=3. Reports within-cluster BCM, success rate, and agent composition per cluster.

**Outputs:**
- `tbf/models/clustered_fingerprints.csv`
- `tbf/models/raw_cluster_profiles.csv`

**Expected output:** `results/clustering_output.txt`

---

### Stage 7 — UMAP Projection

```bash
python models/umap_projection.py
```

Reduces SHAP fingerprints to 2D with UMAP (n_neighbors=15, min_dist=0.1, euclidean metric, seed=42) and saves the projection.

**Output:** `tbf/models/umap_2d_projection.npy`

---

### Stage 8 — Confounder Audit

```bash
python data/confound.py
```

Runs a pairwise shared-task overlap audit across agents and computes BCM stratified by task difficulty. Checks whether BCM differences can be explained by dataset overlap.

**Expected output:** `results/confound_output.txt`

---

### Stage 9 — Experiment 1: BCM Bootstrap and Inconsistency Analysis

```bash
python experiments/exp1_inconsistency.py
```

Bootstraps BCM 1,000 times per agent and reports 95% confidence intervals. Computes Pearson and Spearman correlations between BCM and task success rate.

**Expected console output:**
```
Agent: claude-3-7-sonnet-20250219   | N: 2938 | Mean: 0.7652 | 95% CI: [0.7444, 0.7837]
Agent: swe-agent-llama-70b          | N: 2000 | Mean: 0.0655 | 95% CI: [0.0587, 0.0730]
Agent: claude-3-5-sonnet-20241022   | N: 946  | Mean: 0.8335 | 95% CI: [0.8003, 0.8640]
Agent: swe-agent-llama-8b           | N: 2000 | Mean: 0.0866 | 95% CI: [0.0733, 0.1025]
Agent: swe-agent-llama-405b         | N: 1191 | Mean: 0.0711 | 95% CI: [0.0555, 0.0897]
Agent: gpt-4o-2024-08-06            | N: 116  | Mean: 0.8162 | 95% CI: [0.7127, 0.9062]

Success Rate vs. BCM     | Pearson r: 0.7841 | Spearman rho: 0.6000
Success Variance vs. BCM | Pearson r: 0.5907
```

**Outputs:**
- `tbf/models/agent_bcm_bootstrap_summary.csv`
- `tbf/models/agent_statistical_summary.csv`
- `results/figures/exp1.png`

**Expected output log:** `results/exp1_output.txt`

---

### Stage 10 — Experiment 2: Behavioral Gaming / Failure Diagnostics

```bash
python experiments/exp2_gaming.py
```

Flags trajectories using a two-prong heuristic:
- **Prong A:** `step_velocity > 0.8` AND `file_edit_count < 0.05`
- **Prong B:** `consecutive_repetition_max > 3` AND `error_flag_count > 0.4`

Reports flagged trajectory counts by cluster and success rate.

**Expected console output:**
```
EXPERIMENT 2: UNIFIED BEHAVIORAL FAILURE DIAGNOSTICS
Total Trajectories Flagged via Minimal-Edit Heuristic: 174
Cluster 0: Fraction = 0.7759 (135 runs)
  -> Resolved Successes: 3 (Success Rate: 0.0222)
Cluster 1: Fraction = 0.2069 (36 runs)
  -> Resolved Successes: 0 (Success Rate: 0.0000)
Cluster 2: Fraction = 0.0172 (3 runs)
  -> Resolved Successes: 0 (Success Rate: 0.0000)
```

**Output:** `results/figures/exp2.png`  
**Expected output log:** `results/exp2_output.txt`

---

### Stage 11 — Experiment 3: Difficulty-Stratified Behavioral Drift

```bash
python experiments/exp3_drift.py
```

Bins tasks into easy / medium / hard terciles by per-instance resolution rate, then reports BCM and success rate per agent per bin.

**Expected console output (excerpt):**
```
EXPERIMENT 3: BEHAVIORAL DRIFT AND SUCCESS RATE BY DIFFICULTY BIN
Bin Thresholds: Easy (>0.3333), Medium (0.0 to 0.3333), Hard (<=0.0)

Agent: claude-3-5-sonnet-20241022
  -> EASY   | Trajectories: 452  | BCM: 0.8708 | Success Rate: 0.8695
  -> MEDIUM | Trajectories: 39   | BCM: 0.7893 | Success Rate: 0.3077
  -> HARD   | Trajectories: 455  | BCM: 0.7996 | Success Rate: 0.0

Agent: claude-3-7-sonnet-20250219
  -> EASY   | Trajectories: 1153 | BCM: 0.8241 | Success Rate: 0.967
  -> MEDIUM | Trajectories: 33   | BCM: 0.618  | Success Rate: 0.3636
  -> HARD   | Trajectories: 1752 | BCM: 0.7327 | Success Rate: 0.0
```

**Output:** `results/figures/exp3.png`  
**Expected output log:** `results/exp3_output.txt`

---

### Stage 12 — Generate Feature Distribution Plots

```bash
python figures/plot_all.py
```

Generates class balance analysis and per-feature histograms stratified by outcome (success/failure).

**Output:** `results/figures/feature_histograms.png`  
**Expected output log:** `results/plot_all_output.txt`

---

## Full Reproduction — Command Summary

```bash
# 0. Setup
git clone https://github.com/<your-username>/tbf-eval.git && cd tbf-eval
pip install datasets pandas numpy scikit-learn lightgbm shap umap-learn scipy matplotlib seaborn
mkdir -p tbf/data tbf/models

# 1. Data pipeline
python data/verify_deserialization.py
python data/load_swebench.py
python data/inspect_schema.py
python data/extract_features.py

# 2. Model and fingerprints
python model/behavioral_predictor_pipeline.py

# 3. Metrics and structure
python metrics/bcm.py
python metrics/clustering.py
python models/umap_projection.py

# 4. Confounder audit
python data/confound.py

# 5. Experiments
python experiments/exp1_inconsistency.py
python experiments/exp2_gaming.py
python experiments/exp3_drift.py

# 6. Figures
python figures/plot_all.py
```

All intermediate files are written to `tbf/data/` and `tbf/models/`. All final outputs (logs, figures, CSVs) are written to `results/`.

---

## Key Results Summary

| Result | Value |
|---|---|
| LightGBM ROC-AUC (5-fold OOF) | 0.6900 |
| Top predictive feature | `total_steps` (gain: 0.538) |
| Claude-3.5 BCM | 0.8335 [0.8003, 0.8640] |
| Claude-3.7 BCM | 0.7652 [0.7444, 0.7837] |
| GPT-4o BCM | 0.8162 [0.7127, 0.9062] |
| SWE-agent LLaMA BCM (all sizes) | 0.065–0.087 |
| BCM–Success Pearson r | 0.7841 |
| Gaming-flagged trajectories | 174 / 9191 (1.9%) |
| Gaming cluster success rate | ≤ 2.2% across all clusters |

---

## Notes

- All random operations use `random_state=42` for reproducibility.
- SHAP values are extracted out-of-fold; no test-set leakage occurs.
- BCM is computed on SHAP fingerprint vectors, not raw features.
- Difficulty binning in Experiment 3 uses per-instance resolution rate quantiles (q33, q66) computed from the merged dataset.
