import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_distribution_plots():
    matrix_path = 'tbf/data/engineered_features_matrix.csv'
    if not os.path.exists(matrix_path):
        print(f"Error: Matrix file not found at {matrix_path}. Run extract_features.py first.")
        return

    df = pd.read_csv(matrix_path)

    print("=== 1. CLASS BALANCE ANALYSIS ===")
    counts = df['resolved'].value_counts()
    percentages = df['resolved'].value_counts(normalize=True) * 100

    for label in counts.index:
        status = "Success (1)" if label == 1 else "Failure (0)"
        print(f"{status}: {counts[label]:,} instances ({percentages[label]:.2f}%)")

    print("\n=== 2. GENERATING FEATURE HISTOGRAMS ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_names = [col for col in numeric_cols if col != 'resolved']

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 22))
    axes = axes.flatten()

    for idx, col in enumerate(feature_names):
        ax = axes[idx]

        unique_vals = df[col].dropna().unique()
        should_use_kde = len(unique_vals) > 1 and np.std(df[col].dropna()) > 1e-6

        sns.histplot(
            data=df,
            x=col,
            hue='resolved',
            multiple='layer',
            kde=should_use_kde,
            bins=30,
            palette={1: '#2ecc71', 0: '#e74c3c'},
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count')

    for j in range(len(feature_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    figures_dir = 'tbf/figures'
    os.makedirs(figures_dir, exist_ok=True)
    output_plot_path = os.path.join(figures_dir, 'feature_histograms.png')
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"\nVisualizations compiled successfully and saved to: {output_plot_path}")

if __name__ == "__main__":
    generate_distribution_plots()
