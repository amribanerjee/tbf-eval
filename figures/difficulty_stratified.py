import os
import matplotlib.pyplot as plt
import numpy as np

def generate_difficulty_stratified_chart():
    agents = [
        "SWE-Agent\nLlama-8B",
        "SWE-Agent\nLlama-70B",
        "SWE-Agent\nLlama-405B",
        "Claude 3.7\nSonnet",
        "GPT-4o",
        "Claude 3.5\nSonnet"
    ]
    easy_bcm = [0.0451, 0.2128, 0.0854, 0.8241, 0.7778, 0.8708]
    medium_bcm = [0.1080, 0.1297, 0.1073, 0.6180, 0.9312, 0.7893]
    hard_bcm = [0.1770, 0.0663, 0.2930, 0.7327, 0.8210, 0.7996]
    
    x = np.arange(len(agents))
    width = 0.25
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    ax.bar(x - width, easy_bcm, width, label="Easy Tasks", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.bar(x, medium_bcm, width, label="Medium Tasks", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, hard_bcm, width, label="Hard Tasks", color="#d62728", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Behavioral Consistency Metric (BCM)", fontsize=10)
    ax.set_title("Difficulty-Stratified Behavioral Consistency Metric (BCM) Profiles", fontsize=11, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "difficulty_stratified_bcm.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    generate_difficulty_stratified_chart()
