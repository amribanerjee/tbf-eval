import os
import matplotlib.pyplot as plt
import numpy as np

def generate_global_vs_within_chart():
    agents = [
        "SWE-Agent\nLlama-70B",
        "SWE-Agent\nLlama-405B",
        "SWE-Agent\nLlama-8B",
        "Claude 3.7\nSonnet",
        "GPT-4o",
        "Claude 3.5\nSonnet"
    ]
    global_bcm = [0.0649, 0.0709, 0.0862, 0.7657, 0.8143, 0.8327]
    within_task_bcm = [0.3050, 0.3481, 0.3188, 0.5642, 0.8972, 0.8065]
    
    x = np.arange(len(agents))
    width = 0.35
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.bar(x - width/2, global_bcm, width, label="Global BCM", color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, within_task_bcm, width, label="Within-Task BCM ($N \\geq 3$)", color="#aec7e8", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Behavioral Consistency Metric (BCM)", fontsize=10)
    ax.set_title("Behavioral Consistency Metric Comparison: Global vs. Within-Task", fontsize=11, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "global_vs_within_bcm.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    generate_global_vs_within_chart()
