"""Benchmark figure (python plot_results.py WORK_DIR/results_per_fold.csv OUT.png): per-fold scores (small points) and their mean (large point) for each method, grouped by input modalities."""
import sys
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(sys.argv[1])
groups = {  # method -> (label, modalities)
    "rna_pca": ("RNA PCA", "RNA"), "scvi": ("scVI", "RNA"),
    "prot_markers": ("protein markers", "protein"),
    "concat": ("concatenated PCA + markers", "RNA + protein"), "wnn": ("WNN (muon)", "RNA + protein"),
    "mofa": ("MOFA+", "RNA + protein"), "totalvi": ("totalVI", "RNA + protein"),
    "image": ("image features", "image"), "prot_image": ("markers + image features", "protein + image"),
    "rna_prot_image": ("RNA PCA + markers + image features", "RNA + protein + image"),
}
df = df[df.method.isin(groups)]
order = [m for m in groups if m in set(df.method)]
colors = {"RNA": "#4C72B0", "protein": "#DD8452", "RNA + protein": "#55A868", "image": "#8172B3",
          "protein + image": "#937860", "RNA + protein + image": "#C44E52"}
metrics = {"protein_r": "held-out proteins\n(mean Pearson r)", "gene_r": "held-out genes\n(mean Pearson r)",
           "region_bal_acc": "pathology region\n(balanced accuracy)"}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (metric, title) in zip(axs, metrics.items(), strict=True):
    for i, m in enumerate(order):
        v = df.loc[df.method == m, metric]
        ax.scatter(v, [i] * len(v), color=colors[groups[m][1]], s=12, alpha=0.6)
        ax.scatter(v.mean(), i, color=colors[groups[m][1]], s=70, edgecolor="k", zorder=3)
    ax.grid(axis="x", alpha=0.3)
    lo = df[metric].min()
    ax.set_xlim(lo - 0.1 * (df[metric].max() - lo) - 0.02, df[metric].max() + 0.02)
    ax.set_title(title, fontsize=10)
axs[0].set_yticks(range(len(order)), [groups[m][0] for m in order]); axs[0].invert_yaxis()
handles = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", ms=8) for c in colors.values()]
fig.legend(handles, colors, title="input", loc="lower center", ncol=len(colors), frameon=False, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
