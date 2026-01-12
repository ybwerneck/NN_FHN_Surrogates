import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------------------------
# Style
# -------------------------------------------------
sns.set(style="whitegrid")

p = 1
plt.rcParams.update({
    "font.size": p * 22,
    "axes.titlesize": p * 24,
    "axes.labelsize": p * 24,
    "xtick.labelsize": p * 18,
    "ytick.labelsize": p * 18,
    "legend.fontsize": p * 18,
    "axes.linewidth": p * 1.2
})

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv("all.csv")

problems_order = sorted(df["Problem"].unique())

# -------------------------------------------------
# Artificial bounds for visualization
# -------------------------------------------------
ymin = min(
    df["Final Error 1"].min(),
    df["Final Error 2"].min()
)

ymax = 5.0   # artificial upper bound (visual only)

# -------------------------------------------------
# Figure
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

# -------------------------------------------------
# Violin: Mean Absolute Error
# -------------------------------------------------
sns.violinplot(
    x="Problem",
    y="Final Error 1",
    data=df,
    inner="box",
    palette="Set2",
    order=problems_order,
    cut=0,
    bw_adjust=0.5,
    clip=(ymin, ymax),   # hard KDE bounds
    ax=axes[0]
)

axes[0].set_yscale("log")
axes[0].set_ylim(ymin, ymax)
axes[0].set_ylabel("Absolute Error")
axes[0].set_xlabel("Problem")
axes[0].set_title("Mean Absolute Error")

# -------------------------------------------------
# Violin: Max Absolute Error
# -------------------------------------------------
sns.violinplot(
    x="Problem",
    y="Final Error 2",
    data=df,
    inner="box",
    palette="Set2",
    order=problems_order,
    cut=0,
    bw_adjust=0.5,
    clip=(ymin, ymax),
    ax=axes[1]
)

axes[1].set_yscale("log")
axes[1].set_ylim(ymin, ymax)
axes[1].set_ylabel("")
axes[1].set_xlabel("Problem")
axes[1].set_title("Max Absolute Error")

# -------------------------------------------------
# Cosmetics
# -------------------------------------------------
for ax in axes:
    ax.tick_params(axis="both", which="major", length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3)

plt.tight_layout()
plt.savefig("vhist_errors_horizontal.png", dpi=300, bbox_inches="tight")
plt.close()
