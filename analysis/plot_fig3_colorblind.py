"""
Fig. 3 — MAE vs Total Neurons, grouped by Architecture Shape.
Each color = one architecture shape; lighter = Problem A, darker = Problem C.
The 4 neuron sizes shown are those with the largest spread between shapes
in Problem C (max - min of per-shape medians).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("../Results/aggregated_results.csv")
df = df[df["Problem"].isin(["A", "C"])].copy()
df["total_neurons"] = df["total_neurons"].astype(int)

SHAPES   = ["bottleneck", "bowtie", "diamond", "funnel", "rectangle"]
PROBLEMS = ["A", "C"]

# ── Select 4 neuron sizes with largest inter-shape spread in Problem C ────────
all_neurons = sorted(df["total_neurons"].unique().tolist())
df_c = df[df["Problem"] == "C"]

spread = {}
for n in all_neurons:
    vals = df_c.loc[df_c["total_neurons"] == n, "Final Error 1"].dropna().values
    spread[n] = vals.max() - vals.min() if len(vals) > 1 else 0.0

# Select 4 sizes: keep 176, find 3 others that together cover all shapes,
# balancing spread and even distribution across the neuron range.
from itertools import combinations as _combinations

shapes_at = {
    n: set(s for s in SHAPES
           if df_c.loc[(df_c["total_neurons"] == n) & (df_c["shape"] == s),
                       "Final Error 1"].dropna().shape[0] > 0)
    for n in all_neurons
}
_fixed    = 176
_remain   = [n for n in all_neurons if n != _fixed and len(shapes_at[n]) > 1]
_all_shp  = set(SHAPES)
_n_min, _n_max = min(all_neurons), max(all_neurons)
_max_spr  = max(spread.values())
_ideal_gap = (_n_max - _n_min) / 3

best_score, best_combo = -1, None
for combo in _combinations(_remain, 3):
    covered = shapes_at[_fixed] | shapes_at[combo[0]] | shapes_at[combo[1]] | shapes_at[combo[2]]
    if covered != _all_shp:
        continue
    spread_score   = sum(spread[n] for n in combo) / (3 * _max_spr)
    sizes          = sorted(list(combo) + [_fixed])
    gaps           = [sizes[i + 1] - sizes[i] for i in range(3)]
    evenness_score = 1 - np.std(gaps) / (_ideal_gap + 1e-9)
    range_score    = (sizes[-1] - sizes[0]) / (_n_max - _n_min)
    score          = 0.15 * spread_score + 0.35 * evenness_score + 0.50 * range_score
    if score > best_score:
        best_score, best_combo = score, combo

NEURONS = sorted(list(best_combo) + [_fixed])
print(f"Selected neuron sizes: {NEURONS}")
print(f"  spreads: { {n: f'{spread[n]:.4f}' for n in NEURONS} }")

# ── Color palette (one color per shape) ──────────────────────────────────────
BASE_COLORS = {
    "bottleneck": "#1f77b4",
    "bowtie":     "#ff7f0e",
    "diamond":    "#2ca02c",
    "funnel":     "#d62728",
    "rectangle":  "#9467bd",
}

def darken(hex_color, factor=0.45):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r2 = int(r * (1 - factor))
    g2 = int(g * (1 - factor))
    b2 = int(b * (1 - factor))
    return f"#{r2:02X}{g2:02X}{b2:02X}"

def lighten(hex_color, factor=0.45):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02X}{g2:02X}{b2:02X}"

COLORS = {
    "A": {s: lighten(BASE_COLORS[s], 0.45) for s in SHAPES},
    "C": {s: BASE_COLORS[s]                 for s in SHAPES},
}

# ── Layout ────────────────────────────────────────────────────────────────────
p = 1.0
plt.rcParams.update({
    "font.size":        p * 13,
    "axes.titlesize":   p * 14,
    "axes.labelsize":   p * 13,
    "xtick.labelsize":  p * 11,
    "ytick.labelsize":  p * 11,
    "legend.fontsize":  p * 11,
    "axes.linewidth":   1.2,
    "font.family":      "DejaVu Sans",
})

n_shapes        = len(SHAPES)
n_problems      = len(PROBLEMS)
n_neurons       = len(NEURONS)
boxes_per_group = n_shapes * n_problems   # 10

group_w = 0.72
box_w   = group_w / boxes_per_group

fig, ax = plt.subplots(figsize=(6, 5.5))
fig.subplots_adjust(right=0.80, bottom=0.14)

positions_base = np.arange(n_neurons) * (boxes_per_group * box_w + 0.15)

# ── Subtle spread markers for Problem C (drawn first, behind boxes) ───────────
for gi, n in enumerate(NEURONS):
    vals = df_c.loc[df_c["total_neurons"] == n, "Final Error 1"].dropna().values
    if len(vals) < 2:
        continue
    x_center = positions_base[gi]
    ax.plot([x_center, x_center], [vals.min(), vals.max()],
            color="black", linewidth=1.0, alpha=0.25, zorder=0,
            solid_capstyle="round")
    ax.scatter([x_center, x_center], [vals.min(), vals.max()],
               color="black", s=12, alpha=0.30, zorder=0, linewidths=0)
    proportion = vals.max() / vals.min()
    ax.text(x_center, vals.max()*0.95, f"×{proportion:.1f}",
            ha="center", va="bottom",
            color="black", alpha=0.40)

for si, shape in enumerate(SHAPES):
    for pi, problem in enumerate(PROBLEMS):
        idx = si * n_problems + pi
        offset = (idx - (boxes_per_group - 1) / 2) * box_w
        positions = positions_base + offset

        sub = df[(df["Problem"] == problem) & (df["shape"] == shape)]
        data_per_neuron = [
            sub.loc[sub["total_neurons"] == n, "Final Error 1"].dropna().values
            for n in NEURONS
        ]

        fc = COLORS[problem][shape]
        ec = darken(BASE_COLORS[shape], 0.45)

        ax.boxplot(
            data_per_neuron,
            positions=positions,
            widths=box_w * 0.88,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            boxprops=dict(facecolor=fc, edgecolor=ec, linewidth=1.8),
            medianprops=dict(color=ec, linewidth=2.2),
            whiskerprops=dict(color=ec, linewidth=1.4, linestyle="--"),
            capprops=dict(color=ec, linewidth=1.8),
        )

ax.set_yscale("log")
_half_group = (boxes_per_group * box_w) / 2 + box_w
ax.set_xlim(positions_base[0] - _half_group, positions_base[-1] + _half_group)
ax.set_xticks(positions_base)
ax.set_xticklabels(NEURONS)
ax.set_xlabel("Total Neurons", labelpad=6)
ax.set_ylabel("MAE (log scale)", labelpad=8)
ax.tick_params(axis="both", which="major", length=5, width=1.1)
ax.tick_params(axis="both", which="minor", length=3)
ax.grid(axis="y", which="major", linestyle="--", alpha=0.4, linewidth=0.8)
ax.grid(axis="y", which="minor", linestyle=":",  alpha=0.25, linewidth=0.6)

# ── Legend ────────────────────────────────────────────────────────────────────
shape_patches = [
    mpatches.Patch(facecolor=BASE_COLORS[s], edgecolor=BASE_COLORS[s],
                   linewidth=1.2, label=s.capitalize())
    for s in SHAPES
]
problem_patches = [
    mpatches.Patch(facecolor=lighten(BASE_COLORS["diamond"], 0.45),
                   edgecolor=BASE_COLORS["diamond"], linewidth=1.2,
                   label="Problem A  (lighter)"),
    mpatches.Patch(facecolor=BASE_COLORS["diamond"],
                   edgecolor=BASE_COLORS["diamond"], linewidth=1.2,
                   label="Problem C  (darker)"),
]

ax.legend(
    handles=shape_patches + [mpatches.Patch(facecolor="none", edgecolor="none")] + problem_patches,
    title=None,
    loc="upper left",
    bbox_to_anchor=(1.038, 1.0),
    frameon=True,
    framealpha=0.9,
    edgecolor="#888888",
)

ax.set_title("MAE vs. Total Neurons by Architecture Shape", pad=10)

# ── Inset: boxplot for all shapes, Problem B, 96 neurons ─────────────────────
_df_full = pd.read_csv("../Results/aggregated_results.csv")
_df_full["total_neurons"] = _df_full["total_neurons"].astype(int)
df_b176 = _df_full[(_df_full["Problem"] == "B") & (_df_full["total_neurons"] == 96)]

# figure coords: right of main axes, below the legend
ax_ins = fig.add_axes([0.828, 0.10, 0.148, 0.30])

for si, shape in enumerate(SHAPES):
    vals = df_b176.loc[df_b176["shape"] == shape, "Final Error 1"].dropna().values
    if len(vals) == 0:
        continue
    fc = BASE_COLORS[shape]
    ec = darken(BASE_COLORS[shape], 0.45)
    ax_ins.boxplot(
        [vals],
        positions=[si],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        boxprops=dict(facecolor=fc, edgecolor=ec, linewidth=1.2),
        medianprops=dict(color=ec, linewidth=1.6),
        whiskerprops=dict(color=ec, linewidth=1.0, linestyle="--"),
        capprops=dict(color=ec, linewidth=1.2),
    )

ax_ins.set_yscale("log")
ax_ins.set_xticks(range(len(SHAPES)))
ax_ins.set_xticklabels([])
ax_ins.set_ylabel("")
ax_ins.tick_params(axis="y", which="both", labelsize=9, length=3)
ax_ins.tick_params(axis="x", length=0)
ax_ins.set_title("Prob. B · 96 neurons", pad=4, fontsize=10)
ax_ins.grid(axis="y", which="major", linestyle="--", alpha=0.35, linewidth=0.6)

_ins_vals = df_b176["Final Error 1"].dropna().values
if len(_ins_vals) > 1:
    _ins_min, _ins_max = _ins_vals.min(), _ins_vals.max()
    _ins_x = (len(SHAPES) - 1) / 2 - 0.4
    ax_ins.plot([_ins_x, _ins_x], [_ins_min, _ins_max],
                color="black", linewidth=1.5, alpha=0.35, zorder=0,
                solid_capstyle="round")
    ax_ins.scatter([_ins_x, _ins_x], [_ins_min, _ins_max],
                   color="black", s=16, alpha=0.45, zorder=0, linewidths=0)
    _ins_prop = _ins_max / _ins_min
    _text_y = _ins_max * 0.70
    ax_ins.text(_ins_x, _text_y, f"×{_ins_prop:.1f}",
                ha="center", va="top", fontsize=11,
                color="black", alpha=0.55)
ax_ins.set_facecolor("#f9f9f9")
for spine in ax_ins.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor("#aaaaaa")

plt.savefig("../Results/fig3_colorblind.pdf", dpi=300, bbox_inches="tight")
plt.savefig("../Results/fig3_colorblind.png", dpi=300, bbox_inches="tight")
print("Saved: Results/fig3_colorblind.pdf  +  .png")
