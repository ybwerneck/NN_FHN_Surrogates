"""
ANOVA analysis from aggregated CSV.
Produces a results table per problem and combined.

Usage:
    python anova_from_csv.py <csv_path>
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

SPLIT = 120

# ── Load & map ────────────────────────────────────────────────────────────────
csv_path = sys.argv[1]
df_raw   = pd.read_csv(csv_path)

df_all = pd.DataFrame({
    "neuron":   df_raw["total_neurons"],
    "mean_err": df_raw["Final Error 1"],
    "max_err":  df_raw["Final Error 2"],
    "iccs":     df_raw["shape"],
    "Problem":  df_raw["Problem"] if "Problem" in df_raw.columns else "ALL",
}).dropna()

print(df_all["iccs"])
# ── Per-subset stats ──────────────────────────────────────────────────────────
def get_stats(df):
    small = df[df["neuron"] <= SPLIT]
    large = df[df["neuron"] >  SPLIT]

    def ancova(sub):
        m = ols("mean_err ~ C(iccs)*neuron", data=sub).fit()
        t = sm.stats.anova_lm(m, typ=1)
        return t

    t_s = ancova(small)
    t_l = ancova(large)

    def variances(sub):
        # inter: total variance (shape + activation both vary)
        inter = sub["mean_err"].var()
        # intra: mean within-shape variance (only activation varies)
        intra = sub.groupby("iccs")["mean_err"].var().mean()
        return inter, intra

    inter_s, intra_s = variances(small)
    inter_l, intra_l = variances(large)

    return dict(
        n=len(df), n_small=len(small), n_large=len(large),
        F_shape_small  = t_s.loc["C(iccs)", "F"],
        p_shape_small  = t_s.loc["C(iccs)", "PR(>F)"],
        F_neuron_small = t_s.loc["neuron",  "F"],
        p_neuron_small = t_s.loc["neuron",  "PR(>F)"],
        F_inter_small  = t_s.loc["C(iccs):neuron", "F"],
        p_inter_small  = t_s.loc["C(iccs):neuron", "PR(>F)"],
        F_shape_large  = t_l.loc["C(iccs)", "F"],
        p_shape_large  = t_l.loc["C(iccs)", "PR(>F)"],
        F_neuron_large = t_l.loc["neuron",  "F"],
        p_neuron_large = t_l.loc["neuron",  "PR(>F)"],
        F_inter_large  = t_l.loc["C(iccs):neuron", "F"],
        p_inter_large  = t_l.loc["C(iccs):neuron", "PR(>F)"],
        inter_s=inter_s, intra_s=intra_s,
        inter_l=inter_l, intra_l=intra_l,
        ratio_s=inter_s/intra_s, ratio_l=inter_l/intra_l,
    )


# ── Collect rows ──────────────────────────────────────────────────────────────
rows = {}
for prob in sorted(df_all["Problem"].unique()):
    rows[prob] = get_stats(df_all[df_all["Problem"] == prob])
rows["Combined"] = get_stats(df_all)


# ── Print tables ──────────────────────────────────────────────────────────────
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def pf(f): return f"{f:6.1f}"
def pp(p): return f"{p:.2e}"

labels = list(rows.keys())

# Table 1 — Split ANCOVA F-values
print("\nTable 1 — Split ANCOVA (mean_err ~ C(shape)*neuron, type 1)")
print(f"{'':12} {'─── Small models (≤120 neurons) ───':42} {'─── Large models (>120 neurons) ────':42}")
print(f"{'Problem':12} {'Shape F':>10} {'p':>10} {'Neuron F':>10} {'p':>10} {'Inter F':>10}   "
      f"{'Shape F':>10} {'p':>10} {'Neuron F':>10} {'p':>10} {'Inter F':>10}")
print("─" * 115)
for lbl, s in rows.items():
    print(f"{lbl:12} "
          f"{pf(s['F_shape_small']):>10} {pp(s['p_shape_small']):>10}{sig(s['p_shape_small']):3} "
          f"{pf(s['F_neuron_small']):>10} {pp(s['p_neuron_small']):>10}{sig(s['p_neuron_small']):3} "
          f"{pf(s['F_inter_small']):>10}   "
          f"{pf(s['F_shape_large']):>10} {pp(s['p_shape_large']):>10}{sig(s['p_shape_large']):3} "
          f"{pf(s['F_neuron_large']):>10} {pp(s['p_neuron_large']):>10}{sig(s['p_neuron_large']):3} "
          f"{pf(s['F_inter_large']):>10}")
print("─" * 115)
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")

# Table 2 — Variance
print("\nTable 2 — Inter/intra-group variance of mean_err by shape")
print(f"{'':12} {'─── Small models (≤120) ──────────────':46} {'─── Large models (>120) ──────────────':46}")
print(f"{'Problem':12} {'Inter-group':>14} {'Intra-group':>14} {'Ratio':>8}   "
      f"{'Inter-group':>14} {'Intra-group':>14} {'Ratio':>8}")
print("─" * 100)
for lbl, s in rows.items():
    print(f"{lbl:12} "
          f"{s['inter_s']:>14.3e} {s['intra_s']:>14.3e} {s['ratio_s']:>8.3f}   "
          f"{s['inter_l']:>14.3e} {s['intra_l']:>14.3e} {s['ratio_l']:>8.3f}")
print("─" * 100)
