#!/usr/bin/env python3
"""
Method Comparison Plotter
-------------------------
Builds publication-ready figures comparing methods across:
- Detection Rate (%) by intent (with std error bars)
- Average Lead Time (minutes) by intent (with std error bars)
- Overall False Positives per Day (mean ± std)
- NEW: Overall Root Cause Accuracy (mean ± std)
- Radar chart of overall profiles.

Notes:
- Uses matplotlib only (no seaborn), one chart per figure, and no explicit colors.
- Proposed methods are emphasized via hatching and thicker edges.
- Saves individual PNGs + a multipage PDF and CSVs of the plotted data.
- **MLP is included. MILD is drawn as the rightmost bar in grouped charts.**

Usage:
    python method_comparison_plots.py --outdir out_plots --dpi 200

Author: (generated, then revised)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import patheffects as pe
from matplotlib.patches import Patch


# -----------------------------
# Data from the user - NEW 11-FOLD CV RESULTS
# -----------------------------

ALL_METHODS = ["LSTM", "MLP", "MILD", "WKPI-Tuned", "Dist-Target", "LR-OvR"]
INTENTS = ["analytics", "api", "telemetry"]

# Detection Rate (%) with std - UPDATED
DET_RATE = pd.DataFrame({
    "intent": np.repeat(INTENTS, len(ALL_METHODS)),
    "method": ALL_METHODS * len(INTENTS),
    "mean": [
        96.33, 100.00, 100.00, 98.89, 68.67, 100.00,   # analytics
        96.67, 98.33, 100.00, 98.33, 88.33, 98.33,     # api
        98.33, 100.00, 100.00, 100.00, 78.17, 100.00   # telemetry
    ],
    "std": [
        7.37, 0.00, 0.00, 3.33, 35.38, 0.00,          # analytics
        10.00, 5.00, 0.00, 5.00, 25.87, 5.00,          # api
        5.00, 0.00, 0.00, 0.00, 29.84, 0.00           # telemetry
    ]
})

# Avg Lead Time (minutes) with std - UPDATED
LEAD_TIME = pd.DataFrame({
    "intent": np.repeat(INTENTS, len(ALL_METHODS)),
    "method": ALL_METHODS * len(INTENTS),
    "mean": [
        80.10, 83.20, 88.79, 71.18, 63.67, 74.68,      # analytics
        88.47, 91.99, 95.48, 76.61, 49.61, 65.44,      # api
        95.82, 103.12, 110.93, 87.48, 60.58, 107.90    # telemetry
    ],
    "std": [
        15.78, 11.31, 8.32, 7.56, 20.90, 12.00,        # analytics
        18.95, 19.78, 10.00, 15.24, 13.52, 14.16,      # api
        23.55, 18.72, 11.02, 10.80, 22.27, 10.14       # telemetry
    ]
})

# Overall metrics - UPDATED
OVERALL_FP = pd.DataFrame({
    "method": ALL_METHODS,
    "mean": [5.35, 5.05, 3.90, 35.73, 9.40, 6.23],
    "std":  [6.69, 5.92, 3.74, 58.33, 9.98, 9.25]
})

# Overall Disambiguation - NEW STRUCTURE
OVERALL_DISAMBIG = pd.DataFrame({
    "method": ALL_METHODS,
    "mean": [80.60, 81.03, 88.67, 66.68, 60.82, 79.02],
    "std":  [11.64, 11.26, 7.96, 12.04, 22.80, 12.29]
})
# -----------------------------
# End of Data
# -----------------------------


# Hatch patterns + edge widths to emphasize proposed methods
HATCHES = {
    "LSTM": "|||",
    "MLP": "///",
    "MILD": "\\\\",
    "WKPI-Tuned": "",
    "Dist-Target": "..",
    "LR-OvR": "xx",
}
EDGE_WIDTHS = {m: (2.0 if m in ["MLP", "MILD"] else 0.9) for m in ALL_METHODS}

# -----------------------------
# Exclusions & default order
# -----------------------------
EXCLUDE_METHODS = {}
# For grouped charts, put MILD last (rightmost)
ORDERED_METHODS_GROUPED = [m for m in ALL_METHODS if m not in EXCLUDE_METHODS and m != "MILD"] + ["MILD"]

# -----------------------------
# Helper plotting utilities
# -----------------

def _scale(values, mode="best", invert=False, clip=(0.0, 1.0), targets=None, eps=1e-9):
    """
    Normalize to [0,1].
    mode:
      - "minmax": linear min-max; set targets=(lo, hi) to use fixed bounds.
      - "best":   percent-of-best (higher-better: v/max; lower-better: min/v).
    invert: if True, higher raw values are worse (e.g., FP/day).
    targets: optional (lo, hi) bounds for "minmax" when you want fixed axes.
    """
    v = np.asarray(values, dtype=float)
    if mode == "minmax":
        lo, hi = (np.min(v), np.max(v)) if targets is None else targets
        if hi - lo < eps:
            s = np.ones_like(v)
        else:
            s = (v - lo) / (hi - lo)
    elif mode == "best":
        if invert:
            s = (np.min(v) + eps) / (v + eps)         # lower is better
        else:
            s = v / (np.max(v) + eps)                 # higher is better
    else:
        raise ValueError("Unknown mode")
    if invert:
        s = 1.0 - s if mode == "minmax" else s        # already inverted in "best"
    return np.clip(s, *clip)

def _minmax(series):
    s = np.asarray(series, dtype=float)
    mn, mx = float(np.min(s)), float(np.max(s))
    if mx - mn < 1e-12:  # all equal -> perfect ring
        return np.ones_like(s, dtype=float)
    return (s - mn) / (mx - mn)


def _annotate_value(ax, x, y, text, ha="right"):
    """Draw text with a subtle background halo to prevent lines from piercing through."""
    halo = ax.figure.get_facecolor()
    t = ax.text(x, y, text, va="center", ha=ha, zorder=5, clip_on=False)
    t.set_path_effects([pe.withStroke(linewidth=4, foreground=halo)])
    return t


# --- shared plotting style (place near other helpers) ---
def _apply_pub_style(ax):
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(axis="both", length=4, width=1)

# --- broken x-axis helper for the diagonal "break" marks ---
def _add_xaxis_break_marks(ax_left, ax_right, size=0.015):
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1.0)
    ax_left.plot((1 - size, 1 + size), (-size, +size), **kwargs)
    ax_left.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-size, +size), (-size, +size), **kwargs)
    ax_right.plot((-size, +size), (1 - size, 1 + size), **kwargs)

def _add_bar_value_labels(ax, fmt="{:.1f}", offset_frac=0.01, rotation=0):
    ymax = ax.get_ylim()[1]
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h):
            continue
        x = patch.get_x() + patch.get_width()/2
        ax.text(x, h + ymax * offset_frac, fmt.format(h), ha="center", va="bottom", rotation=rotation)

def grouped_bar_plot(df, category_col, method_col, value_col, err_col=None,
                     ylabel="", title="", ylim=None, out_path=None, as_percent=False, dpi=200,
                     method_order=None):
    cats = list(df[category_col].unique())
    # keep MILD rightmost
    present_methods = [m for m in (method_order or ORDERED_METHODS_GROUPED) if m in df[method_col].unique()]
    n_cat = len(cats)
    n_meth = len(present_methods)

    fig = plt.figure(figsize=(10.5, 6.2)) 
    ax = plt.gca()

    x = np.arange(n_cat)

    # Adjust bar width based on number of methods
    gap = 0.02
    if n_meth == 5:
        width = 0.12
    elif n_meth == 4:
        width = 0.17
    else:
        width = 0.22

    offsets = np.linspace(-(n_meth-1)/2, (n_meth-1)/2, n_meth) * (width + gap)

    for i, m in enumerate(present_methods):
        vals = [df[(df[category_col]==c) & (df[method_col]==m)][value_col].values[0] for c in cats]
        errs = None
        if err_col is not None:
            errs = [df[(df[category_col]==c) & (df[method_col]==m)][err_col].values[0] for c in cats]
        
        bars = ax.bar(x + offsets[i], vals, width, yerr=errs, capsize=3, label=m,
            hatch=HATCHES.get(m, ""), linewidth=EDGE_WIDTHS.get(m, 1.0), edgecolor="black")

        # --- Add labels manually to position them *above* the error bars ---
        fmt = "{:.1f}" if as_percent else "{:.2f}"
        ymax = ax.get_ylim()[1]
        offset = ymax * 0.01 

        for j, bar in enumerate(bars):
            h = bar.get_height()
            std = errs[j] if errs is not None else 0 
            y_pos = h + std + offset 
            if np.isnan(h): continue 

            ax.text(bar.get_x() + bar.get_width()/2, y_pos, fmt.format(h),
                    ha="center", va="bottom", rotation=90)
        # --- END OF NEW BLOCK ---

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend(ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.18), loc="upper center")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig


def horizontal_bar_with_error(df, value_col, err_col, title, xlabel,
                              sort=True, out_path=None, dpi=200, as_percent=False):
    d = df.copy()
    if sort:
        d = d.sort_values(by=value_col, ascending=True)

    vals = d[value_col].to_numpy(dtype=float)
    errs = d[err_col].to_numpy(dtype=float)
    labels = d["method"].tolist()
    n = len(d)

    hi = float(np.max(vals + errs))
    if n > 1:
        idx_hi = int(np.argmax(vals + errs))
        small_hi = float(np.max(np.delete(vals + errs, idx_hi)))
    else:
        small_hi = hi

    use_broken = (small_hi > 0) and (hi / small_hi >= 3.0)

    if not use_broken:
        # ------- Single-axis version -------
        fig = plt.figure(figsize=(10.5, 6.2), constrained_layout=True)
        ax = fig.gca()

        y = np.arange(n)

        ax.barh(y, vals, xerr=errs, capsize=4,
                hatch=[HATCHES.get(m, "") for m in labels],
                edgecolor="black",
                linewidth=[EDGE_WIDTHS.get(m, 1.0) for m in labels],
                zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)

        right = hi * 1.12 if hi > 0 else 1.0
        ax.set_xlim(0, right)

        _apply_pub_style(ax)

        # --- Add legend box for hatches ---
        legend_handles = [
            Patch(facecolor='white',
                hatch=HATCHES.get(m, ""),
                edgecolor="black",
                label=m,
                linewidth=EDGE_WIDTHS.get(m, 1.0))
            for m in labels
        ]
        # ax.legend(handles=legend_handles,
        #         title="Method",
        #         frameon=False,
        #         bbox_to_anchor=(1.02, 1.0),
        #         loc="upper left")

        # Annotate with halo
        x0, x1 = ax.get_xlim()
        pad = (x1 - x0) * 0.02
        fmt = "{:.2f}%" if as_percent else "{:.2f}"
        for i, (val, std) in enumerate(zip(vals, errs)):
            x = min(val + std + pad, x1 - pad)
            text = f"{fmt.format(val)} ± {fmt.format(std)}"
            _annotate_value(ax, x, i, text, ha="right")

        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------- Broken x-axis version -------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(10.5, 6.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1]}
    )

    y = np.arange(n)

    for ax in (ax1, ax2):
        ax.barh(y, vals, xerr=errs, capsize=4,
                hatch=[HATCHES.get(m, "") for m in labels],
                edgecolor="black",
                linewidth=[EDGE_WIDTHS.get(m, 1.0) for m in labels],
                zorder=2)

    left_max  = small_hi * 1.15
    right_min = left_max * 1.02
    right_max = hi * 1.08

    ax1.set_xlim(0, left_max)
    ax2.set_xlim(right_min, right_max)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])

    ax1.set_xlabel(xlabel)

    _apply_pub_style(ax1)
    _apply_pub_style(ax2)

    _add_xaxis_break_marks(ax1, ax2, size=0.012)

    # Annotate on the appropriate axis
    x0L, x1L = ax1.get_xlim(); padL = (x1L - x0L) * 0.02
    x0R, x1R = ax2.get_xlim(); padR = (x1R - x0R) * 0.04
    fmt = "{:.2f}%" if as_percent else "{:.2f}"
    for i, (val, std) in enumerate(zip(vals, errs)):
        text = f"{fmt.format(val)} ± {fmt.format(std)}"
        if val + std <= left_max:
            x = min(val + std + padL, x1L - padL)
            _annotate_value(ax1, x, i, text, ha="right")
        else:
            x = min(val + std + padR, x1R - padR)
            _annotate_value(ax2, x, i, text, ha="right")

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig


def paired_group_bars(df, cols, title, ylabel, ylim=None, out_path=None, dpi=200, method_order=None):
    # two metrics per method as grouped bars
    methods = [m for m in (method_order or ORDERED_METHODS_GROUPED) if m in df["method"].unique()]
    d = df.set_index("method").loc[methods].reset_index()

    x = np.arange(len(methods))
    width = 0.34

    fig = plt.figure(figsize=(10.5, 6.2))
    ax = plt.gca()
    b1 = ax.bar(x - width/2, d[cols[0]], width, label=cols[0], edgecolor="black", linewidth=1.0)
    b2 = ax.bar(x + width/2, d[cols[1]], width, label=cols[1], edgecolor="black", linewidth=1.0)

    for bars in (b1, b2):
        for rect, m in zip(bars, methods):
            rect.set_hatch(HATCHES.get(m, ""))
            rect.set_linewidth(EDGE_WIDTHS.get(m, 1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ymax = ax.get_ylim()[1]
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + ymax*0.03, f"{h:.2f}",
                    ha="center", va="bottom", rotation=90)

    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig

def radar_chart_detection(df, title, out_path=None, dpi=200):
    # One radar chart showing profiles for all (non-excluded) methods
    cats = INTENTS
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    methods = [m for m in ORDERED_METHODS_GROUPED if m in df["method"].unique()]

    fig = plt.figure(figsize=(7.8, 7.8))
    ax = plt.subplot(111, polar=True)

    for m in methods:
        vals = [df[(df["intent"]==c) & (df["method"]==m)]["mean"].values[0] for c in cats]
        vals += vals[:1]
        line, = ax.plot(angles, vals, linewidth=2 if m in ["MILD"] else 1, label=m)
        ax.fill(angles, vals, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([20, 40, 60, 80, 100])
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig

def radar_chart_overall_metrics(det_df, lead_df, fp_df, dis_df, title,
                                out_path=None, dpi=200, method_order=None,
                                highlight="MILD",
                                scaling="best",
                                targets=None):
    methods = [m for m in (method_order or ORDERED_METHODS_GROUPED)
               if m in det_df["method"].unique()
               and m in lead_df["method"].unique()
               and m in fp_df["method"].unique()
               and m in dis_df["method"].unique()]

    # REVISED: Use new single disambiguation metric
    det = det_df.groupby("method")["mean"].mean().reindex(methods)                    # ↑ better
    lt  = lead_df.groupby("method")["mean"].mean().reindex(methods)                   # ↑ better
    fp  = fp_df.set_index("method")["mean"].reindex(methods)                           # ↓ better
    rca = dis_df.set_index("method")["mean"].reindex(methods)                          # ↑ better

    det_n = _scale(det, mode=scaling, invert=False, targets=None if targets is None else targets.get("det"))
    lt_n  = _scale(lt,  mode=scaling, invert=False, targets=None if targets is None else targets.get("lt"))
    fp_n  = _scale(fp,  mode=scaling, invert=True,  targets=None if targets is None else targets.get("fp"))
    rca_n = _scale(rca, mode=scaling, invert=False, targets=None if targets is None else targets.get("rca"))

    # REVISED: 4 axes, including FP rate
    axes_labels = ["Failure\nDetection",
                   "Lead Time",
                   "Reliability\n(Low FP)",
                   "Root Cause\nAccuracy"]

    metrics_by_method = {m: [det_n[i], lt_n[i], fp_n[i], rca_n[i]]
                         for i, m in enumerate(methods)}

    N = len(axes_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9.8, 9.8))
    ax = plt.subplot(111, polar=True)

    # Fills under everything
    for m in methods:
        vals = metrics_by_method[m] + metrics_by_method[m][:1]
        ax.fill(angles, vals, alpha=0.06 if m != highlight else 0.10, zorder=1)

    # Outlines on top with distinct linestyles/markers
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'P', 'X']
    for i, m in enumerate(methods):
        vals = metrics_by_method[m] + metrics_by_method[m][:1]
        lw = 2.2 if m == highlight else 1.4
        ls = line_styles[i % len(line_styles)]
        mk = markers[i % len(markers)]
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, marker=mk, markersize=4,
                label=m, zorder=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"])
    ax.set_ylim(0, 1)

    # REVISED: Simplified label adjustment
    labels = ax.get_xticklabels()
    labels[0].set_ha('left') # Detection
    labels[1].set_ha('left')   # Lead Time
    labels[2].set_ha('right') # FP Rate
    labels[3].set_ha('left')  # Root Cause Acc
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig



# -----------------------------
# Build all figures
# -----------------------------
def build_all(outdir: str, dpi: int = 200):
    os.makedirs(outdir, exist_ok=True)
    figs = []

    # Filter out excluded methods for all downstream plots/exports
    det_rate_f = DET_RATE[~DET_RATE["method"].isin(EXCLUDE_METHODS)].copy()
    lead_time_f = LEAD_TIME[~LEAD_TIME["method"].isin(EXCLUDE_METHODS)].copy()
    overall_fp_f = OVERALL_FP[~OVERALL_FP["method"].isin(EXCLUDE_METHODS)].copy()
    
    # REVISED: Use new disambiguation DataFrame structure
    overall_disambig_f = OVERALL_DISAMBIG[~OVERALL_DISAMBIG["method"].isin(EXCLUDE_METHODS)].copy()
    # Add a "Root Cause Acc. (%)" column for the horizontal bar plot function
    overall_disambig_f["Root Cause Acc. (%)"] = overall_disambig_f["mean"]


    # 1) Detection Rate — grouped bars (MILD rightmost)
    figs.append(grouped_bar_plot(
        det_rate_f, category_col="intent", method_col="method", value_col="mean", err_col="std",
        ylabel="Failure Detection Rate (%)",
        title="Detection Rate by Intent — Proposed Methods vs Baselines",
        ylim=(0, 150),
        out_path=os.path.join(outdir, "detection_rate_by_intent.png"),
        as_percent=True, dpi=dpi
    ))

    # 2) Lead Time — grouped bars (MILD rightmost)
    max_lt = (lead_time_f["mean"] + lead_time_f["std"]).max()
    figs.append(grouped_bar_plot(
        lead_time_f, category_col="intent", method_col="method", value_col="mean", err_col="std",
        ylabel="Average Lead Time (minutes)",
        title="Average Lead Time by Intent — Higher is Better",
        ylim=(0, max(130, float(max_lt) + 50)),
        out_path=os.path.join(outdir, "lead_time_by_intent.png"),
        as_percent=False, dpi=dpi
    ))

    # 3) FP per day — horizontal bars
    figs.append(horizontal_bar_with_error(
        overall_fp_f, value_col="mean", err_col="std",
        title="Overall False Positives per Day — Lower is Better",
        xlabel="FP Rate / Day",
        sort=True,
        out_path=os.path.join(outdir, "fp_rate_overall.png"),
        dpi=dpi,
        as_percent=False
    ))

    # 4) REVISED: Disambiguation — horizontal bars
    figs.append(horizontal_bar_with_error(
        overall_disambig_f, value_col="mean", err_col="std",
        title="Overall Root Cause Accuracy — Higher is Better",
        xlabel="Root Cause Accuracy (%)",
        sort=True,
        out_path=os.path.join(outdir, "root_cause_accuracy.png"),
        dpi=dpi,
        as_percent=True # Add this flag to format labels as %
    ))

    # 5) Radar chart — detection profiles
    figs.append(radar_chart_detection(
        det_rate_f, title="Detection Rate Profiles Across Intents (Radar)",
        out_path=os.path.join(outdir, "detection_rate_radar.png"),
        dpi=dpi
    ))

    # 6) Overall metrics radar (normalized, FP/day inverted)
    figs.append(radar_chart_overall_metrics(
        det_rate_f, lead_time_f, overall_fp_f, overall_disambig_f,
        title="Overall Metrics (normalized)",
        out_path=os.path.join(outdir, "overall_metrics_radar.png"),
        dpi=dpi
    ))

    # Save multipage PDF
    pdf_path = os.path.join(outdir, "method_comparison_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")

    # REVISED: CSV and PNG lists
    png_files = [
            os.path.join(outdir, "detection_rate_by_intent.png"),
            os.path.join(outdir, "lead_time_by_intent.png"),
            os.path.join(outdir, "fp_rate_overall.png"),
            os.path.join(outdir, "root_cause_accuracy.png"), # Renamed
            os.path.join(outdir, "detection_rate_radar.png"),
            os.path.join(outdir, "overall_metrics_radar.png"),
    ]
    csv_files = [
            "det_rate_by_intent.csv",
            "lead_time_by_intent.csv",
            "overall_fp.csv",
            "overall_disambiguation.csv", # This CSV now contains mean/std
    ]

    # CSV exports (filtered)
    det_rate_f.to_csv(os.path.join(outdir, csv_files[0]), index=False)
    lead_time_f.to_csv(os.path.join(outdir, csv_files[1]), index=False)
    overall_fp_f.to_csv(os.path.join(outdir, csv_files[2]), index=False)
    overall_disambig_f.to_csv(os.path.join(outdir, csv_files[3]), index=False)


    return {
        "pngs": png_files,
        "pdf": pdf_path,
        "csvs": csv_files,
    }

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    val = 10
    p = argparse.ArgumentParser(description="Generate comparison plots for methods.")
    p.add_argument("--outdir", type=str, default="plots", help="Output directory for images and CSVs.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures.")

    p.add_argument("--label-size", type=int, default=val, help="Font size for axis labels.")
    p.add_argument("--tick-size", type=int, default=val, help="Font size for axis tick labels.")
    p.add_argument("--legend-size", type=int, default=val, help="Font size for legends.")

    return p.parse_args()

def main():
    args = parse_args()

    # Apply global font settings from arguments
    plt.rcParams.update({
        'axes.labelsize': args.label_size,    # For x and y labels
        'xtick.labelsize': args.tick_size,    # For x-axis tick labels
        'ytick.labelsize': args.tick_size,    # For y-axis tick labels
        'legend.fontsize': args.legend_size,  # For legends
        'font.size': args.tick_size,          # Default for other text
    })
    
    results = build_all(args.outdir, dpi=args.dpi)
    print("Saved PNGs:")
    for pth in results["pngs"]:
        print(" -", pth)
    print("Saved PDF: ", results["pdf"])
    print("Saved CSVs:")
    for pth in results["csvs"]:
        print(" -", os.path.join(args.outdir, pth))

if __name__ == "__main__":
    main()
