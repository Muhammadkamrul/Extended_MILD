# plot_mild_comparisons.py
# Generates comparison figures for MILD vs LSTM vs MLP
# Outputs high-resolution PNG and SVG figures.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
OUT_DIR = "new_mild_comparison_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS = ["MILD", "LSTM", "MLP"]
INTENTS = ["analytics", "api", "telemetry"]
DATASETS = ["Ryu Cloud", "Microservice"]

COLORS = {
    "MILD": "#2E86AB",
    "LSTM": "#F18F01",
    "MLP": "#6A4C93",
}

# -----------------------------
# Data
# -----------------------------
rows = [
    # MILD - Ryu Cloud
    ["MILD", "Ryu Cloud", "analytics", 100.00, 0.00, 10.18, 0.32, None, None, None, None],
    ["MILD", "Ryu Cloud", "api",       100.00, 0.00, 7.15, 0.79, None, None, None, None],
    ["MILD", "Ryu Cloud", "telemetry", 75.59, 13.82, 7.09, 1.35, None, None, None, None],
    ["MILD", "Ryu Cloud", "Overall",   None, None, None, None, 8.45, 0.96, 88.72, 1.03],

    # MILD - Microservice
    ["MILD", "Microservice", "analytics", 83.81, 0.63, 30.44, 1.72, None, None, None, None],
    ["MILD", "Microservice", "api",       99.56, 0.16, 35.61, 0.08, None, None, None, None],
    ["MILD", "Microservice", "telemetry", 93.53, 0.20, 30.48, 0.86, None, None, None, None],
    ["MILD", "Microservice", "Overall",   None, None, None, None, 6.63, 0.17, 71.23, 0.95],

    # LSTM - Ryu Cloud
    ["LSTM", "Ryu Cloud", "analytics", 6.41, 1.28, 3.36, 0.14, None, None, None, None],
    ["LSTM", "Ryu Cloud", "api",       86.02, 10.59, 6.90, 0.89, None, None, None, None],
    ["LSTM", "Ryu Cloud", "telemetry", 15.29, 3.53, 7.65, 0.50, None, None, None, None],
    ["LSTM", "Ryu Cloud", "Overall",   None, None, None, None, 14.73, 8.37, 40.77, 3.33],

    # LSTM - Microservice
    ["LSTM", "Microservice", "analytics", 70.97, 13.45, 26.87, 2.94, None, None, None, None],
    ["LSTM", "Microservice", "api",       98.52, 0.65, 33.59, 1.75, None, None, None, None],
    ["LSTM", "Microservice", "telemetry", 91.06, 2.01, 27.77, 1.60, None, None, None, None],
    ["LSTM", "Microservice", "Overall",   None, None, None, None, 19.54, 6.99, 59.23, 10.06],

    # MLP - Ryu Cloud
    ["MLP", "Ryu Cloud", "analytics", 79.06, 3.85, 10.02, 1.24, None, None, None, None],
    ["MLP", "Ryu Cloud", "api",       61.02, 38.98, 8.65, 1.11, None, None, None, None],
    ["MLP", "Ryu Cloud", "telemetry", 26.47, 25.88, 7.19, 0.19, None, None, None, None],
    ["MLP", "Ryu Cloud", "Overall",   None, None, None, None, 5.09, 0.19, 51.79, 22.56],

    # MLP - Microservice
    ["MLP", "Microservice", "analytics", 64.69, 6.65, 14.29, 1.15, None, None, None, None],
    ["MLP", "Microservice", "api",       96.90, 1.89, 34.13, 0.81, None, None, None, None],
    ["MLP", "Microservice", "telemetry", 77.71, 4.62, 23.59, 1.22, None, None, None, None],
    ["MLP", "Microservice", "Overall",   None, None, None, None, 4.41, 1.44, 41.22, 2.61],
]

df = pd.DataFrame(
    rows,
    columns=[
        "Method",
        "Dataset",
        "Intent",
        "Detection Rate",
        "Detection Rate Std",
        "Avg Lead Time",
        "Avg Lead Time Std",
        "FP Rate/Day",
        "FP Rate/Day Std",
        "Root Cause Accuracy",
        "Root Cause Accuracy Std",
    ],
)

intent_df = df[df["Intent"] != "Overall"].copy()
overall_df = df[df["Intent"] == "Overall"].copy()


# -----------------------------
# Helper functions
# -----------------------------
def save_fig(fig, filename):
    png_path = os.path.join(OUT_DIR, f"{filename}.png")
    #svg_path = os.path.join(OUT_DIR, f"{filename}.svg")
    fig.savefig(png_path, bbox_inches="tight")
    #fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    #print(f"Saved: {svg_path}")


def grouped_bar_by_intent(dataset, metric, err_metric, ylabel, title, filename, ylim=None):
    data = intent_df[intent_df["Dataset"] == dataset]

    x = np.arange(len(INTENTS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for i, method in enumerate(METHODS):
        subset = data[data["Method"] == method].set_index("Intent").loc[INTENTS]

        values = subset[metric].values
        errors = subset[err_metric].values

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
            color=COLORS[method],
            yerr=errors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )

    #ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(INTENTS)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


def overall_grouped_bar(metric, err_metric, ylabel, title, filename, ylim=None, lower_is_better=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x = np.arange(len(DATASETS))
    width = 0.24

    for i, method in enumerate(METHODS):
        subset = overall_df[overall_df["Method"] == method].set_index("Dataset").loc[DATASETS]

        values = subset[metric].values
        errors = subset[err_metric].values

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
            color=COLORS[method],
            yerr=errors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )

    #ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if lower_is_better:
        ax.text(
            0.98,
            0.94,
            "Lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            style="italic",
        )

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    fig.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


def combined_overall_summary():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    metrics = [
        (
            "FP Rate/Day",
            "FP Rate/Day Std",
            "False Positives / Day",
            "Overall FP Rate",
            True,
        ),
        (
            "Root Cause Accuracy",
            "Root Cause Accuracy Std",
            "Root Cause Accuracy (%)",
            "Root Cause Accuracy",
            False,
        ),
    ]

    x = np.arange(len(DATASETS))
    width = 0.24

    for ax, (metric, err_metric, ylabel, title, lower_is_better) in zip(axes, metrics):
        for i, method in enumerate(METHODS):
            subset = overall_df[overall_df["Method"] == method].set_index("Dataset").loc[DATASETS]

            values = subset[metric].values
            errors = subset[err_metric].values

            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=method,
                color=COLORS[method],
                yerr=errors,
                capsize=4,
                edgecolor="black",
                linewidth=0.5,
            )

        #ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if metric == "Root Cause Accuracy":
            ax.set_ylim(0, 100)

        if lower_is_better:
            ax.text(
                0.98,
                0.94,
                "Lower is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                style="italic",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.suptitle("Overall Comparison: FP Rate and Root Cause Accuracy", y=1.12)
    fig.tight_layout()

    save_fig(fig, "overall_fp_and_root_cause_accuracy")
    plt.close(fig)


def detection_heatmap_like_table():
    """
    Makes a compact slide-friendly heatmap-style table for detection rates.
    This uses matplotlib only, no seaborn.
    """
    pivot = intent_df.pivot_table(
        index=["Dataset", "Intent"],
        columns="Method",
        values="Detection Rate",
    ).loc[
        [
            ("Ryu Cloud", "analytics"),
            ("Ryu Cloud", "api"),
            ("Ryu Cloud", "telemetry"),
            ("Microservice", "analytics"),
            ("Microservice", "api"),
            ("Microservice", "telemetry"),
        ],
        METHODS,
    ]

    values = pivot.values
    row_labels = [f"{dataset}\n{intent}" for dataset, intent in pivot.index]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    im = ax.imshow(values, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if values[i, j] < 55 else "black",
            )

    #ax.set_title("Detection Rate Summary")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection Rate (%)")

    fig.tight_layout()
    save_fig(fig, "detection_rate_summary_heatmap")
    plt.close(fig)


# -----------------------------
# Generate figures
# -----------------------------
grouped_bar_by_intent(
    dataset="Ryu Cloud",
    metric="Detection Rate",
    err_metric="Detection Rate Std",
    ylabel="Detection Rate (%)",
    title="Detection Rate by Intent — Ryu Cloud Data",
    filename="detection_rate_by_intent_ryu_cloud",
    ylim=(0, 110),
)

grouped_bar_by_intent(
    dataset="Microservice",
    metric="Detection Rate",
    err_metric="Detection Rate Std",
    ylabel="Detection Rate (%)",
    title="Detection Rate by Intent — Microservice Data",
    filename="detection_rate_by_intent_microservice",
    ylim=(0, 110),
)

grouped_bar_by_intent(
    dataset="Ryu Cloud",
    metric="Avg Lead Time",
    err_metric="Avg Lead Time Std",
    ylabel="Average Lead Time (min)",
    title="Average Lead Time by Intent — Ryu Cloud Data",
    filename="avg_lead_time_by_intent_ryu_cloud",
)

grouped_bar_by_intent(
    dataset="Microservice",
    metric="Avg Lead Time",
    err_metric="Avg Lead Time Std",
    ylabel="Average Lead Time (min)",
    title="Average Lead Time by Intent — Microservice Data",
    filename="avg_lead_time_by_intent_microservice",
)

overall_grouped_bar(
    metric="FP Rate/Day",
    err_metric="FP Rate/Day Std",
    ylabel="False Positives / Day",
    title="Overall False Positive Rate",
    filename="overall_fp_rate_per_day",
    lower_is_better=True,
)

overall_grouped_bar(
    metric="Root Cause Accuracy",
    err_metric="Root Cause Accuracy Std",
    ylabel="Root Cause Accuracy (%)",
    title="Overall Root Cause Accuracy",
    filename="overall_root_cause_accuracy",
    ylim=(0, 100),
)

def scenario_overall_metrics(dataset):
    """
    Creates a scenario-wise figure for one dataset showing:
    1. FP Rate/Day
    2. Root Cause Accuracy
    """

    data = overall_df[overall_df["Dataset"] == dataset].set_index("Method").loc[METHODS]

    metrics = [
        ("FP Rate/Day", "FP Rate/Day Std", "FP Rate/Day", "Lower is better"),
        ("Root Cause Accuracy", "Root Cause Accuracy Std", "Root Cause Accuracy (%)", None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (metric, err_metric, ylabel, note) in zip(axes, metrics):
        values = data[metric].values
        errors = data[err_metric].values

        x = np.arange(len(METHODS))

        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=5,
            color=[COLORS[m] for m in METHODS],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylabel(ylabel)
        #ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if metric == "Root Cause Accuracy":
            ax.set_ylim(0, 100)

        if note:
            ax.text(
                0.98,
                0.94,
                note,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                style="italic",
            )

        # Add value labels above bars
        for i, value in enumerate(values):
            ax.text(
                i,
                value + max(values) * 0.03,
                f"{value:.2f}",
                ha="right",
                va="bottom",
                fontsize=10,
            )

    #fig.suptitle(f"Scenario-wise Overall Metrics — {dataset}", y=1.05)
    #fig.tight_layout()

    filename = f"scenario_overall_metrics_{dataset.lower().replace(' ', '_')}"
    save_fig(fig, filename)
    plt.close(fig)


scenario_overall_metrics("Ryu Cloud")
scenario_overall_metrics("Microservice")

combined_overall_summary()
detection_heatmap_like_table()

print("\nDone. Figures saved in:", OUT_DIR)