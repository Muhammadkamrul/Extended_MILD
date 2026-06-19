import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------
# Data
# ------------------------
METHODS = ["MILD", "Weighted-KPI", "Distance", "Logistic (OvR)", "MLP"]
OVERALL_FP = pd.DataFrame({
    "method": METHODS,
    "mean": [3.90, 35.73, 9.40, 6.23, 5.05],
    "std":  [3.74, 58.33, 9.98, 9.25, 5.92]
})

# Hatch patterns and edge widths
HATCHES = {
    "MILD": "///",
    "Weighted-KPI": "\\\\",
    "Distance": "..",
    "Logistic (OvR)": "xx",
    "MLP": "--"
}
EDGE_WIDTHS = {m: 2.0 if m == "MILD" else 1.2 for m in METHODS}
bar_color = "#4C72B0"

# Sort by FP rate
df = OVERALL_FP.sort_values("mean", ascending=True)
y_pos = range(len(df))

# ------------------------
# Broken X-axis limits
# ------------------------
left_max = 22  # small FP values
right_min = left_max * 1.02
# Ensure full range including error bars is captured
right_max = df["mean"].max() + df["std"].max() + 5

# ------------------------
# Create figure with two horizontal axes
# ------------------------
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, sharey=True, figsize=(10, 6),
    gridspec_kw={"width_ratios": [3, 1]}
)

# Draw bars on both axes
for ax in (ax_left, ax_right):
    ax.barh(
        y=y_pos,
        width=df["mean"],
        xerr=df["std"],
        capsize=6,
        color=bar_color,
        edgecolor="black",
        linewidth=[EDGE_WIDTHS[m] for m in df["method"]],
        hatch=[HATCHES[m] for m in df["method"]],
        zorder=3
    )

# Set x-limits
#ax_left.set_xlim(0, left_max)
ax_left.set_xlim(0, left_max) 

ax_right.set_xlim(right_min, right_max)

#ax_left.set_xticks([-5, 0, 5, 10, 15, 20])

# Remove touching spines
ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['right'].set_visible(False) 

# Hide y-ticks on right
ax_right.set_yticks([])
#ax_left.set_yticks(y_pos)
#ax_left.set_yticklabels(df["method"], fontsize=12)

# ------------------------
# Annotate bars with mean ± std
# ------------------------
for i, (mean, std) in enumerate(zip(df["mean"], df["std"])):
    if mean <= left_max:
        ax_left.text(mean + std + 0.5, i, f"{mean:.2f} ± {std:.2f}", va='center', ha='left', fontsize=16)
    else:
        ax_right.text(mean + std + 0.5, i, f"{mean:.2f} ± {std:.2f}", va='center', ha='left', fontsize=16)

# ------------------------
# Diagonal break marks
# ------------------------
def add_break_marks(ax1, ax2, size=0.015):
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.2)
    ax1.plot((1 - size, 1 + size), (-size, +size), **kwargs)
    ax1.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-size, +size), (-size, +size), **kwargs)
    ax2.plot((-size, +size), (1 - size, 1 + size), **kwargs)

add_break_marks(ax_left, ax_right)

# --- Add dotted connection for the broken bar (Weighted-KPI) ---
wk_name = "Weighted-KPI"
wk_idx = list(df["method"]).index(wk_name)

# Coordinates of right edge of the left axis and left edge of the right axis
y = wk_idx
x_left = ax_left.get_xlim()[1]
x_right = ax_right.get_xlim()[0]

# Draw dotted connector lines across the break
for offset in [-0.4, 0.4]:  # small vertical offsets for visual thickness
    ax_left.plot([x_left, x_left + 8], [y + offset, y + offset],
                 color='black', linestyle=':', linewidth=1.5, clip_on=False)
    ax_right.plot([x_right - 8, x_right], [y + offset, y + offset],
                  color='black', linestyle=':', linewidth=1.5, clip_on=False)


# ------------------------
# Style, labels, grid, title
# ------------------------
for ax in (ax_left, ax_right):
    #ax.set_xlabel("FP Rate / Day", fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=1)

#fig.suptitle("Comparison of FP Rate per Day by Method", fontsize=14, weight='bold')

# ------------------------
# Legend
# ------------------------
legend_handles = [
    Patch(facecolor=bar_color, hatch=HATCHES[m], edgecolor='black', linewidth=EDGE_WIDTHS[m], label=m)
    for m in df["method"]
]
ax_right.legend(handles=legend_handles, title="Method", loc="lower right", frameon=False)

# X-axis ticks
ax_left.tick_params(axis='x', labelsize=16)
ax_right.tick_params(axis='x', labelsize=16)

# Axis labels
ax_left.set_xlabel("FP Rate / Day ± Standard Deviation", fontsize=16)
#ax_right.set_xlabel("FP Rate / Day", fontsize=16)



# Legend
ax_right.legend(handles=legend_handles, loc="lower right",
                frameon=False, fontsize=16, title_fontsize=14)


plt.tight_layout()
plt.show()
