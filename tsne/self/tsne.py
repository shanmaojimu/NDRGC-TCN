import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics.pairwise import euclidean_distances

plt.rcParams['font.family'] = 'Times New Roman'

# ========================== Configuration ==========================
sub = 1                                      # Change this: 1, 5, 13, 6, etc.
tsne_file = f"tsne_augmented_Sub{sub:02}.csv"

# ========================== Load Data ==========================
tsne_data = pd.read_csv(tsne_file)

plot_data = tsne_data.rename(columns={
    'label': 'Group',
    'x': 'Dim1',
    'y': 'Dim2'
})

# ========================== Data Cleaning ==========================
# Replace inf/-inf with NaN and drop missing values
plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()

# Convert Group to string for consistency
plot_data['Group'] = plot_data['Group'].astype(str)

# Map numeric labels to meaningful class names
label_map = {
    '0': 'Left Hand',
    '1': 'Right Hand',
}
plot_data['Group'] = plot_data['Group'].map(label_map)

# Define desired group order
groups = ['Left Hand', 'Right Hand']

# Assign colors in order
color_list = ["#8DBF8D", "#DDA0DD"]
sci_colors = {group: color_list[i % len(color_list)] for i, group in enumerate(groups)}

# ========================== Set Plot Limits (with 10% padding) ==========================
x_min = plot_data['Dim1'].min()
x_max = plot_data['Dim1'].max()
y_min = plot_data['Dim2'].min()
y_max = plot_data['Dim2'].max()
print(f"Original range - X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}]")

extend_ratio = 0.10
x_diff = x_max - x_min
y_diff = y_max - y_min

x_lim = [x_min - x_diff * extend_ratio, x_max + x_diff * extend_ratio]
y_lim = [y_min - y_diff * extend_ratio, y_max + y_diff * extend_ratio]
print(f"Extended range - X: [{x_lim[0]:.1f}, {x_lim[1]:.1f}], Y: [{y_lim[0]:.1f}, {y_lim[1]:.1f}]")

# ========================== Create Figure ==========================
fig = plt.figure(figsize=(8, 6.5))

# Main title
fig.suptitle('w/o N-R & PF & M-SDF', fontsize=16)

# Use GridSpec to simulate marginal density layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, width_ratios=[5, 0.4], height_ratios=[0.4, 5], wspace=0, hspace=0)

# ========================== Top Marginal Density ==========================
ax_top = fig.add_subplot(gs[0, 0])
for group, color in sci_colors.items():
    group_df = plot_data[plot_data['Group'] == group]
    if not group_df.empty:
        sns.kdeplot(
            data=group_df,
            x='Dim1', fill=True, alpha=0.4, color=color, linewidth=0.5, ax=ax_top
        )

ax_top.set_xlim(x_lim)
ax_top.set_yticks([])
ax_top.set_xticks([])
ax_top.set_xlabel("")
ax_top.set_ylabel("")
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['left'].set_visible(False)

# Top-right spacer
ax_spacer = fig.add_subplot(gs[0, 1])
ax_spacer.axis('off')

# ========================== Main t-SNE Scatter Plot ==========================
ax_main = fig.add_subplot(gs[1, 0])
for group, color in sci_colors.items():
    group_df = plot_data[plot_data['Group'] == group]
    if not group_df.empty:
        ax_main.scatter(
            group_df['Dim1'], group_df['Dim2'],
            facecolor=color, edgecolor=color,
            alpha=0.8, s=50, linewidth=0.4, marker='o'
        )

# Add 68% confidence ellipses for each group
for group, color in sci_colors.items():
    group_df = plot_data[plot_data['Group'] == group]
    if len(group_df) < 2:
        continue  # Skip if too few points

    try:
        mean = group_df[['Dim1', 'Dim2']].mean()
        cov = group_df[['Dim1', 'Dim2']].cov()
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        # 68% confidence ellipse (chi-square with df=2)
        width = 2 * lambda_[0] * np.sqrt(chi2.ppf(0.68, 2))
        height = 2 * lambda_[1] * np.sqrt(chi2.ppf(0.68, 2))
        angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))

        ell = Ellipse(xy=(mean[0], mean[1]), width=width, height=height,
                      angle=angle, edgecolor=color, facecolor='none', linewidth=1)
        ax_main.add_patch(ell)
    except Exception as e:
        print(f"Skipping ellipse for {group} due to error: {e}")

ax_main.set_xlim(x_lim)
ax_main.set_ylim(y_lim)
ax_main.set_xlabel("")
ax_main.set_ylabel("")
ax_main.set_xticks([])
ax_main.set_yticks([])
ax_main.grid(False)

# Set spine thickness
ax_main.spines['top'].set_linewidth(0.5)
ax_main.spines['right'].set_linewidth(0.5)
ax_main.spines['bottom'].set_linewidth(0.5)
ax_main.spines['left'].set_linewidth(0.5)

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=group,
                      markerfacecolor=color, markersize=10)
           for group, color in sci_colors.items()]
ax_main.legend(handles=handles, loc='lower right', frameon=True)

# ========================== Right Marginal Density ==========================
ax_right = fig.add_subplot(gs[1, 1])
for group, color in sci_colors.items():
    group_df = plot_data[plot_data['Group'] == group]
    if not group_df.empty:
        sns.kdeplot(
            data=group_df,
            y='Dim2', fill=True, alpha=0.4, color=color, linewidth=0.5, ax=ax_right
        )

ax_right.set_ylim(y_lim)
ax_right.set_yticks([])
ax_right.set_xticks([])
ax_right.set_xlabel("")
ax_right.set_ylabel("")
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['bottom'].set_visible(False)
ax_right.spines['left'].set_visible(False)

# Adjust layout
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.0, wspace=0.0)

# ========================== Save & Display ==========================
plt.savefig("tsne_wo3.png", dpi=300, bbox_inches='tight')
plt.show()