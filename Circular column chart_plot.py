import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors as mcolors
from matplotlib.patches import Wedge

# =========================================================
# 莫兰蒂配色方案
# 这里小编自己设置了几种配色，大家可以直接选择，或者可以自己设置
# =========================================================
theme_name = "morandi_warm"  # 可改为 morandi_cool / morandi_gray / morandi_soft
morandi_palettes = {
    "morandi_warm": {
        "colors": {
            "T1": "#C97A7A", "T2": "#D8A7A7", "T3": "#8FB3C9",
            "T4": "#9DB7A0", "T5": "#D1C089", "T6": "#7F9A8C"
        },
        "group_colors": {
            "Group1": "#F2ECE6", "Group2": "#E7DFC9", "Group3": "#EEE6EB",
            "Group4": "#E6D1DA", "Group5": "#DED7EB"
        }
    },
    "morandi_cool": {
        "colors": {
            "T1": "#8C9AAE", "T2": "#A5B1C2", "T3": "#7FA6A3",
            "T4": "#8FB0A9", "T5": "#B4B8A8", "T6": "#6F8F8B"
        },
        "group_colors": {
            "Group1": "#EDF1F4", "Group2": "#E3E9EE", "Group3": "#E9EFF0",
            "Group4": "#E1E8E6", "Group5": "#E6E9E3"
        }
    },
    "morandi_gray": {
        "colors": {
            "T1": "#9A8F8F", "T2": "#B0A7A7", "T3": "#8F9AA1",
            "T4": "#9AA39A", "T5": "#B2AA97", "T6": "#7F8A86"
        },
        "group_colors": {
            "Group1": "#F0EEEE", "Group2": "#E6E3E3", "Group3": "#ECEAEA",
            "Group4": "#E3E1E1", "Group5": "#E8E6E6"
        }
    },
    "morandi_soft": {
        "colors": {
            "T1": "#D5A6A6", "T2": "#E1C2C2", "T3": "#B7CFDD",
            "T4": "#C1D6C7", "T5": "#E3D9B5", "T6": "#B5CBC2"
        },
        "group_colors": {
            "Group1": "#F7F4EF", "Group2": "#F3EEDD", "Group3": "#F5EFF3",
            "Group4": "#F1E4EA", "Group5": "#EFEAF5"
        }
    }
}
pal = morandi_palettes[theme_name]
orig_colors = pal["colors"]
orig_group_colors = pal["group_colors"]

# =========================================================
# ======================= 绘图函数 ========================
# =========================================================
def draw_circos(data, metric_colors, group_colors, P_label, file_path, type_="pdf", width=10, height=10, res=300):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.deg2rad(90))
    ax.set_theta_direction(-1)

    inner_r = 0.0  # 可以调整为0.3以创建内孔
    outer_r = 1.0
    track_margin = 0.12
    label_height = 0.035
    r_label_min = outer_r + track_margin
    r_label_max = r_label_min + label_height
    ax.set_rlim(0, r_label_max + 0.1)
    ax.set_rticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    groups = data["Group"].unique()
    N = len(groups)
    centers, seg_starts, seg_ends, seg_span_deg = compute_segment_layout(N, gap_deg=7, start_ang=90)

    for i, group in enumerate(groups):
        theta_start = seg_starts[i]
        theta_end = seg_ends[i]
        center_deg = centers[i]

        # 第二轨道背景（柱状图，alpha=0.4）
        bg_color = mcolors.to_rgba(group_colors[group], alpha=0.4)
        bg_wedge = Wedge((0, 0), outer_r, theta_start, theta_end, width=outer_r - inner_r, facecolor=bg_color, zorder=1)
        ax.add_patch(bg_wedge)

        # 第一轨道背景（Group标签）
        label_bg_color = group_colors[group]
        label_wedge = Wedge((0, 0), r_label_max, theta_start, theta_end, width=label_height, facecolor=label_bg_color, zorder=1)
        ax.add_patch(label_wedge)

        # Group标签文本
        label_text = f"{group} {P_label.get(group, '')}"
        label_r = (r_label_min + r_label_max) / 2
        tangent_text(ax, np.deg2rad(center_deg), label_r, label_text, fontsize=15, fontweight="normal", color="black")

        # 获取当前组数据
        d = data[data["Group"] == group]
        M = len(d)
        max_value = np.ceil(d["mean"].max() * 1.3)

        # 网格线
        at = np.linspace(0, max_value, 5)[1:-1]
        for a in at:
            r_a = inner_r + (a / max_value) * (outer_r - inner_r)
            theta_arc = np.linspace(theta_start, theta_end, 100)
            r_line = np.full(100, r_a)
            ax.plot(np.deg2rad(theta_arc), r_line, ls="--", color="grey", lw=1)
            # 网格标签
            text_theta = theta_start - 2  # 近似 mm_h(2)
            tangent_text(ax, np.deg2rad(text_theta), r_a, f"{a:.1f}", fontsize=8, color="black")

        # 柱状图
        bar_width_x = 0.10
        gap_width_x = 0.05
        for m in range(M):
            i = m + 1  # 1-based for R-like calculation
            xleft = (i + 2) * (bar_width_x + gap_width_x) - (M - 1) * (bar_width_x + gap_width_x) / 2
            xright = xleft + bar_width_x
            xc = (xleft + xright) / 2

            theta_left = theta_start + xleft * seg_span_deg
            theta_right = theta_start + xright * seg_span_deg
            theta_c = theta_start + xc * seg_span_deg

            val = d.iloc[m]["mean"]
            se = d.iloc[m]["se"]
            lable = d.iloc[m]["lable"]
            color = metric_colors[d.iloc[m]["T"]]

            val_norm = inner_r + (val / max_value) * (outer_r - inner_r)
            high_norm = inner_r + ((val + se) / max_value) * (outer_r - inner_r)

            # 柱子
            wedge = Wedge((0, 0), val_norm, theta_left, theta_right, width=val_norm - inner_r, facecolor=color, edgecolor="black", lw=1.5, zorder=2)
            ax.add_patch(wedge)

            # 误差线
            ax.plot(np.deg2rad([theta_c, theta_c]), [val_norm, high_norm], color="black", lw=1.5, zorder=3)

            # 上帽
            delta_x = 0.03
            delta_theta = delta_x * seg_span_deg
            theta_cap_left = theta_c - delta_theta
            theta_cap_right = theta_c + delta_theta
            ax.plot(np.deg2rad([theta_cap_left, theta_cap_right]), [high_norm, high_norm], color="black", lw=1.5, zorder=3)

            # 显著性标记
            sig_r = high_norm + 0.05 * (outer_r - inner_r)
            ax.text(np.deg2rad(theta_c), sig_r, lable, ha="center", va="bottom", fontsize=10)

    # 图例
    metrics = list(metric_colors.keys())
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=metric_colors[m]) for m in metrics]
    ax.legend(legend_patches, metrics, loc="upper right", bbox_to_anchor=(1.1, 1.1), title="Metrics", frameon=False)

    if type_ == "pdf":
        plt.savefig(file_path, format="pdf")
    elif type_ == "png":
        plt.savefig(file_path, dpi=res)
    plt.show()  # 展示图表
    plt.close()

def compute_segment_layout(n_segs, gap_deg, start_ang):
    total_gap = n_segs * gap_deg
    if total_gap >= 360:
        raise ValueError("间隙角度总和不能超过360度")
    seg_span_deg = (360.0 - total_gap) / n_segs

    centers = []
    current = start_ang
    for _ in range(n_segs):
        centers.append(current)
        current -= (seg_span_deg + gap_deg)

    starts = [c - seg_span_deg / 2 for c in centers]
    ends = [c + seg_span_deg / 2 for c in centers]
    return centers, starts, ends, seg_span_deg

def tangent_text(ax, theta, r, text, fontsize=10, fontweight="normal", color="#111111"):
    display_theta = ax.get_theta_direction() * theta + ax.get_theta_offset()
    angle_deg = (np.degrees(display_theta) + 360) % 360
    rotation = angle_deg - 90
    rotation = (rotation + 180) % 360 - 180

    if rotation < -90 or rotation > 90:
        rotation += 180
        rotation = (rotation + 180) % 360 - 180

    ax.text(theta, r, text, ha="center", va="center",
            rotation=rotation, rotation_mode="anchor",
            fontsize=fontsize, fontweight=fontweight, color=color, zorder=30)

# 用户数据准备
layouts = [
    "Physical Montage(PM)", "random", "PCC", "PLV",
    "COH", "PM-PCC", "PM-PLV", "PM-COH"
]
metrics = ["Accuracy", "Kappa", "Recall", "F1-Score"]
means_list = [

]
errors_list = [

]
sig_letters = [
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', ''],
    ['', '', '', '']
]

data = pd.DataFrame({
    "Group": np.repeat(layouts, len(metrics)),
    "T": np.tile(metrics, len(layouts)),
    "mean": np.ravel(means_list),
    "se": np.ravel(errors_list),
    "lable": np.ravel(sig_letters)
})

# 适应配色（取前4个颜色给指标，前5个group_colors循环给8个组）
metric_colors = dict(zip(metrics, list(orig_colors.values())[:len(metrics)]))
group_list = list(orig_group_colors.values())
group_colors = dict(zip(layouts, group_list * ((len(layouts) // len(group_list)) + 1)))

# P_label（假设为空，可自定义）
P_label = {g: '' for g in layouts}

# 调用函数（type_="png" 或 "pdf"，file_path指定保存路径）
draw_circos(data, metric_colors, group_colors, P_label, file_path="circos_chart.png", type_="png", width=10, height=10, res=300)