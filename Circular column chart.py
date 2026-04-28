import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

# 全局绘图参数，调整为8组（布局数量）
CONFIG = {
    'num_groups': 8,
    'inner_r': 0.3,
    'outer_r': 1.0,
    'gap_deg': 8.0,
    'pad_side_deg': 0.5,
    'start_offset': 90,
    'bar_colors': ['#DCEDF7', '#C0DEF6', '#AAD6F0', '#73C6EC'], # 对应4个指标：Accuracy, Kappa, Recall, F1-Score
    'seg_colors': ['#F9F5E8', '#FCF2C6', '#FCD435', '#FCA510', '#FB8500', '#FB6107', '#F94D29', '#F92929'], # 8种扇区背景颜色
    'show_sig': False, # 无显著性数据，关闭
    'error_cap_deg': 1.5
}

plt.rcParams.update({
    'font.size': 10,
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

# 替换为用户数据：每个元组代表一个布局（相当于原代码的“指标”）
# (布局名称, 4个指标值 [Acc, Kappa, Rec, F1], 误差 [全0，无数据], 显著性 [空，无数据], min_val, max_val)
RAW_DATA = [
    ("Physical Montage(PM)", [0.8480, 0.7973, 0.8480, 0.8469], [0.0936, 0.1248, 0.0936, 0.0953], ['', '', '', ''],
     0.5694, 0.9757),
    ("random", ),
    ("PCC", ),
    ("PLV", ),
    ("COH", ),
    ("PM-PCC", ),
    ("PM-PLV", ),
    ("PM-COH", ),
]

# 解包数据
metrics = [item[0] for item in RAW_DATA] # 这里metrics变为布局名称
means_list = [item[1] for item in RAW_DATA]
errors_list = [item[2] for item in RAW_DATA]
sig_letters = [item[3] for item in RAW_DATA]
min_vals = [item[4] for item in RAW_DATA]
max_vals = [item[5] for item in RAW_DATA]


def compute_segment_layout(n_segs, gap_deg, start_ang):
    total_gap = n_segs * gap_deg
    if total_gap >= 360:
        raise ValueError("间隙角度总和不能超过360度")
    seg_span = (360.0 - total_gap) / n_segs

    centers = []
    current = start_ang
    for _ in range(n_segs):
        centers.append(current)
        current -= (seg_span + gap_deg) # 顺时针方向排列

    starts = [c - seg_span / 2 for c in centers]
    ends = [c + seg_span / 2 for c in centers]
    return centers, starts, ends, seg_span


# 调用并赋值
centers, seg_starts, seg_ends, seg_span = compute_segment_layout(
    CONFIG['num_groups'], CONFIG['gap_deg'], CONFIG['start_offset']
)


def allocate_bar_angles(seg_deg, pad_deg, n_bars, gap_factor=0.1):
    usable = seg_deg - 2 * pad_deg
    if usable <= 0:
        raise ValueError("扇区太窄，无法容纳条形")
    total_units = n_bars + (n_bars - 1) * gap_factor
    unit_width = usable / total_units
    return unit_width, unit_width * gap_factor # 条形宽, 间隙角


# 计算条形宽度和间隙
bar_width, intra_gap = allocate_bar_angles(
    seg_span, CONFIG['pad_side_deg'], len(CONFIG['bar_colors'])
)

# 计算所有条形的起始角度（修复NameError：使用seg_starts）
bar_positions = [] # 每个扇区的条形中心位置列表（嵌套）
for seg_start in seg_starts:
    seg_bar_centers = []
    pos = seg_start + CONFIG['pad_side_deg']
    for _ in range(len(CONFIG['bar_colors'])):
        seg_bar_centers.append(pos + bar_width / 2) # 使用中心角度
        pos += bar_width + intra_gap
    bar_positions.append(seg_bar_centers)


def scale_value_to_radius(value, min_val, max_val, inner_r, outer_r):
    """将值线性映射到内径-外径"""
    if min_val == max_val:
        return outer_r
    normalized = (value - min_val) / (max_val - min_val)
    return inner_r + normalized * (outer_r - inner_r)


def draw_radial_bar(ax, center_deg, width_deg, radius, color, inner_r):
    """绘制一个径向楔形条形"""
    wedge = Wedge(
        center=(0, 0),
        r=radius,
        theta1=center_deg - width_deg / 2,
        theta2=center_deg + width_deg / 2,
        width=radius - inner_r,
        facecolor=color,
        edgecolor='black',
        linewidth=0.5,
        zorder=2
    )
    ax.add_patch(wedge)


def draw_error_caps(ax, angle_deg, r_low, r_high, cap_size_deg):
    """绘制误差线两端的横帽（简化版，使用线段模拟）"""
    if r_low == r_high: # 无误差，跳过
        return
    rad = np.radians(angle_deg)
    cap_half = np.radians(cap_size_deg / 2)

    # 下帽
    x1_low = r_low * np.cos(rad - cap_half)
    y1_low = r_low * np.sin(rad - cap_half)
    x2_low = r_low * np.cos(rad + cap_half)
    y2_low = r_low * np.sin(rad + cap_half)
    ax.plot([x1_low, x2_low], [y1_low, y2_low], color='black', linewidth=1, zorder=3)

    # 上帽
    x1_high = r_high * np.cos(rad - cap_half)
    y1_high = r_high * np.sin(rad - cap_half)
    x2_high = r_high * np.cos(rad + cap_half)
    y2_high = r_high * np.sin(rad + cap_half)
    ax.plot([x1_high, x2_high], [y1_high, y2_high], color='black', linewidth=1, zorder=3)

    # 垂直误差线
    x_vert = r_low * np.cos(rad) # 中点，但实际从low到high
    y_vert = r_low * np.sin(rad)
    dx_vert = (r_high - r_low) * np.cos(rad)
    dy_vert = (r_high - r_low) * np.sin(rad)
    ax.plot([x_vert, x_vert + dx_vert], [y_vert, y_vert + dy_vert], color='black', linewidth=1, zorder=3)


fig = plt.figure(figsize=(12, 12))  # 增大画布尺寸以减少重叠
ax = fig.add_subplot(111, polar=True) # 使用极坐标轴
ax.set_theta_offset(np.deg2rad(90)) # 起始于顶部
ax.set_theta_direction(-1) # 顺时针
ax.set_rlim(0, CONFIG['outer_r'])
ax.set_rticks([]) # 隐藏径向刻度
ax.grid(False) # 隐藏网格
ax.spines['polar'].set_visible(False) # 隐藏边框

# 遍历每个布局（原“指标”）
for i in range(CONFIG['num_groups']):
    metric_name = metrics[i]
    means = means_list[i]
    errors = errors_list[i]
    sigs = sig_letters[i]
    min_val = min_vals[i]
    max_val = max_vals[i]

    # 绘制扇区背景
    bg_color = CONFIG['seg_colors'][i % len(CONFIG['seg_colors'])]
    bg_wedge = Wedge(
        (0, 0), CONFIG['outer_r'], seg_starts[i], seg_ends[i],
        width=CONFIG['outer_r'] - CONFIG['inner_r'],
        facecolor=bg_color, alpha=0.3, zorder=1
    )
    ax.add_patch(bg_wedge)

    # 添加布局名称标签（在扇区中心外，修改旋转为0以使标签水平可读，避免倒置）
    label_angle = np.deg2rad(centers[i])
    label_r = CONFIG['outer_r'] + 0.08  # 稍增大标签距离以避免重叠
    ax.text(label_angle, label_r, metric_name, ha='center', va='center', rotation=0, fontsize=10)  # 将rotation改为0，使所有标签水平

    # 绘制每个指标的条形（4个）
    seg_bar_centers = bar_positions[i]
    for j, (m, e, sig) in enumerate(zip(means, errors, sigs)):
        radius = scale_value_to_radius(m, min_val, max_val, CONFIG['inner_r'], CONFIG['outer_r'])
        r_low = max(scale_value_to_radius(m - e, min_val, max_val, CONFIG['inner_r'], CONFIG['outer_r']), CONFIG['inner_r'])  # 防止r_low低于inner_r
        r_high = min(scale_value_to_radius(m + e, min_val, max_val, CONFIG['inner_r'], CONFIG['outer_r']), CONFIG['outer_r'])  # 防止r_high超过outer_r

        color = CONFIG['bar_colors'][j]
        center_deg = seg_bar_centers[j]

        draw_radial_bar(ax, center_deg, bar_width, radius, color, CONFIG['inner_r'])
        draw_error_caps(ax, center_deg, r_low, r_high, CONFIG['error_cap_deg'])

        # 如果show_sig，添加显著性字母（在条形上方）
        if CONFIG['show_sig'] and sig:
            sig_r = r_high + 0.02
            ax.text(np.deg2rad(center_deg), sig_r, sig, ha='center', va='bottom', fontsize=8)

# 添加图例（4个指标）
legend_labels = ['Accuracy', 'Kappa', 'Recall', 'F1-Score']
legend_patches = [plt.Rectangle((0, 0), 1, 1, color=CONFIG['bar_colors'][i]) for i in range(4)]
ax.legend(legend_patches, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1.15), title='Metrics')  # 调整图例位置

plt.tight_layout()  # 自动调整布局以减少边缘裁切

# 保存结果（可选，如果需要文件输出）
# plt.savefig('circular_chart.png', dpi=300, bbox_inches='tight')

plt.show()