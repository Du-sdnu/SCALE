import matplotlib.pyplot as plt
import numpy as np

# --- 设置绘图风格 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# --- 数据定义 (保持不变) ---
data = {
    'Embedding Size ($d$)': (['32', '64', '128', '256'], [0.885, 0.902, 0.915, 0.908], [0.852, 0.875, 0.892, 0.881]),
    'VIB Weight ($\\beta$)': (
    ['1e-4', '1e-3', '1e-2', '1e-1'], [0.890, 0.915, 0.905, 0.875], [0.865, 0.892, 0.878, 0.840]),
    'GNN Layers ($L$)': (['1', '2', '3', '4'], [0.882, 0.915, 0.895, 0.865], [0.850, 0.892, 0.868, 0.835]),
    'Compactness ($\\alpha$)': (
    ['0.1', '0.5', '1.0', '2.0'], [0.888, 0.905, 0.915, 0.900], [0.860, 0.882, 0.892, 0.875]),
    'Scale Factor ($\\tau_s$)': (
    ['0.1', '1.0', '5.0', '10.0'], [0.892, 0.915, 0.908, 0.895], [0.870, 0.892, 0.885, 0.868]),
    'Threshold ($q_{known}$)': (
    ['0.85', '0.90', '0.95', '0.99'], [0.875, 0.895, 0.915, 0.885], [0.845, 0.872, 0.892, 0.855]),
    'GSE Coeff. ($\\gamma$)': (['1', '2', '3', '4'], [0.865, 0.890, 0.915, 0.902], [0.830, 0.865, 0.892, 0.878]),
    'EMA Momentum ($\\eta$)': (
    ['0.9', '0.99', '0.999', '4-9s'], [0.880, 0.900, 0.915, 0.908], [0.855, 0.878, 0.892, 0.885]),
    'Init. Min ($N_{min}$)': (['5', '10', '20', '50'], [0.885, 0.902, 0.915, 0.895], [0.860, 0.880, 0.892, 0.865])
}

# --- 绘图逻辑 ---
bar_width = 0.35
opacity = 0.9
patterns = ['//', '..']
colors = ['#1f77b4', '#ff7f0e']

for i, (param_name, (labels, auc_scores, f1_scores)) in enumerate(data.items()):
    # 创建独立画布
    fig, ax = plt.subplots(figsize=(6, 5))

    # [修改点 1] 调整 bottom 边距
    # 之前是 0.2，因为我们把文字上移了，稍微减小一点留白，让图更紧凑
    plt.subplots_adjust(bottom=0.18, top=0.92, left=0.15, right=0.95)

    x = np.arange(len(labels))

    # 绘制柱状图
    rects1 = ax.bar(x - bar_width / 2, auc_scores, bar_width, alpha=opacity, color=colors[0],
                    hatch=patterns[0], edgecolor='black', label='AUC-ROC')
    rects2 = ax.bar(x + bar_width / 2, f1_scores, bar_width, alpha=opacity, color=colors[1],
                    hatch=patterns[1], edgecolor='black', label='Macro-F1')

    # 设置标签与标题
    ax.set_xlabel(param_name, fontweight='bold')
    ax.set_ylabel('Performance')
    ax.set_title(f'Effect of {param_name}')

    # [修改点 2] 向上移动小标位置
    # -0.22 -> -0.16 (数字越接近0，位置越靠上)
    # 这个位置应该刚好贴在 x 轴 label 下方
    ax.text(0.5, -0.16, f'({chr(97 + i)})', transform=ax.transAxes,
            ha='center', va='top', fontweight='normal', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 设置 Y 轴范围 (顶部留白给图例)
    min_val = min(min(auc_scores), min(f1_scores)) - 0.02
    max_val = max(max(auc_scores), max(f1_scores)) + 0.04
    ax.set_ylim(min_val, max_val)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 独立图例
    ax.legend(loc='upper left', frameon=True)

    # 自动保存为9个文件 (可选)
    # plt.savefig(f"param_plot_{i}.pdf", bbox_inches='tight')

    plt.show()