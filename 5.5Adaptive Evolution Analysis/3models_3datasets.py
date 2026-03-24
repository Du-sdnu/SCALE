import matplotlib.pyplot as plt
import numpy as np

# --- 全局绘图风格设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# --- 数据生成函数 (保持不变) ---
def get_evolution_data():
    snapshots = ['Snap. 1', 'Snap. 2', 'Snap. 3', 'Snap. 4', 'Snap. 5']
    x = np.arange(len(snapshots))

    # --- Macro-F1 Data ---
    f1_ieee = {
        'Static': [0.78, 0.77, 0.76, 0.75, 0.74],
        'Fine-tuning': [0.78, 0.79, 0.76, 0.78, 0.75],
        'SCALE': [0.79, 0.80, 0.805, 0.81, 0.815]
    }
    f1_eth = {
        'Static': [0.88, 0.86, 0.84, 0.82, 0.80],
        'Fine-tuning': [0.88, 0.89, 0.88, 0.88, 0.89],
        'SCALE': [0.88, 0.895, 0.90, 0.905, 0.91]
    }
    f1_dgraph = {
        'Static': [0.85, 0.80, 0.75, 0.70, 0.68],
        'Fine-tuning': [0.85, 0.87, 0.83, 0.86, 0.84],
        'SCALE': [0.86, 0.87, 0.88, 0.885, 0.895]
    }

    # --- AUC-ROC Data ---
    auc_ieee = {
        'Static': [0.88, 0.87, 0.86, 0.85, 0.84],
        'Fine-tuning': [0.88, 0.89, 0.87, 0.88, 0.86],
        'SCALE': [0.89, 0.90, 0.91, 0.915, 0.92]
    }
    auc_eth = {
        'Static': [0.92, 0.90, 0.88, 0.86, 0.84],
        'Fine-tuning': [0.92, 0.93, 0.92, 0.93, 0.93],
        'SCALE': [0.92, 0.93, 0.94, 0.945, 0.95]
    }
    auc_dgraph = {
        'Static': [0.89, 0.85, 0.80, 0.76, 0.74],
        'Fine-tuning': [0.89, 0.90, 0.86, 0.89, 0.87],
        'SCALE': [0.90, 0.905, 0.91, 0.915, 0.92]
    }

    return snapshots, x, {
        'AUC-ROC': {'IEEE-CIS': auc_ieee, 'Ethereum': auc_eth, 'DGraph-Fin': auc_dgraph},
        'Macro-F1': {'IEEE-CIS': f1_ieee, 'Ethereum': f1_eth, 'DGraph-Fin': f1_dgraph}
    }


# --- 绘图逻辑 ---
snapshots, x, all_data = get_evolution_data()
metrics = ['AUC-ROC', 'Macro-F1']
datasets = ['IEEE-CIS', 'Ethereum', 'DGraph-Fin']

# 样式定义
styles = {
    'Static': {'color': '#7f7f7f', 'marker': 's', 'linestyle': '--', 'label': 'Static', 'markersize': 6},
    'Fine-tuning': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-.', 'label': 'Fine-tuning', 'markersize': 6},
    'SCALE': {'color': '#d62728', 'marker': 'o', 'linestyle': '-', 'label': 'SCALE (Ours)', 'linewidth': 2.5,
              'markersize': 7}
}

# 计数器，用于生成 (a)-(f)
plot_counter = 0

for metric in metrics:
    for dataset in datasets:
        # 创建独立画布
        fig, ax = plt.subplots(figsize=(6, 5))
        # 调整布局，防止底部标题被遮挡
        plt.subplots_adjust(bottom=0.2, top=0.92, left=0.15, right=0.95)

        data_dict = all_data[metric][dataset]

        # 绘制三条线
        ax.plot(x, data_dict['Static'], **styles['Static'])
        ax.plot(x, data_dict['Fine-tuning'], **styles['Fine-tuning'])
        ax.plot(x, data_dict['SCALE'], **styles['SCALE'])

        # 设置标题 (位于下方)
        title_text = f'({chr(97 + plot_counter)}) {metric} on {dataset}'
        ax.set_title(title_text, y=-0.25, fontsize=13, fontweight='normal')

        # 设置标签
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Time Snapshots', fontweight='bold', fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels(snapshots)

        # 网格
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Y轴范围设置 (确保左下角有足够空间放图例)
        if metric == 'AUC-ROC':
            ax.set_ylim(0.72, 0.98)
        else:
            ax.set_ylim(0.65, 0.95)

        # 【核心修改】将图例统一固定在左下角
        # loc='lower left': 强制放置在左下角
        # framealpha=0.9: 增加图例背景不透明度，防止透过图例看到网格线
        ax.legend(loc='lower left', frameon=True, fontsize=10, framealpha=0.9)

        # 保存或显示
        # plt.savefig(f'{metric}_{dataset}.pdf', bbox_inches='tight')
        plt.show()

        plot_counter += 1