import matplotlib.pyplot as plt

# --- Data extraction ---
methods = [
    'GraphSAGE',
    'TGN',
    'SCALE(Ours)',
    'AHGNN',
    'CAFD',
    'SE-GNN'
]
latency = [9.8, 12.2, 14.5, 28.4, 35.6, 42.1]

# --- Plotting configuration ---
# Standard academic font family
plt.rcParams['font.family'] = 'serif'

plt.figure(figsize=(9, 6))

# 设置颜色：我们的方法 SCALE (第3个) 设定为经典的学术红，其他为学术常用色
colors = [
    # '#1f77b4',  # 蓝色 (GraphSAGE)
    # '#ff7f0e',  # 橙色 (TGN)
    # '#d62728',  # 红色 (SCALE - 我们的方法)
    # '#2ca02c',  # 绿色 (AHGNN)
    # '#9467bd',  # 紫色 (CAFD)
    # '#8c564b'   # 棕色 (SE-GNN)

    '#FF6B6B',  # (GraphSAGE)
    '#d62728',  # (TGN)
    '#FFEAA7',  # (SCALE - 我们的方法)
    '#4ECDC4',  # (AHGNN)
    '#45B7D1',  # (CAFD)
    '#1f77b4'   # (SE-GNN)
]

# Create bar chart
bars = plt.bar(methods, latency, color=colors, edgecolor='black', width=0.6, zorder=3)

# Add data labels directly on top of each bar for precise reading
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
             f'{yval}', ha='center', va='bottom', fontsize=11)

# --- Labeling and formatting ---
# 【修改点1】: 标题上方黑色加粗
plt.title('Inference Latency Comparison',
          fontsize=14, pad=15, fontweight='bold', color='black')

# 【修改点2】: 坐标轴取消加粗
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Latency (ms/batch)', fontsize=12)

# X轴标签水平放置
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)

# Add a subtle y-axis grid to help read values
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Adjust layout so labels don't get cut off
plt.tight_layout()

# Save the figure as a high-quality PDF for your paper (optional)
# plt.savefig('latency_comparison_final.pdf', format='pdf', bbox_inches='tight')

# Display the plot
plt.show()