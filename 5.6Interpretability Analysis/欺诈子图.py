import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

# ==========================================
# 1. 整体画布和网格布局设置
# ==========================================
fig = plt.figure(figsize=(16, 9))
# 左侧占满两行，右侧分上下两行
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], wspace=0.3, hspace=0.3)

ax_normal = fig.add_subplot(gs[:, 0])  # 左侧正常网络
ax_star = fig.add_subplot(gs[0, 1])    # 右上星型网络
ax_ring = fig.add_subplot(gs[1, 1])    # 右下环形网络

# ==========================================
# 2. 左侧：Normal Transaction Network (正常交易)
# ==========================================
ax_normal.set_title("Normal Transaction Network", fontweight='bold', fontsize=16, pad=20)

# 使用 Barabasi-Albert 模型生成更符合真实交易的无标度网络
G_normal_undirected = nx.barabasi_albert_graph(n=35, m=2, seed=42)
G_normal = nx.DiGraph()
# 随机赋予交易方向
import random
random.seed(42)
for u, v in G_normal_undirected.edges():
    if random.random() > 0.5:
        G_normal.add_edge(u, v)
    else:
        G_normal.add_edge(v, u)

# 布局与绘制
pos_normal = nx.spring_layout(G_normal, seed=10)
nx.draw_networkx_nodes(G_normal, pos_normal, ax=ax_normal, node_color='#4A90E2', node_size=120, edgecolors='white', linewidths=1.5)
nx.draw_networkx_edges(G_normal, pos_normal, ax=ax_normal, edge_color='#A0A0A0', alpha=0.6, arrowsize=10, arrowstyle='-|>', node_size=120)

# 底部 GSE 文本
ax_normal.text(0.5, -0.05, "GSE = 2.1 (Normal Baseline)",
               transform=ax_normal.transAxes, ha='center', color='#27AE60', fontweight='bold', fontsize=14)
ax_normal.axis('off')

# ==========================================
# 3. 右上：Star-shaped Gang Fraud (星型 - 资金归集)
# ==========================================
ax_star.set_title("Topology Type: Star (Fund Aggregation)", fontweight='bold', fontsize=14, pad=15)

G_star = nx.DiGraph()
num_spokes = 12
hub = 0
# 真实金融中的资金归集：所有边缘节点向中心节点转账
edges_star = [(i, hub) for i in range(1, num_spokes + 1)]
G_star.add_edges_from(edges_star)

# 强制布局为标准圆形星状
pos_star = {hub: [0, 0]}
import numpy as np
for i in range(1, num_spokes + 1):
    angle = 2 * np.pi * i / num_spokes
    pos_star[i] = [np.cos(angle), np.sin(angle)]

# 颜色区分：中心目标为深红，边缘汇款人为橙色
node_colors_star = ['#D0021B' if node == hub else '#F5A623' for node in G_star.nodes()]
sizes_star = [250 if node == hub else 120 for node in G_star.nodes()]

nx.draw_networkx_nodes(G_star, pos_star, ax=ax_star, node_color=node_colors_star, node_size=sizes_star, edgecolors='white', linewidths=1.5)
nx.draw_networkx_edges(G_star, pos_star, ax=ax_star, edge_color='#555555', alpha=0.8, arrowsize=15, arrowstyle='-|>', node_size=sizes_star)

ax_star.text(0.5, -0.15, "GSE = 4.5 (Anomaly Detected!)",
             transform=ax_star.transAxes, ha='center', color='#D0021B', fontweight='bold', fontsize=13)
ax_star.axis('off')

# ==========================================
# 4. 右下：Ring-shaped Gang Fraud (环形 - 资金清洗分层)
# ==========================================
ax_ring.set_title("Topology Type: Ring (Fund Layering)", fontweight='bold', fontsize=14, pad=15)

G_ring = nx.DiGraph()
num_ring_nodes = 10
# 真实金融中的资金清洗闭环
edges_ring = [(i, (i+1)%num_ring_nodes) for i in range(num_ring_nodes)]
G_ring.add_edges_from(edges_ring)

# 使用圆形布局
pos_ring = nx.circular_layout(G_ring)

nx.draw_networkx_nodes(G_ring, pos_ring, ax=ax_ring, node_color='#F5A623', node_size=150, edgecolors='white', linewidths=1.5)
nx.draw_networkx_edges(G_ring, pos_ring, ax=ax_ring, edge_color='#555555', alpha=0.9, arrowsize=18, arrowstyle='-|>', node_size=150, connectionstyle="arc3,rad=0.1")
# 注：加入弧度(connectionstyle)让环形资金流转更有动态感

ax_ring.text(0.5, -0.15, "GSE = 4.2 (Anomaly Detected!)",
             transform=ax_ring.transAxes, ha='center', color='#D0021B', fontweight='bold', fontsize=13)
ax_ring.axis('off')

# ==========================================
# 5. 中间：添加演化指示箭头
# ==========================================
# 箭头 1: 从正常网络指向星型网络
arrow1 = FancyArrowPatch((0.45, 0.65), (0.55, 0.75),
                         transform=fig.transFigure,
                         color='#2C3E50', arrowstyle='simple, head_length=1.5, head_width=1.5, tail_width=0.4',
                         alpha=0.8, mutation_scale=20)
fig.patches.append(arrow1)

# 箭头 2: 从正常网络指向环形网络
arrow2 = FancyArrowPatch((0.45, 0.35), (0.55, 0.25),
                         transform=fig.transFigure,
                         color='#2C3E50', arrowstyle='simple, head_length=1.5, head_width=1.5, tail_width=0.4',
                         alpha=0.8, mutation_scale=20)
fig.patches.append(arrow2)

# ==========================================
# 6. 显示与保存
# ==========================================
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1)
# 取消注释下一行可以保存为高清 PDF (供论文插入)
# plt.savefig("fraud_topology_evolution.pdf", dpi=300, bbox_inches='tight')
plt.show()