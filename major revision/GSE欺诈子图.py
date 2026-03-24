import matplotlib.pyplot as plt
import networkx as nx

# 设置整体画布和子图 (比例调整以匹配原图的长宽比)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.1)

# ==========================================
# 左图：Normal Transaction Network (正常交易网络)
# ==========================================
ax1 = axes[0]
ax1.set_title("Normal Transaction Network", fontweight='bold', fontsize=14, pad=20)

# 手动精确定义左图节点坐标，以完全还原图中的分散且均匀的拓扑结构
pos_normal = {
    0: (2.5, 8.0), 1: (5.0, 9.0), 2: (6.5, 9.5), 3: (8.5, 9.0),
    4: (4.5, 7.0), 5: (6.5, 7.5), 6: (7.5, 7.5), 7: (10.0, 6.5),
    8: (3.5, 5.0), 9: (4.5, 5.5), 10: (6.0, 5.5), 11: (8.0, 5.0),
    12: (1.0, 4.0), 13: (2.5, 4.5), 14: (4.5, 4.0), 15: (6.5, 4.5),
    16: (8.5, 4.0), 17: (10.0, 4.5), 18: (3.5, 2.0), 19: (4.5, 2.5),
    20: (6.5, 2.0), 21: (8.5, 2.5),
    # 原图左下方的两个孤立节点
    22: (1.0, 1.5), 23: (5.0, 0.0)
}

# 根据原图精确连接网状边
edges_normal = [
    (0,1), (0,8), (1,2), (1,4), (1,9), (2,3), (2,6), (3,7),
    (4,5), (4,9), (4,10), (5,6), (5,10), (6,11), (7,11), (7,17),
    (8,9), (8,12), (8,13), (9,10), (9,14), (10,11), (10,14), (10,15),
    (11,15), (11,16), (12,13), (13,14), (13,18), (14,15), (14,18), (14,19),
    (15,16), (15,19), (15,20), (16,17), (16,20), (16,21), (17,21),
    (18,19), (19,20), (20,21)
]

G_normal = nx.Graph()
G_normal.add_nodes_from(pos_normal.keys())
G_normal.add_edges_from(edges_normal)

# 绘制左图：蓝色节点，浅灰色网状细边
nx.draw_networkx_nodes(G_normal, pos_normal, ax=ax1, node_color='#4186C6', node_size=150, edgecolors='white', linewidths=0.5)
nx.draw_networkx_edges(G_normal, pos_normal, ax=ax1, edge_color='#CCCCCC', width=1.0)

# 左图底部 GSE 文本 (绿色加粗)
ax1.text(0.5, -0.1, "GSE = 2.1 (Normal Baseline)",
         transform=ax1.transAxes, ha='center', color='#177A23', fontweight='bold', fontsize=12)
ax1.axis('off')

# ==========================================
# 右图：Gang Fraud Network (Detected) (团伙欺诈网络)
# ==========================================
ax2 = axes[1]
ax2.set_title("Gang Fraud Network (Detected)", fontweight='bold', fontsize=14, pad=20)

# 手动精确定义右图节点坐标（星型+局部聚集）
pos_fraud = {
    'c': (5, 5),          # 中心核心红点
    't1': (2.5, 8.5),     # 上方3个辐射点
    't2': (5, 9.5),
    't3': (7, 9),
    'r1': (8.5, 6.5),     # 右侧菱形环路
    'r2': (10, 6),
    'r3': (8.5, 4.5),
    'br': (7.5, 2.5),     # 右下辐射点
    'lc1': (4.5, 2.5),    # 左下角的高密聚集团伙
    'lc2': (3.5, 1),
    'lc3': (1.5, 2.5),
    'lc4': (1, 4),
    'lc5': (2.5, 4.5)
}

# 根据原图精确连接星型发散边与团伙内聚边
edges_fraud = [
    # 中心点向外的辐射连线
    ('c', 't1'), ('c', 't2'), ('c', 't3'),
    ('c', 'r1'), ('c', 'r3'), ('c', 'br'),
    ('c', 'lc1'), ('c', 'lc2'), ('c', 'lc3'), ('c', 'lc4'), ('c', 'lc5'),
    # 右侧的小闭环
    ('r1', 'r2'), ('r3', 'r2'),
    # 左侧团伙内部的紧密连线
    ('lc4', 'lc5'), ('lc4', 'lc3'),
    ('lc5', 'lc1'), ('lc3', 'lc2'), ('lc3', 'lc1'), ('lc2', 'lc1')
]

G_fraud = nx.Graph()
G_fraud.add_nodes_from(pos_fraud.keys())
G_fraud.add_edges_from(edges_fraud)

# 设置颜色区分：中心节点为红色，边缘节点为橘色
node_colors = ['#C8282D' if node == 'c' else '#F78F21' for node in G_fraud.nodes()]

# 绘制右图：红/橘节点，深灰色清晰边
nx.draw_networkx_nodes(G_fraud, pos_fraud, ax=ax2, node_color=node_colors, node_size=150, edgecolors='white', linewidths=0.5)
nx.draw_networkx_edges(G_fraud, pos_fraud, ax=ax2, edge_color='#555555', width=1.2)

# 右图底部 GSE 文本 (红色加粗)
ax2.text(0.5, -0.1, "GSE = 4.5 (Anomaly Detected!)",
         transform=ax2.transAxes, ha='center', color='#C8282D', fontweight='bold', fontsize=12)
ax2.axis('off')

# 展示并保持布局紧凑
plt.tight_layout()
plt.show()