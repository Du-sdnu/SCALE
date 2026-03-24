import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch

# --- 设置绘图风格 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ==========================================
# 代码部分 1: 生成 Figure 8(a) - Macro-level Interpretation (GSE)
# ==========================================
def draw_figure_8a():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 左图：正常交易网络 (Normal Transaction Network)
    # 使用随机图模拟稀疏、无明显结构的正常交易
    ax1 = axes[0]
    G_normal = nx.erdos_renyi_graph(n=25, p=0.12, seed=42)
    pos_normal = nx.spring_layout(G_normal, seed=42)

    nx.draw_networkx_nodes(G_normal, pos_normal, ax=ax1, node_size=100, node_color='#1f77b4', alpha=0.8)
    nx.draw_networkx_edges(G_normal, pos_normal, ax=ax1, width=1.0, alpha=0.5, edge_color='gray')

    ax1.set_title("Normal Transaction Network", fontweight='bold', fontsize=14)
    # 在底部添加 GSE 标签
    ax1.text(0.5, -0.1, "GSE = 2.1 (Normal Baseline)", transform=ax1.transAxes,
             ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.axis('off')

    # 2. 右图：团伙欺诈网络 (Gang Fraud Network)
    # 使用星型图 + 环形图模拟高度协同的团伙结构
    ax2 = axes[1]
    G_fraud = nx.star_graph(10)  # 核心星型结构
    # 添加一些外围节点形成环状或复杂连接
    nx.add_cycle(G_fraud, [1, 2, 3, 4, 5])
    nx.add_path(G_fraud, [10, 11, 12, 0])  # 连接到核心

    pos_fraud = nx.spring_layout(G_fraud, seed=24)

    # 核心节点标红
    node_colors = ['#d62728' if i == 0 else '#ff7f0e' for i in G_fraud.nodes()]

    nx.draw_networkx_nodes(G_fraud, pos_fraud, ax=ax2, node_size=120, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G_fraud, pos_fraud, ax=ax2, width=1.5, alpha=0.7, edge_color='black')

    ax2.set_title("Gang Fraud Network (Detected)", fontweight='bold', fontsize=14)
    # 在底部添加 GSE 标签
    ax2.text(0.5, -0.1, "GSE = 4.5 (Anomaly Detected!)", transform=ax2.transAxes,
             ha='center', fontsize=12, color='#d62728', fontweight='bold')
    ax2.axis('off')

    # 调整布局并显示标题
    plt.suptitle("(a) Macro-level Interpretation: Graph Structure Entropy (GSE)", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# 代码部分 2: 生成 Figure 8(b) - Micro-level Interpretation
# ==========================================
def draw_figure_8b():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)

    # --- 左子图：注意力权重可视化 (Attention Weights) ---
    ax1 = axes[0]
    paths = ['User-Trans-Merch\n(Transaction Pattern)', 'User-Device-User\n(Device Sharing)',
             'User-Attribute\n(Basic Info)']
    weights = [0.415, 0.382, 0.203]
    colors = ['#d62728', '#ff7f0e', '#1f77b4']  # 红(高危), 橙(中危), 蓝(基础)

    bars = ax1.bar(paths, weights, color=colors, alpha=0.85, edgecolor='black', width=0.6)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylim(0, 0.5)
    ax1.set_ylabel('Attention Weight',fontsize=12)
    ax1.set_title('Step 1: Meta-path Attention Analysis', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加说明文字
    ax1.text(0.5, 0.85, "High attention on\nTransaction & Device\nimplies 'Cash-out' risk",
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    # --- 右子图：潜在空间原型匹配 (Latent Space Prototype Matching) ---
    ax2 = axes[1]

    # 1. 生成模拟数据点
    np.random.seed(42)
    # 正常行为簇
    normal_x = np.random.normal(2, 0.5, 50)
    normal_y = np.random.normal(2, 0.5, 50)
    # 欺诈行为簇 (Cash-out Fraud)
    fraud_x = np.random.normal(5, 0.4, 30)
    fraud_y = np.random.normal(5, 0.4, 30)

    # 2. 绘制散点
    ax2.scatter(normal_x, normal_y, c='#1f77b4', alpha=0.3, label='Normal Samples', s=30)
    ax2.scatter(fraud_x, fraud_y, c='#d62728', alpha=0.3, label='Fraud Samples', s=30)

    # 3. 绘制原型 (Prototypes)
    proto_normal = (2, 2)
    proto_fraud = (5, 5)
    ax2.scatter(*proto_normal, c='blue', s=200, marker='X', edgecolors='black', label='Prototype: Normal')
    ax2.scatter(*proto_fraud, c='red', s=200, marker='X', edgecolors='black', label='Prototype: Cash-out')

    # 4. 绘制待检测样本 Node A
    # 让 Node A 非常接近 Fraud 原型
    node_a = (4.9, 4.9)
    ax2.scatter(*node_a, c='yellow', s=250, marker='*', edgecolors='black', label='Target Node A', zorder=10)

    # 5. 绘制距离连线和标注
    arrow = FancyArrowPatch(node_a, proto_fraud, arrowstyle='<->', color='black', mutation_scale=15)
    ax2.add_patch(arrow)
    ax2.text(5.1, 4.8, "d = 0.12\n(High Conf.)", fontsize=11, fontweight='bold')

    # 绘制边界圈 (Manifold Boundary)
    circle = Circle(proto_fraud, radius=1.0, color='red', fill=False, linestyle='--', alpha=0.5, linewidth=2)
    ax2.add_patch(circle)
    ax2.text(5.8, 5.5, "Known Fraud\nManifold", fontsize=9, color='#d62728')

    ax2.set_title('Step 2: Latent Prototype Matching', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Latent Dimension 1')
    ax2.set_ylabel('Latent Dimension 2')
    ax2.legend(loc='lower right', fontsize=9, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("(b) Micro-level Interpretation: Explainable Decision Path", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()


# --- 执行绘图 ---
if __name__ == "__main__":
    # print("Generating Figure 8(a)...")
    # draw_figure_8a()
    print("\nGenerating Figure 8(b)...")
    draw_figure_8b()