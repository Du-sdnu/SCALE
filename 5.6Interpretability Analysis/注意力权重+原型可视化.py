import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

# --- 设置绘图风格 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ==========================================
#  图 1: 注意力权重可视化 (保持不变)
# ==========================================
def draw_attention_chart():
    fig, ax = plt.subplots(figsize=(7, 6))

    paths = ['User-Trans-Merch\n(Transaction Pattern)', 'User-Device-User\n(Device Sharing)',
             'User-Attribute\n(Basic Info)']
    weights = [0.415, 0.382, 0.203]
    colors = ['#d62728', '#ff7f0e', '#1f77b4']

    bars = ax.bar(paths, weights, color=colors, alpha=0.85, edgecolor='black', width=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 0.5)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Step 1: Meta-path Attention Analysis', fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.text(0.5, 0.85, "High attention on\nTransaction & Device\nimplies 'Cash-out' risk",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# ==========================================
#  图 2: 潜在空间原型匹配 (单行标注版)
# ==========================================
def draw_latent_space():
    fig, ax = plt.subplots(figsize=(7, 6))

    np.random.seed(42)

    # 1. 定义中心点
    proto_normal = (2, 2)  # 左下
    proto_cashout = (5, 5)  # 右上
    proto_ml = (2, 5.5)  # 左上
    proto_ato = (5.5, 2)  # 右下

    # 2. 生成模拟数据点
    normal_x = np.random.normal(proto_normal[0], 0.4, 40)
    normal_y = np.random.normal(proto_normal[1], 0.4, 40)

    fraud_x = np.random.normal(proto_cashout[0], 0.35, 30)
    fraud_y = np.random.normal(proto_cashout[1], 0.35, 30)

    ml_x = np.random.normal(proto_ml[0], 0.35, 30)
    ml_y = np.random.normal(proto_ml[1], 0.35, 30)

    ato_x = np.random.normal(proto_ato[0], 0.35, 30)
    ato_y = np.random.normal(proto_ato[1], 0.35, 30)

    # 3. 绘制散点
    ax.scatter(normal_x, normal_y, c='#1f77b4', alpha=0.3, label='Normal', s=30)
    ax.scatter(fraud_x, fraud_y, c='#d62728', alpha=0.3, label='Cash-out', s=30)
    ax.scatter(ml_x, ml_y, c='#9467bd', alpha=0.3, label='Money Laundering', s=30)
    ax.scatter(ato_x, ato_y, c='#2ca02c', alpha=0.3, label='Account Takeover', s=30)

    # 绘制原型中心 'X'
    ax.scatter(*proto_normal, c='blue', s=150, marker='X', edgecolors='black')
    ax.scatter(*proto_cashout, c='red', s=150, marker='X', edgecolors='black')
    ax.scatter(*proto_ml, c='#9467bd', s=150, marker='X', edgecolors='black')
    ax.scatter(*proto_ato, c='#2ca02c', s=150, marker='X', edgecolors='black')

    # 4. 绘制边界圈与标注 (改为单行，去除\n)

    # --- Normal (左下) ---
    circle_normal = Circle(proto_normal, radius=0.9, color='#1f77b4', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle_normal)
    # y=3.1 (中心2.0 + 1.1)
    ax.text(proto_normal[0], 3.1, "Normal Behavior",
            fontsize=10, color='#1f77b4', fontweight='bold', ha='center', va='center')

    # --- Cash-out (右上) ---
    circle_cashout = Circle(proto_cashout, radius=0.9, color='#d62728', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle_cashout)
    # y=6.1 (中心5.0 + 1.1)
    ax.text(proto_cashout[0], 6.1, "Cash-out Fraud",
            fontsize=10, color='#d62728', fontweight='bold', ha='center', va='center')

    # --- Money Laundering (左上) ---
    circle_ml = Circle(proto_ml, radius=0.9, color='#9467bd', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle_ml)
    # y=6.6 (中心5.5 + 1.1)
    ax.text(proto_ml[0], 6.6, "Money Laundering Pattern",
            fontsize=10, color='#9467bd', fontweight='bold', ha='center', va='center')

    # --- Account Takeover (右下) ---
    circle_ato = Circle(proto_ato, radius=0.9, color='#2ca02c', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle_ato)
    # y=3.1 (中心2.0 + 1.1)
    ax.text(proto_ato[0], 3.1, "Account Takeover",
            fontsize=10, color='#2ca02c', fontweight='bold', ha='center', va='center')

    # 5. 绘制待检测样本 Node A
    node_a = (4.9, 4.9)
    ax.scatter(*node_a, c='yellow', s=250, marker='*', edgecolors='black', label='Target Node A', zorder=10)

    # 绘制距离连线
    arrow = FancyArrowPatch(node_a, proto_cashout, arrowstyle='<->', color='black', mutation_scale=15)
    ax.add_patch(arrow)
    ax.text(5.0, 4.6, "d = 0.12", fontsize=11, fontweight='bold')

    # 6. 图表设置
    ax.set_title('Step 2: Latent Prototype Matching', fontweight='bold', fontsize=14)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')

    # --- Legend (保持在底部) ---
    ax.legend(loc='lower center', fontsize=9, frameon=True,
              ncol=3, framealpha=0.9, edgecolor='gray')

    ax.grid(True, linestyle=':', alpha=0.6)

    # 严格限制坐标轴范围
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)

    plt.tight_layout()
    plt.show()


# --- 执行绘图 ---
if __name__ == "__main__":
    print("Generating Figure 1 (Attention Weights)...")
    draw_attention_chart()

    print("\nGenerating Figure 2 (Single Line Labels)...")
    draw_latent_space()