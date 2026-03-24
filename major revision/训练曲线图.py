import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为学术论文常用的西文字体 (如 Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2

# 1. 模拟逼真的训练和验证数据 (如果你有真实数据，请替换这里)
np.random.seed(42) # 固定随机种子以保证可复现
epochs = np.arange(1, 81)

train_cls = 1.2 * np.exp(-epochs / 15) + 0.12 + np.random.normal(0, 0.01, 80)
train_pro = 0.9 * np.exp(-epochs / 20) + 0.08 + np.random.normal(0, 0.008, 80)
train_vib = 0.6 * np.exp(-epochs / 30) + 0.15 + np.random.normal(0, 0.005, 80)
train_total = train_cls + train_pro + train_vib

val_cls = 1.25 * np.exp(-epochs / 16) + 0.16 + np.random.normal(0, 0.015, 80)
val_pro = 0.95 * np.exp(-epochs / 21) + 0.10 + np.random.normal(0, 0.012, 80)
val_vib = 0.62 * np.exp(-epochs / 32) + 0.17 + np.random.normal(0, 0.008, 80)
val_total = val_cls + val_pro + val_vib

# ---------------------------------------------------------
# 2. 生成 Training Loss 独立图片
# ---------------------------------------------------------
fig_train, ax_train = plt.subplots(figsize=(6, 5), dpi=300)

ax_train.plot(epochs, train_total, label=r'Total Loss ($\mathcal{L}_{total}$)', color='#d62728', linewidth=2.5)
ax_train.plot(epochs, train_cls, label=r'Classification Loss ($\mathcal{L}_{cls}$)', color='#1f77b4', linestyle='--', linewidth=2)
ax_train.plot(epochs, train_pro, label=r'Prototype Loss ($\mathcal{L}_{pro}$)', color='#ff7f0e', linestyle='-.', linewidth=2)
ax_train.plot(epochs, train_vib, label=r'VIB Loss ($\mathcal{L}_{VIB}$)', color='#2ca02c', linestyle=':', linewidth=2)

ax_train.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold', pad=10)
ax_train.set_xlabel('Epochs', fontsize=13)
ax_train.set_ylabel('Loss', fontsize=13)
ax_train.set_xlim(0, 80)
ax_train.set_ylim(0, 3.0)
ax_train.grid(True, linestyle='--', alpha=0.6)
ax_train.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax_train.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.savefig('Figure_7_Training_Loss.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_train)

# ---------------------------------------------------------
# 3. 生成 Validation Loss 独立图片
# ---------------------------------------------------------
fig_val, ax_val = plt.subplots(figsize=(6, 5), dpi=300)

ax_val.plot(epochs, val_total, label=r'Total Loss ($\mathcal{L}_{total}$)', color='#d62728', linewidth=2.5)
ax_val.plot(epochs, val_cls, label=r'Classification Loss ($\mathcal{L}_{cls}$)', color='#1f77b4', linestyle='--', linewidth=2)
ax_val.plot(epochs, val_pro, label=r'Prototype Loss ($\mathcal{L}_{pro}$)', color='#ff7f0e', linestyle='-.', linewidth=2)
ax_val.plot(epochs, val_vib, label=r'VIB Loss ($\mathcal{L}_{VIB}$)', color='#2ca02c', linestyle=':', linewidth=2)

ax_val.set_title('Validation Loss over Epochs', fontsize=14, fontweight='bold', pad=10)
ax_val.set_xlabel('Epochs', fontsize=13)
ax_val.set_ylabel('Loss', fontsize=13)
ax_val.set_xlim(0, 80)
ax_val.set_ylim(0, 3.0)
ax_val.grid(True, linestyle='--', alpha=0.6)
ax_val.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax_val.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.savefig('Figure_7_Validation_Loss.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_val)