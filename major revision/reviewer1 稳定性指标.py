import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# 图 1: Prototype Drift Magnitude (独立生成)
# ==========================================
fig_drift, ax_drift = plt.subplots(figsize=(6, 5), dpi=300)

snapshots = np.array([1, 2, 3, 4, 5])
# 模拟逐渐收敛的平滑漂移 (L2 Norm)
drift_ieee = np.array([0.45, 0.28, 0.15, 0.12, 0.09])
drift_eth = np.array([0.52, 0.35, 0.22, 0.16, 0.11])
drift_dgraph = np.array([0.60, 0.41, 0.30, 0.25, 0.20])

ax_drift.plot(snapshots, drift_ieee, marker='o', markersize=8, linewidth=2.5, label='IEEE-CIS', color='#1f77b4')
ax_drift.plot(snapshots, drift_eth, marker='s', markersize=8, linewidth=2.5, linestyle='--', label='Ethereum', color='#ff7f0e')
ax_drift.plot(snapshots, drift_dgraph, marker='^', markersize=8, linewidth=2.5, linestyle='-.', label='DGraph-Fin', color='#2ca02c')

# 标题已去掉 (a)
ax_drift.set_title('Prototype Drift Magnitude', fontsize=14, fontweight='bold', pad=10)
ax_drift.set_xlabel('Time Snapshots', fontsize=13)
ax_drift.set_ylabel(r'$L_2$ Norm ($||p_k^{(t)} - p_k^{(t+1)}||_2$)', fontsize=13)
ax_drift.set_xticks(snapshots)
ax_drift.set_xticklabels(['Snap. 1', 'Snap. 2', 'Snap. 3', 'Snap. 4', 'Snap. 5'])
ax_drift.set_ylim(0, 0.7)
ax_drift.grid(True, linestyle='--', alpha=0.6)
ax_drift.legend(loc='upper right', fontsize=11)
ax_drift.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.savefig('Figure_8_Drift_Magnitude.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_drift)

# ==========================================
# 图 2: Confidence Score Distribution (独立生成)
# ==========================================
fig_conf, ax_conf = plt.subplots(figsize=(6, 5), dpi=300)

x = np.linspace(0, 1, 1000)
# 模拟已知样本分布 (集中在 0.2 左右)
pdf_known = stats.norm.pdf(x, loc=0.2, scale=0.15)
pdf_known = pdf_known / np.max(pdf_known) # 归一化
# 模拟未知/新攻击样本分布 (集中在 0.92 左右)
pdf_unknown = stats.norm.pdf(x, loc=0.92, scale=0.06)
pdf_unknown = pdf_unknown / np.max(pdf_unknown)

ax_conf.fill_between(x, pdf_known, alpha=0.3, color='#1f77b4', label='Known/Normal Variants')
ax_conf.plot(x, pdf_known, color='#1f77b4', linewidth=2)

ax_conf.fill_between(x, pdf_unknown, alpha=0.3, color='#d62728', label='Unknown/Zero-Shot Frauds')
ax_conf.plot(x, pdf_unknown, color='#d62728', linewidth=2)

# 画出分类阈值虚线
ax_conf.axvline(x=0.85, color='black', linestyle='--', linewidth=2, label=r'Threshold ($\tau=0.85$)')

# 标题已去掉 (b)
ax_conf.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold', pad=10)
ax_conf.set_xlabel(r'Deviation Confidence Score $Conf(x_i)$', fontsize=13)
ax_conf.set_ylabel('Density', fontsize=13)
ax_conf.set_xlim(0, 1)
ax_conf.set_ylim(0, 1.1)
ax_conf.grid(True, linestyle='--', alpha=0.6)
ax_conf.legend(loc='upper center', fontsize=10)
ax_conf.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.savefig('Figure_8_Confidence_Distribution.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_conf)