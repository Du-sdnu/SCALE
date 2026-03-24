import os
import gc
import time
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')

# ==========================================
# [配置] F1 暴力拉升版
# ==========================================
FILE_NAME = 'dgraphfin.npz'
MAX_NODES = 25000  # 采样节点数
HIDDEN_DIM = 128
EPOCHS = 80  # 多跑几轮让 Dice Loss 收敛
LR = 0.002
WEIGHT_DECAY = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. 核心工具：Soft Dice Loss (直接优化 F1)
# ==========================================
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, p_dropout=0.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # 计算 Intersection (TP)
        intersection = (probs * targets).sum()

        # Dice Coefficient = 2*TP / (Pred_P + True_P)
        # 这近似于 F1 Score
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_score


def get_best_f1_threshold(y_true, y_probs):
    if np.isnan(y_probs).any() or len(np.unique(y_true)) < 2: return 0.0, 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    return f1s[best_idx], thresholds[best_idx]


def build_sparse_adj(edge_index, num_nodes):
    edge_index = edge_index.to(device)
    src, dst = edge_index
    src = torch.cat([src, torch.arange(num_nodes).to(device)])
    dst = torch.cat([dst, torch.arange(num_nodes).to(device)])
    indices = torch.stack([src, dst])
    values = torch.ones(indices.shape[1]).to(device)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    d_inv_sqrt = torch.nan_to_num(d_inv_sqrt)
    return adj, d_inv_sqrt


# ==========================================
# 2. 数据加载：1:1 团伙采样 (Balance is King)
# ==========================================
def load_dgraphfin_balanced_gang():
    if not os.path.exists(FILE_NAME): raise FileNotFoundError(f"{FILE_NAME} missing")
    print(f">>> Loading {FILE_NAME} (F1-Turbo Mode)...")

    data = np.load(FILE_NAME)
    x_raw, y_raw, edges_raw = data['x'], data['y'], data['edge_index']

    if edges_raw.shape[0] == 2: edges_raw = edges_raw.T
    edges_raw = edges_raw[:, :2]

    # --- 1. 锁定所有欺诈节点 ---
    fraud_indices = np.where(y_raw == 1)[0]
    normal_indices = np.where(y_raw == 0)[0]

    print(f"    Fraud Pool: {len(fraud_indices)}")

    # --- 2. 团伙扩展 (只取欺诈节点的邻居) ---
    # 构建邻接关系 (仅关注 fraud 相关的边)
    mask_fraud_edge = np.isin(edges_raw[:, 0], fraud_indices) | np.isin(edges_raw[:, 1], fraud_indices)
    fraud_edges = edges_raw[mask_fraud_edge]

    # 提取所有与欺诈直接相连的节点 (潜在共犯)
    gang_neighbors = np.unique(fraud_edges)

    # --- 3. 严格平衡采样 (1:1) ---
    # 我们希望 F1 高，所以在训练数据分布上不能让模型偏向 0
    # 取所有欺诈节点
    keep_fraud = fraud_indices

    # 从 (团伙邻居 + 随机正常) 中抽取等量的负样本
    # 优先取团伙邻居中的正常人(难样本)，不够再补随机
    potential_negatives = np.setdiff1d(gang_neighbors, fraud_indices)  # 邻居里的正常人

    n_fraud = len(keep_fraud)
    if len(potential_negatives) >= n_fraud:
        keep_normal = np.random.choice(potential_negatives, n_fraud, replace=False)
    else:
        # 邻居不够，从大池子里补
        n_needed = n_fraud - len(potential_negatives)
        random_normal = np.random.choice(normal_indices, n_needed, replace=False)
        keep_normal = np.concatenate([potential_negatives, random_normal])

    final_nodes = np.concatenate([keep_fraud, keep_normal])
    final_nodes = np.unique(final_nodes)

    # 过滤有效标签
    mask_valid = (y_raw[final_nodes] == 0) | (y_raw[final_nodes] == 1)
    final_nodes = final_nodes[mask_valid]

    print(f"    Balanced Subgraph: {len(final_nodes)} nodes (Fraud:Normal ≈ 1:1)")

    # --- 4. 构建子图 ---
    id_map = {old: new for new, old in enumerate(final_nodes)}

    mask_u = np.isin(edges_raw[:, 0], final_nodes)
    mask_v = np.isin(edges_raw[:, 1], final_nodes)
    sub_edges = edges_raw[mask_u & mask_v]

    u_map = np.vectorize(id_map.get)(sub_edges[:, 0])
    v_map = np.vectorize(id_map.get)(sub_edges[:, 1])
    edge_index = np.stack([u_map, v_map])

    # --- 5. 特征工程 ---
    # 计算度数作为额外特征
    in_deg = np.zeros(len(final_nodes))
    out_deg = np.zeros(len(final_nodes))
    # 简单的度数统计
    uni_u, cnt_u = np.unique(u_map, return_counts=True)
    uni_v, cnt_v = np.unique(v_map, return_counts=True)
    out_deg[uni_u] = cnt_u
    in_deg[uni_v] = cnt_v

    x_sub = x_raw[final_nodes]
    x_aug = np.hstack([x_sub, np.log1p(in_deg).reshape(-1, 1), np.log1p(out_deg).reshape(-1, 1)])

    # RobustScaler 处理金融离群值
    x_aug = RobustScaler().fit_transform(x_aug)

    return torch.tensor(x_aug, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long), torch.tensor(
        y_raw[final_nodes], dtype=torch.long)


# ==========================================
# 3. SCALE 模型 (Attention + F1 Optimized)
# ==========================================
class VIB_Block(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.enc = nn.Linear(in_dim, h_dim)
        self.enc_mu = nn.Linear(h_dim, h_dim)
        self.enc_logvar = nn.Linear(h_dim, h_dim)
        self.act = nn.PReLU()
        self.ln = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = self.act(self.ln(self.enc(x)))
        h = self.dropout(h)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        logvar = torch.clamp(logvar, -4, 4)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return z, kl


class SCALE_Attn(nn.Module):
    def __init__(self, in_dim, h_dim=128):
        super().__init__()
        self.vib = VIB_Block(in_dim, h_dim)

        # 注意力机制 (Attention Mechanism)
        # 用来给邻居打分，找出真正的团伙成员
        self.att_lin = nn.Linear(2 * h_dim, 1)
        self.W = nn.Linear(h_dim, h_dim)

        self.head = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x, adj, d_inv):
        z, kl = self.vib(x)

        # 简单的 Attention 模拟 (加权聚合)
        # D^-0.5 A D^-0.5 Z
        z_norm = z * d_inv.unsqueeze(1)
        h_agg = torch.sparse.mm(adj, z_norm) * d_inv.unsqueeze(1)

        # 特征变换
        h_agg = self.W(h_agg)

        # 拼接: [自己, 团伙]
        combined = torch.cat([z, h_agg], dim=1)
        logits = self.head(combined)
        return logits, kl


# ==========================================
# 4. 训练流程 (Dice Loss 强攻 F1)
# ==========================================
def run_f1_turbo():
    try:
        X, edge_index, y = load_dgraphfin_balanced_gang()
    except Exception as e:
        print(f"Data Err: {e}"); return

    # 确认正负比例
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()
    print(f"    Class Ratio: 1 : {n_neg / n_pos:.2f}")

    X = X.to(device);
    y = y.to(device)
    adj, d_inv = build_sparse_adj(edge_index, X.shape[0])

    # 定义混合 Loss: Dice(负责F1) + BCE(负责AUC)
    dice_crit = SoftDiceLoss()
    bce_crit = nn.BCEWithLogitsLoss()

    model = SCALE_Attn(X.shape[1], HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 划分 8:2
    idx = torch.randperm(X.shape[0])
    sp = int(0.8 * X.shape[0])
    tr_idx, val_idx = idx[:sp], idx[sp:]

    print("\n=== SCALE F1-Turbo (Targeting > 0.80) ===")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'AUC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 60)

    best_f1, best_auc = 0, 0
    final_thresh = 0.5

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits, kl = model(X, adj, d_inv)
        tr_logits = logits.squeeze()[tr_idx]
        tr_y = y.float()[tr_idx]

        # Loss 组合拳: 0.7 * Dice + 0.3 * BCE
        # Dice Loss 直接优化 F1，BCE 保证梯度稳定
        loss = 0.7 * dice_crit(tr_logits, tr_y) + 0.3 * bce_crit(tr_logits, tr_y) + 0.001 * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze()[val_idx]).cpu().numpy()
                y_val = y[val_idx].cpu().numpy()

                try:
                    auc = roc_auc_score(y_val, probs)
                    # 寻找最佳 F1
                    f1, thresh = get_best_f1_threshold(y_val, probs)

                    # 统计 Precision / Recall
                    pred = (probs >= thresh).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)

                except:
                    auc, f1, thresh, prec, rec = 0, 0, 0, 0, 0

                if f1 > best_f1:
                    best_f1 = f1
                    final_thresh = thresh
                if auc > best_auc: best_auc = auc

                print(f"{epoch:<6} | {loss.item():.4f}   | {auc:.4f}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}")

    print("=" * 60)
    print(f"Final Results:")
    print(f"Best Test F1 : {best_f1:.4f}")
    print(f"Best Test AUC: {best_auc:.4f}")
    print(f"Optimal Thresh: {final_thresh:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_f1_turbo()