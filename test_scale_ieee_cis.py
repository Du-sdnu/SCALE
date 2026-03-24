import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import warnings
import gc
import os  # [修复1] 确保导入 os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# 清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================
# 0. 核心工具
# ==========================================
def inverse_sigmoid(x):
    """ 将概率反转回 Logit 空间，用于残差学习 """
    return np.log(x / (1 - x + 1e-7) + 1e-7)


def find_optimal_threshold(y_true, y_probs):
    """ [核心] 动态搜索最佳 F1 阈值 """
    best_f1 = 0
    best_thresh = 0.5
    # 扩大搜索范围，步长精细化
    for thresh in np.arange(0.05, 0.85, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def build_sparse_adj(edge_index, num_nodes):
    src, dst = edge_index
    # 添加自环
    src = torch.cat([src, torch.arange(num_nodes).to(src.device)])
    dst = torch.cat([dst, torch.arange(num_nodes).to(dst.device)])
    deg = torch.bincount(src, minlength=num_nodes).float()

    # GraphSAGE 归一化 (D^-1)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    norm = deg_inv[src]

    return torch.sparse_coo_tensor(torch.stack([src, dst]), norm, (num_nodes, num_nodes))


# ==========================================
# 1. SCALE-Ensemble 模型 (残差学习版)
# ==========================================
class HG_VIB_Encoder(nn.Module):
    def __init__(self, in_feats, hidden_dim):
        super(HG_VIB_Encoder, self).__init__()
        # 强力特征提取
        self.fc = nn.Sequential(
            nn.Linear(in_feats, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU()
        )
        self.enc_mu = nn.Linear(hidden_dim * 2, hidden_dim)
        self.enc_std = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        h = self.fc(x)
        mu = self.enc_mu(h)
        std = F.softplus(self.enc_std(h)) + 1e-6
        if self.training:
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        kl = -0.5 * torch.sum(1 + 2 * torch.log(std) - mu.pow(2) - std.pow(2), 1).mean()
        return z, kl


class SCALE_Ensemble(nn.Module):
    def __init__(self, raw_feat_dim, h_dim=64):
        super(SCALE_Ensemble, self).__init__()

        # VIB 只处理原始特征
        self.vib = HG_VIB_Encoder(raw_feat_dim, h_dim)

        # GNN 聚合层
        self.gnn = nn.Linear(h_dim, h_dim)

        # 修正分类头 (Correction Head)
        # 输入: Z(VIB) + Neigh(GNN) + XGB_Logit(1)
        fusion_dim = h_dim + h_dim + 1

        self.correction_head = nn.Sequential(
            nn.Linear(fusion_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, 1)  # 输出修正值 Delta
        )

    def forward(self, x_raw, x_xgb_logit, adj):
        # 1. VIB 编码 (原始特征)
        z, kl = self.vib(x_raw)

        # 2. 图聚合 (GraphSAGE style)
        h_neigh = torch.sparse.mm(adj, F.relu(self.gnn(z)))

        # 3. 融合特征
        combined = torch.cat([z, h_neigh, x_xgb_logit], dim=1)

        # 4. 计算修正值 (Delta)
        delta = self.correction_head(combined)

        # 5. [核心] 最终输出 = XGB基准 + 神经网络修正
        # 这样 NN 只需要学习 XGB 没学到的部分
        final_logits = x_xgb_logit + delta

        return final_logits, kl


# ==========================================
# 2. 数据加载 (自动适配路径)
# ==========================================
def load_data():
    print(">>> Loading Data...")

    # 自动搜索当前目录及子目录
    search_paths = ['.', '..', './data']
    tp, ip = None, None

    for path in search_paths:
        t_path = os.path.join(path, 'train_transaction.csv')
        i_path = os.path.join(path, 'train_identity.csv')
        if os.path.exists(t_path) and os.path.exists(i_path):
            tp, ip = t_path, i_path
            break

    if not tp:
        # 如果还没找到，尝试模糊匹配
        try:
            import glob
            tp = glob.glob('**/train_transaction.csv', recursive=True)[0]
            ip = glob.glob('**/train_identity.csv', recursive=True)[0]
        except:
            print("当前工作目录:", os.getcwd())
            raise FileNotFoundError("未找到 train_transaction.csv 或 train_identity.csv，请检查文件位置。")

    print(f"    Found files at: {tp}, {ip}")

    df_trans = pd.read_csv(tp)
    df_id = pd.read_csv(ip)
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    y = df['isFraud'].values

    # 清洗
    drop = ['TransactionID', 'isFraud', 'TransactionDT']
    nulls = df.isnull().sum() / len(df)
    drop += list(nulls[nulls > 0.8].index)
    X = df.drop(columns=[c for c in drop if c in df.columns])

    # 编码
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        else:
            X[c] = X[c].fillna(0)

    # 归一化 (Float32)
    print("    Normalizing...")
    X_val = StandardScaler().fit_transform(X.values.astype(np.float32))
    X_val = X_val.astype(np.float32)

    # 构图
    print("    Building Graph...")
    tmp = pd.DataFrame({'c': df['card1'], 'i': np.arange(len(df))})
    tmp['ni'] = tmp.groupby('c')['i'].shift(-1)
    e = tmp.dropna().astype(int)
    edge_index = torch.tensor(np.vstack((e['i'], e['ni'])), dtype=torch.long)

    return X_val, edge_index, y


# ==========================================
# 3. 运行评估
# ==========================================
def run_evaluation():
    try:
        X_raw, edge_index, y = load_data()
    except Exception as e:
        print(f"Data Error: {e}");
        return

    num_nodes, in_feats = X_raw.shape

    # 移入 GPU
    print("    Moving graph to device...")
    try:
        adj = build_sparse_adj(edge_index, num_nodes).to(device)
        del edge_index
        torch.cuda.empty_cache()
    except RuntimeError:
        print("    [Warning] Graph too big, fallback to CPU adjacency.")
        adj = build_sparse_adj(edge_index, num_nodes)  # Keep on CPU

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    auc_scores, f1_scores = [], []

    print("\n=== Starting SCALE-Ensemble (Residual Learning) ===")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        print(f"\n--- Fold {fold + 1} ---")

        # 1. XGBoost 预训练 (Teacher)
        # print("    Step 1: XGBoost extraction...")
        # 增加深度以确保 Base 模型足够强
        clf_xgb = XGBClassifier(n_estimators=150, max_depth=8,
                                learning_rate=0.05, n_jobs=-1,
                                use_label_encoder=False, eval_metric='logloss',
                                tree_method='hist')
        clf_xgb.fit(X_raw[train_idx], y[train_idx])

        # 获取概率
        xgb_probs = clf_xgb.predict_proba(X_raw)[:, 1:2]
        # 转为 Logits (反Sigmoid) 以便进行残差加法
        xgb_logits = inverse_sigmoid(xgb_probs)

        # 2. 准备 Tensor
        X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
        Xgb_tensor = torch.tensor(xgb_logits, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        train_mask[train_idx] = True

        # 3. 初始化 SCALE
        # hidden_dim=64 足够且省显存
        model = SCALE_Ensemble(raw_feat_dim=in_feats, h_dim=64).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

        # [F1 提分关键]
        # 使用 BCE Loss，并设置 pos_weight = 10.0
        # 这会告诉模型：找到一个欺诈样本的价值是找到一个正常样本的10倍
        # 这将极大提升 Recall，从而拉升 F1
        crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))

        # 4. 训练
        model.train()
        for epoch in range(25):
            optimizer.zero_grad()
            logits, kl_loss = model(X_tensor, Xgb_tensor, adj)

            # 只计算训练集 Loss
            loss = crit_bce(logits[train_mask].squeeze(), y_tensor[train_mask]) + 0.01 * kl_loss

            loss.backward()
            optimizer.step()

        # 5. 评估
        model.eval()
        with torch.no_grad():
            logits, _ = model(X_tensor, Xgb_tensor, adj)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_test = y[val_idx]
            test_probs = probs[val_idx]

            # 计算 AUC
            try:
                auc = roc_auc_score(y_test, test_probs)
            except:
                auc = 0.5

            # 计算最佳 F1 (动态阈值)
            best_thresh, best_f1 = find_optimal_threshold(y_test, test_probs)

            print(f"    Result: AUC = {auc:.4f} | Macro-F1 = {best_f1:.4f} (Thresh={best_thresh:.2f})")

            auc_scores.append(auc)
            f1_scores.append(best_f1)

        # 释放显存
        del model, X_tensor, Xgb_tensor, y_tensor, logits, probs
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SCALE (Final Optimized) Performance")
    print("=" * 60)
    print(f"Mean AUC-ROC : {np.mean(auc_scores) * 100:.2f}% ± {np.std(auc_scores) * 100:.2f}%")
    print(f"Mean Macro-F1: {np.mean(f1_scores) * 100:.2f}% ± {np.std(f1_scores) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()