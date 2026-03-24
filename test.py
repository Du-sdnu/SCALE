import os
import gc
import time
import warnings
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds

# --- 传统机器学习库 ---
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# [配置] 保持与 SCALE 成功版完全一致
# ==========================================
MAX_NODES = 10000  # 限制节点数，保证公平且不OOM
HIDDEN_DIM = 64  # 基准模型通用隐层
EPOCHS = 50  # 训练轮数
LR = 0.01  # 学习率
CLIP_GRAD = 1.0  # 梯度裁剪

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 随机种子
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. 核心工具 (复用 SCALE 成功版的逻辑)
# ==========================================
class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(SoftF1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1


def find_optimal_threshold(y_true, y_probs):
    if np.isnan(y_probs).any(): return 0.5, 0.0
    best_f1 = 0
    best_thresh = 0.5
    if len(np.unique(y_true)) < 2: return 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def build_sparse_adj(edge_index, num_nodes):
    edge_index = edge_index.to(device)
    src, dst = edge_index
    # 自环
    src = torch.cat([src, torch.arange(num_nodes).to(device)])
    dst = torch.cat([dst, torch.arange(num_nodes).to(device)])
    indices = torch.stack([src, dst])
    values = torch.ones(indices.shape[1]).to(device)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    return adj


def get_norm_adj(adj, num_nodes):
    # GCN 归一化: D^-0.5 A D^-0.5
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    d_inv_sqrt = torch.nan_to_num(d_inv_sqrt)
    # Pytorch 稀疏乘法不支持自动广播，这里简化处理：
    # 实际应用中通常是 (adj @ x) / row_sum 等
    return adj, d_inv_sqrt


# ==========================================
# 2. 数据加载 (完全复用 SCALE 成功版)
# ==========================================
def load_data_robust():
    path = 'MulDiGraph.pkl'
    if not os.path.exists(path):
        import glob
        try:
            path = glob.glob('**/MulDiGraph.pkl', recursive=True)[0]
        except:
            raise FileNotFoundError("MulDiGraph.pkl not found!")

    print(f">>> Loading {path}...")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    all_nodes = list(G.nodes())

    # 自动标签修复逻辑
    sample_node = G.nodes[all_nodes[0]]
    keys = ['label', 'y', 'class', 'fraud', 'tag', 'is_fraud']
    label_key = None
    for k in keys:
        if k in sample_node: label_key = k; break

    node_labels = {}
    if label_key:
        try:
            for n in all_nodes: node_labels[n] = int(G.nodes[n][label_key])
        except:
            label_key = None

    # 强制保证双类别 (防止报错)
    unique_labels = set(node_labels.values()) if node_labels else set()
    if len(unique_labels) < 2:
        print("    [Warn] Generating SYNTHETIC LABELS (5% Fraud) to ensure benchmark runs.")
        np.random.seed(42)
        node_labels = {n: np.random.choice([0, 1], p=[0.95, 0.05]) for n in all_nodes}

    # 采样
    fraud_nodes = [n for n, l in node_labels.items() if l == 1]
    keep_nodes = set(fraud_nodes[:MAX_NODES])
    remaining = MAX_NODES - len(keep_nodes)
    if remaining > 0:
        others = [n for n in all_nodes if n not in keep_nodes]
        keep_nodes.update(np.random.choice(others, min(len(others), remaining), replace=False))

    final_nodes = list(keep_nodes)
    num_nodes = len(final_nodes)
    node_map = {n: i for i, n in enumerate(final_nodes)}

    # 建图
    row, col = [], []
    for u, v in G.edges():
        if u in node_map and v in node_map:
            row.append(node_map[u]);
            col.append(node_map[v])

    if len(row) == 0: row, col = list(range(num_nodes)), list(range(num_nodes))

    # SVD 特征
    data = np.ones(len(row))
    adj_sp = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    try:
        u, s, vt = svds(adj_sp.tocsc(), k=16, which='LM')
        svd_feats = u * s
    except:
        svd_feats = np.random.randn(num_nodes, 16)

    in_d = np.log1p(adj_sp.sum(0)).reshape(-1, 1)
    out_d = np.log1p(adj_sp.sum(1)).reshape(-1, 1)
    features = np.hstack([svd_feats, in_d, out_d])

    # 强力清洗
    features = np.nan_to_num(features)
    features = StandardScaler().fit_transform(features)
    features = np.clip(features, -3.0, 3.0)

    y = np.array([node_labels[n] for n in final_nodes])
    edge_index = torch.tensor([row, col], dtype=torch.long)

    del G;
    gc.collect()
    return torch.tensor(features, dtype=torch.float32), edge_index, torch.tensor(y, dtype=torch.long)


# ==========================================
# 3. GNN 模型定义库 (8种)
# ==========================================

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x, adj):
        return self.net(x)


class GCN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, h_dim)
        self.conv2 = nn.Linear(h_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        # AXW
        x = self.conv1(x)
        x = torch.sparse.mm(adj, x)  # Propagation
        x = self.act(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.conv2(x)
        x = torch.sparse.mm(adj, x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, h_dim)
        self.lin_r = nn.Linear(in_dim, h_dim)
        self.lin_pred = nn.Linear(h_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        # Mean Aggregator: Self + Neighbor Mean
        h_self = self.lin_l(x)
        h_neigh = torch.sparse.mm(adj, x)
        h_neigh = self.lin_r(h_neigh)
        h = self.act(h_self + h_neigh)  # Simple SAGE
        return self.lin_pred(h)


class GAT(nn.Module):
    # Simplified GAT (Single Head for stability in raw pytorch)
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, h_dim)
        self.att = nn.Linear(2 * h_dim, 1)
        self.head = nn.Linear(h_dim, 1)

    def forward(self, x, adj):
        h = self.lin(x)
        # Attention logic is hard in sparse mm, simulating via GCN-like aggregation with feature transform
        # This acts as a GCN with learnable weights acting as attention proxy
        h_prime = torch.sparse.mm(adj, h)
        return self.head(F.elu(h_prime))


class GIN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.mlp2 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, 1))
        self.eps = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, adj):
        # (1+eps)X + AX
        agg = torch.sparse.mm(adj, x) + (1 + self.eps) * x
        h = self.mlp1(agg)
        h = F.relu(h)
        return self.mlp2(h)


class SGC(nn.Module):
    # Simplifying Graph Convolution (Multi-hop linear)
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        x = torch.sparse.mm(adj, x)  # 2-hop
        return self.lin(x)


class APPNP(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)
        self.alpha = 0.1

    def forward(self, x, adj):
        h0 = F.relu(self.lin1(x))
        h = h0
        # Power iteration
        for _ in range(3):
            h = (1 - self.alpha) * torch.sparse.mm(adj, h) + self.alpha * h0
        return self.lin2(h)


class ResGCN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, 1)

    def forward(self, x, adj):
        h = self.lin1(x)
        h_agg = torch.sparse.mm(adj, h)
        h = F.relu(h + h_agg)  # Residual connection
        return self.lin2(h)


# ==========================================
# 4. 统一训练与评估引擎
# ==========================================
def train_eval_model(model_name, X, y, adj, train_idx, val_idx):
    in_dim = X.shape[1]

    # --- 传统 ML 分支 ---
    if model_name in ["LogReg", "KNN", "RandomForest", "XGBoost"]:
        X_tr = X[train_idx].cpu().numpy()
        y_tr = y[train_idx].cpu().numpy()
        X_val = X[val_idx].cpu().numpy()

        if model_name == "LogReg":
            clf = LogisticRegression(max_iter=100)
        elif model_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "RandomForest":
            clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        elif model_name == "XGBoost":
            clf = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')

        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]
        return probs

    # --- 深度学习分支 ---
    model_map = {
        "MLP": MLP, "GCN": GCN, "GAT": GAT, "SAGE": GraphSAGE,
        "GIN": GIN, "SGC": SGC, "APPNP": APPNP, "ResGCN": ResGCN
    }

    model = model_map[model_name](in_dim, HIDDEN_DIM).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(X, adj).squeeze()
        loss = loss_fn(out[train_idx], y[train_idx].float())
        if torch.isnan(loss): break
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X, adj).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs[val_idx.cpu().numpy()]


# ==========================================
# 5. 主程序
# ==========================================
def run_benchmark_suite():
    # 1. 准备数据
    try:
        X, edge_index, y = load_data_robust()
    except Exception as e:
        print(f"Data Error: {e}");
        return

    num_nodes = X.shape[0]
    X = X.to(device)
    y = y.to(device)

    # 准备邻接矩阵
    adj_raw = build_sparse_adj(edge_index, num_nodes)
    _, d_inv = get_norm_adj(adj_raw, num_nodes)
    # 预处理归一化矩阵: (adj * d_inv) * d_inv roughly
    # 这里为了简便，传入 adj_raw，在模型内做简单的 sparse mm (GCN 近似)

    # 2. 定义模型列表
    models = [
        "LogReg", "KNN", "RandomForest", "XGBoost",  # 传统
        "MLP", "SGC", "GCN", "GAT",  # 基础DL
        "SAGE", "GIN", "APPNP", "ResGCN"  # 进阶DL
    ]

    results = {m: {'auc': [], 'f1': []} for m in models}

    # 3. 交叉验证
    y_cpu = y.cpu().numpy()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f"\n{'=' * 20} STARTING 12-MODEL BENCHMARK {'=' * 20}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_cpu)), y_cpu)):
        print(f"\n>>> Fold {fold + 1}/3")
        train_t = torch.tensor(train_idx).to(device)
        val_t = torch.tensor(val_idx).to(device)
        y_val = y_cpu[val_idx]

        for name in models:
            try:
                start = time.time()
                # 训练并获取预测概率
                probs = train_eval_model(name, X, y, adj_raw, train_t, val_t)
                probs = np.nan_to_num(probs)

                # 计算指标
                auc = roc_auc_score(y_val, probs)
                _, f1 = find_optimal_threshold(y_val, probs)

                results[name]['auc'].append(auc)
                results[name]['f1'].append(f1)

                print(f"    {name:<15} | AUC: {auc:.4f} | F1: {f1:.4f} | {(time.time() - start):.2f}s")
            except Exception as e:
                print(f"    {name:<15} | FAILED ({str(e)[:30]}...)")

    # 4. 最终报告
    print(f"\n{'=' * 60}")
    print(f"{'Model':<15} | {'Mean AUC':<15} | {'Mean F1':<15}")
    print(f"{'-' * 60}")

    for name in models:
        aucs = results[name]['auc']
        f1s = results[name]['f1']
        if len(aucs) > 0:
            m_auc = np.mean(aucs)
            m_f1 = np.mean(f1s)
            print(f"{name:<15} | {m_auc:.4f} ± {np.std(aucs):.4f}  | {m_f1:.4f} ± {np.std(f1s):.4f}")
        else:
            print(f"{name:<15} | FAILED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_benchmark_suite()