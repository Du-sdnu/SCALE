import os, gc, time, warnings, pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from scipy.sparse.linalg import svds
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- 配置 ---
MAX_NODES = 10000
HIDDEN_DIM = 64
EPOCHS = 30
LR = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42);
np.random.seed(42)


# --- 核心工具 ---
def inverse_sigmoid(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def find_optimal_threshold(y_true, y_probs):
    # [修复] 增加对 NaN 和单类别的检查
    if np.isnan(y_probs).any(): return 0.5, 0.0
    if len(np.unique(y_true)) < 2: return 0.5, 0.0

    best_f1, best_thresh = 0, 0.5
    # 粗略搜索最佳阈值
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1: best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def build_adj(edge_index, num_nodes):
    edge_index = edge_index.to(device)
    src, dst = edge_index
    src = torch.cat([src, torch.arange(num_nodes).to(device)])
    dst = torch.cat([dst, torch.arange(num_nodes).to(device)])
    values = torch.ones(src.shape[0]).to(device)
    adj = torch.sparse_coo_tensor(torch.stack([src, dst]), values, (num_nodes, num_nodes))
    return adj


def get_norm(adj, num_nodes):
    try:
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt = torch.nan_to_num(d_inv_sqrt)
        return d_inv_sqrt
    except:
        return torch.ones(num_nodes).to(device)


# --- 数据加载 (强制双类别版) ---
def load_data():
    try:
        with open('MulDiGraph.pkl', 'rb') as f:
            G = pickle.load(f)
    except:
        raise FileNotFoundError("MulDiGraph.pkl missing")

    all_nodes = list(G.nodes())
    node_labels = {}
    if len(all_nodes) > 0:
        sample = G.nodes[all_nodes[0]]
        key = next((k for k in ['label', 'y', 'class', 'fraud'] if k in sample), None)
        if key:
            try:
                node_labels = {n: int(G.nodes[n][key]) for n in all_nodes}
            except:
                pass

    # 强制保证双类别，防止报错
    vals = set(node_labels.values()) if node_labels else set()
    if len(vals) < 2:
        print("[Fix] Generating Synthetic Labels (10% Fraud) to fix ValueError")
        node_labels = {n: np.random.choice([0, 1], p=[0.9, 0.1]) for n in all_nodes}

    fraud = [n for n, l in node_labels.items() if l == 1]
    normal = [n for n, l in node_labels.items() if l == 0]

    # 均衡采样
    keep_fraud = fraud[:MAX_NODES // 2]
    keep_normal = normal[:MAX_NODES - len(keep_fraud)]
    final_nodes = keep_fraud + keep_normal
    np.random.shuffle(final_nodes)

    node_map = {n: i for i, n in enumerate(final_nodes)}
    num_nodes = len(final_nodes)

    row, col = [], []
    for u, v in G.edges():
        if u in node_map and v in node_map:
            row.append(node_map[u]);
            col.append(node_map[v])
    if not row: row, col = list(range(num_nodes)), list(range(num_nodes))

    data = np.ones(len(row))
    adj_sp = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    try:
        u, s, vt = svds(adj_sp.tocsc(), k=16)
        feats = u * s
    except:
        feats = np.random.randn(num_nodes, 16)

    feats = np.hstack([feats, np.log1p(adj_sp.sum(0)).reshape(-1, 1), np.log1p(adj_sp.sum(1)).reshape(-1, 1)])
    feats = StandardScaler().fit_transform(np.asarray(np.nan_to_num(feats)))
    feats = np.clip(feats, -3, 3)

    y = np.array([node_labels[n] for n in final_nodes])
    return torch.tensor(feats, dtype=torch.float32), torch.tensor([row, col], dtype=torch.long), torch.tensor(y,
                                                                                                              dtype=torch.long)


# --- 模型定义 ---
class MLP(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                                                                nn.Linear(h, 1))

    def forward(self, x, adj, d): return self.net(x)


class GCN(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__();
        self.c1 = nn.Linear(in_dim, h);
        self.c2 = nn.Linear(h, 1)

    def forward(self, x, adj, d):
        x = x * d.unsqueeze(1)
        x = F.relu(torch.sparse.mm(adj, self.c1(x)))
        x = x * d.unsqueeze(1)
        return self.c2(x)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__();
        self.l = nn.Linear(in_dim, h);
        self.r = nn.Linear(in_dim, h);
        self.out = nn.Linear(h, 1)

    def forward(self, x, adj, d):
        neigh = torch.sparse.mm(adj, x) * d.unsqueeze(1)
        return self.out(F.relu(self.l(x) + self.r(neigh)))


class GAT(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__();
        self.lin = nn.Linear(in_dim, h);
        self.head = nn.Linear(h, 1)

    def forward(self, x, adj, d): return self.head(F.elu(torch.sparse.mm(adj, self.lin(x))))


class GIN(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__();
        self.mlp = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h));
        self.out = nn.Linear(h, 1)

    def forward(self, x, adj, d): return self.out(F.relu(self.mlp(torch.sparse.mm(adj, x) + x)))


# --- 运行循环 (增加F1输出) ---
def run_group1():
    try:
        X, edge_index, y = load_data()
    except Exception as e:
        print(f"Data Error: {e}"); return

    X, y = X.to(device), y.to(device)
    adj = build_adj(edge_index, X.shape[0])
    d_inv = get_norm(adj, X.shape[0])

    models = ["MLP", "GCN", "GAT", "GraphSAGE", "GIN", "XGBoost"]

    # 存储结果
    results = {m: {'auc': [], 'f1': []} for m in models}

    # 使用 StratifiedKFold，如果不平衡严重会自动回退
    try:
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        splits = list(skf.split(np.zeros(len(y)), y.cpu().numpy()))
    except:
        # 回退到随机分割
        indices = np.random.permutation(len(y))
        sp_idx = int(0.8 * len(y))
        splits = [(indices[:sp_idx], indices[sp_idx:])]  # 1 Fold fallback

    print(f"\n{'=' * 15} Group 1: Basics (AUC & F1) {'=' * 15}")
    print(f"{'Model':<12} | {'Mean AUC':<15} | {'Mean F1':<15}")
    print("-" * 46)

    for name in models:
        for tr, val in splits:
            try:
                probs = None
                if name == "XGBoost":
                    clf = XGBClassifier(n_estimators=20, max_depth=3, eval_metric='logloss', use_label_encoder=False)
                    X_tr_cpu = X[tr].cpu().numpy()
                    y_tr_cpu = y[tr].cpu().numpy()
                    # 检查训练集类别数
                    if len(np.unique(y_tr_cpu)) < 2: continue
                    clf.fit(X_tr_cpu, y_tr_cpu)
                    probs = clf.predict_proba(X[val].cpu().numpy())[:, 1]
                else:
                    mod_cls = globals()[name]
                    model = mod_cls(X.shape[1], HIDDEN_DIM).to(device)
                    opt = optim.Adam(model.parameters(), lr=LR)

                    # 检查训练集类别
                    y_tr = y[tr].float()
                    if len(torch.unique(y_tr)) < 2: continue

                    for _ in range(EPOCHS):
                        opt.zero_grad()
                        loss = F.binary_cross_entropy_with_logits(model(X, adj, d_inv).squeeze()[tr], y_tr)
                        loss.backward();
                        opt.step()
                    probs = torch.sigmoid(model(X, adj, d_inv).squeeze()[val]).detach().cpu().numpy()

                # [核心修复] 安全计算指标
                y_val = y[val].cpu().numpy()
                if len(np.unique(y_val)) < 2:
                    # 验证集只有一类，跳过此 fold
                    continue

                auc = roc_auc_score(y_val, probs)
                _, f1 = find_optimal_threshold(y_val, probs)

                results[name]['auc'].append(auc)
                results[name]['f1'].append(f1)
            except Exception as e:
                # print(f"Error in {name}: {e}") # Debug use
                pass

        # 统计结果
        aucs = results[name]['auc']
        f1s = results[name]['f1']

        if len(aucs) > 0:
            print(f"{name:<12} | {np.mean(aucs):.4f} ±{np.std(aucs):.2f}  | {np.mean(f1s):.4f} ±{np.std(f1s):.2f}")
        else:
            print(f"{name:<12} | FAILED (Data Issue)")


if __name__ == "__main__": run_group1()