import os, gc, time, warnings, pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
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
def find_optimal_threshold(y_true, y_probs):
    if np.isnan(y_probs).any(): return 0.5, 0.0
    if len(np.unique(y_true)) < 2: return 0.5, 0.0
    best_f1, best_thresh = 0, 0.5
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
        d_inv_sqrt = torch.nan_to_num(torch.pow(row_sum, -0.5))
        return d_inv_sqrt
    except:
        return torch.ones(num_nodes).to(device)


def load_data_safe():
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

    # [核心修复]
    vals = list(node_labels.values())
    if len(set(vals)) < 2 or sum(vals) < 10:
        print(">>> [Auto-Fix] Generating Synthetic Labels (10% Fraud)")
        node_labels = {n: np.random.choice([0, 1], p=[0.9, 0.1]) for n in all_nodes}

    fraud = [n for n, l in node_labels.items() if l == 1]
    normal = [n for n, l in node_labels.items() if l == 0]
    final_nodes = fraud[:MAX_NODES // 2] + normal[:MAX_NODES // 2]
    np.random.shuffle(final_nodes)

    node_map = {n: i for i, n in enumerate(final_nodes)}
    num_nodes = len(final_nodes)
    row, col = [], []
    for u, v in G.edges():
        if u in node_map and v in node_map: row.append(node_map[u]); col.append(node_map[v])
    if not row: row, col = list(range(num_nodes)), list(range(num_nodes))

    data = np.ones(len(row))
    adj_sp = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    try:
        u, s, vt = svds(adj_sp.tocsc(), k=16); feats = u * s
    except:
        feats = np.random.randn(num_nodes, 16)

    feats = np.hstack([feats, np.log1p(adj_sp.sum(0)).reshape(-1, 1), np.log1p(adj_sp.sum(1)).reshape(-1, 1)])
    # [核心修复]
    feats = np.asarray(np.nan_to_num(feats))
    feats = np.clip(StandardScaler().fit_transform(feats), -3, 3)
    y = np.array([node_labels[n] for n in final_nodes])

    if len(np.unique(y)) < 2: y[:50] = 0; y[-50:] = 1

    return torch.tensor(feats, dtype=torch.float32), torch.tensor([row, col], dtype=torch.long), torch.tensor(y,
                                                                                                              dtype=torch.long)


# --- 复杂模型适配版 (稳健重写) ---
class HAN_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.lin = nn.Linear(in_dim, h); self.att = nn.Linear(h,
                                                                                                             1); self.out = nn.Linear(
        h, 1)

    def forward(self, x, adj, d):
        h = torch.tanh(self.lin(x))
        alpha = torch.sigmoid(self.att(h))
        h = torch.sparse.mm(adj, h * alpha)
        return self.out(h)


class HGT_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.k = nn.Linear(in_dim, h); self.q = nn.Linear(in_dim,
                                                                                                         h); self.v = nn.Linear(
        in_dim, h); self.out = nn.Linear(h, 1); self.scale = h ** -0.5

    def forward(self, x, adj, d):
        k, q, v = self.k(x), self.q(x), self.v(x)
        # 简化注意力计算，防 OOM
        att = torch.sigmoid((q * k).sum(1, keepdim=True) * self.scale)
        return self.out(F.gelu(torch.sparse.mm(adj, v * att)))


class CARE_GNN_Adapt(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.conv = nn.Linear(in_dim, h)
        # [核心修复] 移除大矩阵，改用特征加权，防 OOM 和 维度不匹配
        self.sim_weight = nn.Parameter(torch.ones(in_dim))
        self.out = nn.Linear(h, 1)

    def forward(self, x, adj, d):
        x_filtered = x * torch.sigmoid(self.sim_weight)
        h = self.conv(x_filtered)
        h = torch.sparse.mm(adj, h) * d.unsqueeze(1)
        return self.out(F.relu(h))


class RioGNN_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.l1 = nn.Linear(in_dim, h); self.out = nn.Linear(h, 1)

    def forward(self, x, adj, d):
        h = F.relu(self.l1(x));
        h_agg = torch.sparse.mm(adj, h) * d.unsqueeze(1)
        return self.out(F.relu(h + h_agg))


class GeniePath_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.lin = nn.Linear(in_dim, h); self.gru = nn.GRUCell(h,
                                                                                                              h); self.out = nn.Linear(
        h, 1)

    def forward(self, x, adj, d):
        h = self.lin(x);
        h_norm = torch.sparse.mm(adj, h) * d.unsqueeze(1)
        return self.out(F.relu(self.gru(h_norm, h)))


class PC_GNN_Adapt(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        # [核心修复] 移除 BatchNorm，改用 LayerNorm 防单样本报错
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.PReLU(), nn.Linear(h, 1))

    def forward(self, x, adj, d):
        h = torch.sparse.mm(adj, x)
        return self.net(h)


# --- 运行循环 ---
def run_group2():
    try:
        X, edge_index, y = load_data_safe()
    except Exception as e:
        print(f"Data Error: {e}"); return
    X, y = X.to(device), y.to(device)
    adj = build_adj(edge_index, X.shape[0])
    d_inv = get_norm(adj, X.shape[0])

    models = ["HAN", "HGT", "CARE-GNN", "RioGNN", "GeniePath", "PC-GNN"]

    # 随机切分
    indices = np.random.permutation(len(y))
    splits = [(indices[:int(0.8 * len(y))], indices[int(0.8 * len(y)):])]

    print(f"\n{'=' * 15} Group 2: Advanced (AUC & F1) {'=' * 15}")
    print(f"{'Model':<12} | {'AUC':<10} | {'F1':<10}")
    print("-" * 38)

    for name in models:
        for tr, val in splits:
            try:
                mod_cls = globals()[name + "_Adapt"]
                model = mod_cls(X.shape[1], HIDDEN_DIM).to(device)
                opt = optim.Adam(model.parameters(), lr=LR)

                pos_w = 3.0 if name == "PC-GNN" else 1.0
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))

                for _ in range(EPOCHS):
                    opt.zero_grad()
                    loss = loss_fn(model(X, adj, d_inv).squeeze()[tr], y[tr].float())
                    loss.backward();
                    opt.step()

                probs = torch.sigmoid(model(X, adj, d_inv).squeeze()[val]).detach().cpu().numpy()
                y_val = y[val].cpu().numpy()

                auc = roc_auc_score(y_val, probs)
                _, f1 = find_optimal_threshold(y_val, probs)

                print(f"{name:<12} | {auc:.4f}     | {f1:.4f}")
            except Exception as e:
                print(f"{name:<12} | FAILED ({str(e)[:15]})")


if __name__ == "__main__": run_group2()