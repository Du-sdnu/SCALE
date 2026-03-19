import os, gc, time, warnings, pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# --- 配置 ---
FILE_NAME = 'dgraphfin.npz'
MAX_NEIGHBORS = 20
HIDDEN_DIM = 128
EPOCHS = 50
LR = 0.005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42);
np.random.seed(42)


# --- 核心工具 ---
def get_best_f1(y_true, y_probs):
    if np.isnan(y_probs).any() or len(np.unique(y_true)) < 2: return 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    return np.max(f1s)


def build_adj(edge_index, num_nodes):
    edge_index = edge_index.to(device)
    src, dst = edge_index
    src = torch.cat([src, torch.arange(num_nodes).to(device)])
    dst = torch.cat([dst, torch.arange(num_nodes).to(device)])
    indices = torch.stack([src, dst])
    values = torch.ones(indices.shape[1]).to(device)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5).nan_to_num(0.0)
    return adj, d_inv_sqrt


def load_data_turbo():
    if not os.path.exists(FILE_NAME): raise FileNotFoundError(f"{FILE_NAME} missing")
    print(f">>> Loading {FILE_NAME} (Turbo Mode)...")
    data = np.load(FILE_NAME)
    x_raw, y_raw, edges_raw = data['x'], data['y'], data['edge_index']
    if edges_raw.shape[0] == 2: edges_raw = edges_raw.T
    edges_raw = edges_raw[:, :2]

    fraud_indices = np.where(y_raw == 1)[0]
    normal_indices = np.where(y_raw == 0)[0]

    adj_dict = {}
    mask_fraud = np.isin(edges_raw[:, 0], fraud_indices) | np.isin(edges_raw[:, 1], fraud_indices)
    for u, v in edges_raw[mask_fraud]:
        if u not in adj_dict: adj_dict[u] = []
        if v not in adj_dict: adj_dict[v] = []
        adj_dict[u].append(v);
        adj_dict[v].append(u)

    gang_nodes = set(fraud_indices)
    for seed in fraud_indices:
        if seed in adj_dict:
            neighbors = adj_dict[seed]
            if len(neighbors) > MAX_NEIGHBORS: neighbors = np.random.choice(neighbors, MAX_NEIGHBORS, replace=False)
            gang_nodes.update(neighbors)

    needed_normal = len(gang_nodes) // 2
    pure_normal = np.random.choice(normal_indices, needed_normal, replace=False)
    final_nodes = np.unique(np.concatenate([list(gang_nodes), pure_normal]))
    final_nodes = final_nodes[(y_raw[final_nodes] == 0) | (y_raw[final_nodes] == 1)]

    id_map = {old: new for new, old in enumerate(final_nodes)}
    mask_u = np.isin(edges_raw[:, 0], final_nodes)
    mask_v = np.isin(edges_raw[:, 1], final_nodes)
    sub_edges = edges_raw[mask_u & mask_v]
    u_map = np.vectorize(id_map.get)(sub_edges[:, 0])
    v_map = np.vectorize(id_map.get)(sub_edges[:, 1])
    edge_index = np.stack([u_map, v_map])

    num_sub = len(final_nodes)
    data_ones = np.ones(edge_index.shape[1])
    adj_sp = sp.coo_matrix((data_ones, (edge_index[0], edge_index[1])), shape=(num_sub, num_sub))
    deg = np.log1p(np.array(adj_sp.sum(0)).flatten()).reshape(-1, 1)

    x_sub = x_raw[final_nodes]
    x_aug = np.hstack([x_sub, deg])
    x_aug = RobustScaler().fit_transform(x_aug)
    return torch.tensor(x_aug, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long), torch.tensor(
        y_raw[final_nodes], dtype=torch.long)


# --- SOTA 模型 (增强版) ---
class HAN_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.lin = nn.Linear(in_dim, h); self.att = nn.Linear(h,
                                                                                                             1); self.out = nn.Linear(
        h, 1)

    def forward(self, x, adj, d):
        h = torch.tanh(self.lin(x));
        alpha = torch.sigmoid(self.att(h));
        h = torch.sparse.mm(adj, h * alpha)
        return self.out(h)


class HGT_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.k = nn.Linear(in_dim, h); self.q = nn.Linear(in_dim,
                                                                                                         h); self.v = nn.Linear(
        in_dim, h); self.out = nn.Linear(h, 1); self.scale = h ** -0.5

    def forward(self, x, adj, d):
        k, q, v = self.k(x), self.q(x), self.v(x)
        att = torch.sigmoid((q * k).sum(1, keepdim=True) * self.scale)
        return self.out(F.gelu(torch.sparse.mm(adj, v * att)))


class CARE_GNN_Adapt(nn.Module):
    def __init__(self, in_dim, h): super().__init__(); self.conv = nn.Linear(in_dim, h); self.sim = nn.Parameter(
        torch.ones(in_dim)); self.out = nn.Linear(h, 1)

    def forward(self, x, adj, d):
        x_filtered = x * torch.sigmoid(self.sim);
        h = self.conv(x_filtered);
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
    def __init__(self, in_dim, h): super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h),
                                                                                nn.PReLU(), nn.Linear(h, 1))

    def forward(self, x, adj, d): return self.net(torch.sparse.mm(adj, x))


# --- 运行循环 ---
def run_group2_turbo():
    try:
        X, edge_index, y = load_data_turbo()
    except Exception as e:
        print(f"Err: {e}"); return
    X_gpu, y_gpu = X.to(device), y.to(device)
    adj, d_inv = build_adj(edge_index, X.shape[0])

    models = ["HAN", "HGT", "CARE-GNN", "RioGNN", "GeniePath", "PC-GNN"]
    idx = torch.randperm(len(y));
    sp = int(0.8 * len(y));
    tr_idx, val_idx = idx[:sp], idx[sp:]

    print("\n=== Group 2 Turbo (Competitive SOTA) ===")
    print(f"{'Model':<12} | {'AUC':<8} | {'F1 (Best)':<8}")
    print("-" * 35)

    for name in models:
        try:
            mod_cls = globals()[name + "_Adapt"]
            model = mod_cls(X.shape[1], HIDDEN_DIM).to(device)
            opt = optim.Adam(model.parameters(), lr=LR)

            # PC-GNN 给予更高权重模拟 Pick
            pos_w = 3.0 if name == "PC-GNN" else 2.0
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))

            for _ in range(EPOCHS):
                opt.zero_grad()
                loss = loss_fn(model(X_gpu, adj, d_inv).squeeze()[tr_idx], y_gpu.float()[tr_idx])
                loss.backward();
                opt.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_gpu, adj, d_inv).squeeze()[val_idx]).cpu().numpy()

            y_val = y.numpy()[val_idx]
            auc = roc_auc_score(y_val, probs)
            f1 = get_best_f1(y_val, probs)
            print(f"{name:<12} | {auc:.4f}   | {f1:.4f}")

        except Exception as e:
            print(f"{name:<12} | FAILED")


if __name__ == "__main__": run_group2_turbo()