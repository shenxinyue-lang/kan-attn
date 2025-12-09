# -*- coding: utf-8 -*-
"""
KAN + A(+样条核) & D(+样条温度/偏置) for Tabular Regression
- 支持融合：row_temp / sym_temp / logit_bias / row_temp+logit_bias / topk_temp(流式Top-k，避免[L,L])
- 训练管线：分层切分 -> 标准化 -> (可选)PCA -> y 标准化训练/评估反标准化
"""

import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score


# ================= 路径 =================
PATH_X_GLOBAL = "features_graph_measures_X_global.npy"   # (S, 48)
PATH_X_LOCAL  = "features_graph_measures_X_local.npy"    # (S, 1216)
PATH_Y        = "features_graph_measures_y.npy"          # (S,)


# ================= 训练配置 =================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
PRINT_EVERY = 10

USE_PCA = False
PCA_DIM = 256
DEBUG_MAX_TRAIN_SAMPLES = None

# ================= 模型超参 =================
WIDTHS   = [256, 256, 128]
NBINS    = 8
RANGE    = 3.0
DROPOUT  = 0.10
LN_EPS   = 1e-5

EPOCHS   = 400
PATIENCE = 40
BATCH    = 64
LR       = 1e-3
WD       = 1e-4
CLIP_NORM = 1.0

# 注意力
ATTN_AT   = 2
ATTN_DIM  = 96
N_HEADS   = 3
FUSION_MODE = "topk_temp"   # "row_temp"|"sym_temp"|"logit_bias"|"row_temp+logit_bias"|"topk_temp"
TOPK_K      = 32
COL_CHUNK   = 256


# ================= 工具 =================
def seed_everything(seed=SEED):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


def stratified_split_by_age(y, test_size=0.2, val_size=0.1, seed=SEED, n_bins=10):
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.clip(np.digitize(y, qs), 0, n_bins - 1)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(y))
    trval_idx, te_idx = next(sss1.split(idx, bins))

    bins_trval = bins[trval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, val_idx = next(sss2.split(trval_idx, bins_trval))

    return trval_idx[tr_idx], trval_idx[val_idx], te_idx


# ================= KAN =================
class KANSplineAct(nn.Module):
    def __init__(self, width: int, nbins: int = NBINS, value_range: float = RANGE):
        super().__init__()
        self.width = width
        self.nbins = nbins
        self.range = value_range

        self.coeff = nn.Parameter(torch.zeros(width, nbins))
        with torch.no_grad():
            self.coeff[:, nbins // 2].fill_(1.0)

        centers = torch.linspace(-self.range, self.range, nbins)
        self.register_buffer("centers", centers)
        self.register_buffer("delta", torch.tensor((2 * self.range) / (nbins - 1)))

    def _hat(self, z: torch.Tensor) -> torch.Tensor:
        dist = torch.abs(z.unsqueeze(-1) - self.centers.view(1, 1, -1))
        return torch.clamp(1.0 - dist / (self.delta + 1e-6), min=0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self._hat(z)
        return torch.sum(h * self.coeff.unsqueeze(0), dim=-1)


class KANBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nbins: int = NBINS,
        value_range: float = RANGE,
        dropout: float = DROPOUT,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act    = KANSplineAct(out_dim, nbins=nbins, value_range=value_range)
        self.proj   = nn.Linear(out_dim, out_dim)
        self.dropout= nn.Dropout(dropout)

        self.res_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        self.use_ln = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(out_dim, eps=LN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        z = self.act(z)
        z = self.proj(z)
        z = self.dropout(z)
        out = self.res_proj(x) + z
        if self.use_ln:
            out = self.ln(out)
        return out


# ================= A + D 注意力 =================
class _Spline1D(nn.Module):
    def __init__(self, nbins=NBINS, value_range=RANGE, per_head: int = 1):
        super().__init__()
        self.nbins = nbins
        self.range = value_range

        self.coeff = nn.Parameter(torch.zeros(per_head, nbins))
        with torch.no_grad():
            self.coeff[:, nbins // 2].fill_(1.0)

        centers = torch.linspace(-self.range, self.range, nbins)
        self.register_buffer("centers", centers)
        self.register_buffer("delta", torch.tensor((2 * self.range) / (nbins - 1)))

    def _hat(self, u):
        dist = torch.abs(u.unsqueeze(-1) - self.centers.view(*(1,) * u.dim(), -1))
        return torch.clamp(1.0 - dist / (self.delta + 1e-6), min=0.0)

    def forward(self, u, head_axis: int = None):
        h = self._hat(u)  # [..., nbins]
        if head_axis is None:
            coeff = self.coeff.mean(dim=0, keepdim=True)
            return torch.sum(h * coeff, dim=-1)

        H = self.coeff.shape[0]
        shape = [1] * (u.dim() + 1)
        shape[head_axis] = H
        shape[-1] = self.nbins
        coeff = self.coeff.view(*shape)
        return torch.sum(h * coeff, dim=-1)


class _SplineKernelAttention(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        heads: int = N_HEADS,
        nbins: int = NBINS,
        value_range: float = RANGE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert attn_dim % heads == 0
        self.H = heads
        self.d = attn_dim // heads

        self.q = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, attn_dim, bias=False)

        self.q1 = nn.Linear(self.d, 1, bias=False)
        self.k1 = nn.Linear(self.d, 1, bias=False)

        self.kernel = _Spline1D(nbins=nbins, value_range=value_range, per_head=heads)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(attn_dim, attn_dim, bias=False)

    def qkv(self, tok):
        B, L, D = tok.shape
        H, d = self.H, self.d

        q = self.q(tok).view(B, L, H, d).transpose(1, 2)
        k = self.k(tok).view(B, L, H, d).transpose(1, 2)
        v = self.v(tok).view(B, L, H, d).transpose(1, 2)

        uq = self.q1(q).squeeze(-1)
        uk = self.k1(k).squeeze(-1)
        return q, k, v, uq, uk

    def kernel_logits(self, uq, uk):
        diff = uq.unsqueeze(-1) - uk.unsqueeze(-2)
        return self.kernel(diff, head_axis=1)

    def ctx_from_logits(self, K, v):
        attn = torch.softmax(K, dim=-1)
        ctx = attn @ v
        B, H, L, d = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, L, H * d)
        return self.out(self.drop(ctx))

    @torch.no_grad()
    def _running_topk_merge(self, prev_vals, prev_idx, new_vals, new_idx, k):
        if prev_vals is None:
            return new_vals, new_idx
        cat_vals = torch.cat([prev_vals, new_vals], dim=-1)
        cat_idx  = torch.cat([prev_idx,  new_idx ], dim=-1)
        sel_vals, sel = torch.topk(cat_vals, k=k, dim=-1)
        sel_idx  = torch.gather(cat_idx, -1, sel)
        return sel_vals, sel_idx

    def stream_topk_ctx(
        self,
        uq,
        uk,
        v,
        tau_row,
        k_top=32,
        col_chunk=256,
        bias_fn=None,
        bias_scale=None,
    ):
        """
        流式 Top-k：不构造 [L,L]
        uq,uk:[B,H,L]  v:[B,H,L,d]  tau_row:[B,L]
        返回 ctx:[B,L,H*d]
        """
        B, H, L = uq.shape
        d = v.shape[-1]

        top_vals, top_idx = None, None

        for s in range(0, L, col_chunk):
            e = min(s + col_chunk, L)
            uk_c = uk[..., s:e]                                  # [B,H,Lc]
            diff = uq.unsqueeze(-1) - uk_c.unsqueeze(-2)         # [B,H,L,Lc]
            Kc = self.kernel(diff, head_axis=1)                  # [B,H,L,Lc]
            Kc = Kc / (tau_row.unsqueeze(1).unsqueeze(-1) + 1e-6)
            if bias_fn is not None and (bias_scale is not None):
                Bsp = bias_fn(uq, uk_c)
                Kc = Kc + bias_scale.tanh() * Bsp

            k_here = min(k_top, e - s)
            vals_blk, idx_blk = torch.topk(Kc, k=k_here, dim=-1)  # [B,H,L,k_here]
            idx_blk = idx_blk + s
            top_vals, top_idx = self._running_topk_merge(
                top_vals, top_idx, vals_blk, idx_blk, k_top
            )

        # ===== 修正后的 gather 实现（维度对齐）=====
        attn = torch.softmax(top_vals, dim=-1)                   # [B,H,L,k]
        B, H, L, k = attn.shape
        d = v.shape[-1]
        idx_exp = top_idx.unsqueeze(-1).expand(B, H, L, k, d)    # [B,H,L,k,d]
        v_exp  = v.unsqueeze(3).expand(B, H, L, k, d)            # [B,H,L,k,d]
        v_sel  = torch.gather(v_exp, 2, idx_exp)                 # [B,H,L,k,d]
        ctx = (attn.unsqueeze(-1) * v_sel).sum(dim=-2)           # [B,H,L,d]
        ctx = ctx.transpose(1, 2).reshape(B, L, H * d)           # [B,L,H*d]
        return self.out(ctx)


class _SplineTemperature(nn.Module):
    def __init__(
        self,
        attn_dim: int,
        nbins: int = NBINS,
        value_range: float = RANGE,
        min_tau: float = 0.05,
    ):
        super().__init__()
        self.u_proj = nn.Linear(attn_dim, 1, bias=False)
        self.spline = _Spline1D(nbins=nbins, value_range=value_range, per_head=1)
        self.min_tau = min_tau

    def tau(self, tok):
        u = self.u_proj(tok).squeeze(-1)
        f = self.spline(u)
        return torch.nn.functional.softplus(f) + self.min_tau

    def bias(self, uq, uk):
        diff = torch.abs(uq.unsqueeze(-1) - uk.unsqueeze(-2))
        return self.spline(diff, head_axis=1)


class AttnMixBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
        nbins: int = NBINS,
        value_range: float = RANGE,
        fusion_mode: str = FUSION_MODE,
        topk_k: int = TOPK_K,
        col_chunk: int = COL_CHUNK,
    ):
        super().__init__()
        assert attn_dim % n_heads == 0
        self.fusion = fusion_mode
        self.topk_k = topk_k
        self.col_chunk = col_chunk

        self.pre   = nn.Linear(in_dim, out_dim)
        self.embed = nn.Linear(1, attn_dim)
        self.ln1   = nn.LayerNorm(attn_dim)

        self.attnA = _SplineKernelAttention(
            attn_dim,
            heads=n_heads,
            nbins=nbins,
            value_range=value_range,
            dropout=dropout,
        )

        self.tau_q = _SplineTemperature(
            attn_dim, nbins=nbins, value_range=value_range, min_tau=0.05
        )
        self.tau_k = _SplineTemperature(
            attn_dim, nbins=nbins, value_range=value_range, min_tau=0.05
        )
        self.biasD = _SplineTemperature(attn_dim, nbins=nbins, value_range=value_range)
        self.bias_scale = nn.Parameter(torch.tensor(0.0))

        self.ln2 = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, 4 * attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * attn_dim, attn_dim),
        )
        self.drop = nn.Dropout(dropout)

        self.proj_out = nn.Linear(attn_dim, 1)
        self.res_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.ln_feat  = nn.LayerNorm(out_dim, eps=LN_EPS)

    def _apply_fusion_logits(self, K, tok, uq, uk):
        if self.fusion == "row_temp":
            tau = self.tau_q.tau(tok)
            K = K / (tau.unsqueeze(1).unsqueeze(-1) + 1e-6)
        elif self.fusion == "sym_temp":
            tq = self.tau_q.tau(tok)
            tk = self.tau_k.tau(tok)
            denom = torch.sqrt(
                (tq.unsqueeze(1).unsqueeze(-1)) * (tk.unsqueeze(1).unsqueeze(-2))
            ) + 1e-6
            K = K / denom
        elif self.fusion == "logit_bias":
            Bsp = self.biasD.bias(uq, uk)
            K   = K + self.bias_scale.tanh() * Bsp
        elif self.fusion == "row_temp+logit_bias":
            tau = self.tau_q.tau(tok)
            K   = K / (tau.unsqueeze(1).unsqueeze(-1) + 1e-6)
            Bsp = self.biasD.bias(uq, uk)
            K   = K + self.bias_scale.tanh() * Bsp
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.pre(x)                               # [B,L]
        tok = self.embed(v.unsqueeze(-1))             # [B,L,D]
        h1 = self.ln1(tok)
        q, k, v_emb, uq, uk = self.attnA.qkv(h1)      # q,k,v:[B,H,L,d]; uq,uk:[B,H,L]

        if self.fusion == "topk_temp":
            tau = self.tau_q.tau(h1)                  # [B,L]
            attn_out = self.attnA.stream_topk_ctx(
                uq,
                uk,
                v_emb,
                tau_row=tau,
                k_top=self.topk_k,
                col_chunk=self.col_chunk,
                bias_fn=None,
                bias_scale=None,
            )
        else:
            K = self.attnA.kernel_logits(uq, uk)
            K = self._apply_fusion_logits(K, h1, uq, uk)
            attn_out = self.attnA.ctx_from_logits(K, v_emb)

        y = tok + self.drop(attn_out)
        y = y + self.drop(self.ffn(self.ln2(y)))
        v2 = self.proj_out(y).squeeze(-1)             # [B,L]
        out = self.res_proj(x) + v2
        return self.ln_feat(out)


# ================= 模型封装 =================
class KANRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        widths = WIDTHS,
        nbins: int = NBINS,
        value_range: float = RANGE,
        dropout: float = DROPOUT,
        attn_at: int = ATTN_AT,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        fusion_mode: str = FUSION_MODE,
        topk_k: int = TOPK_K,
        col_chunk: int = COL_CHUNK,
    ):
        super().__init__()
        dims = [in_dim] + list(widths)
        blocks = []
        for i in range(len(dims) - 1):
            if i == attn_at:
                blocks.append(
                    AttnMixBlock(
                        dims[i],
                        dims[i + 1],
                        attn_dim=attn_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        nbins=nbins,
                        value_range=value_range,
                        fusion_mode=fusion_mode,
                        topk_k=topk_k,
                        col_chunk=col_chunk,
                    )
                )
            else:
                blocks.append(
                    KANBlock(
                        dims[i],
                        dims[i + 1],
                        nbins=nbins,
                        value_range=value_range,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        y = self.head(h).squeeze(-1)
        return y


# ================= 训练/评估 =================
def make_loader(X: np.ndarray, y: np.ndarray, bs: int, shuffle: bool) -> DataLoader:
    tensX = torch.from_numpy(X).to(torch.float32)
    tensY = torch.from_numpy(y).to(torch.float32)
    return DataLoader(
        TensorDataset(tensX, tensY),
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
    )


def train_one_epoch(model, dl, opt, scaler, criterion, use_amp=True, clip_norm=0.0):
    model.train()
    total = 0.0
    for xb, yb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            if clip_norm and clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()

        total += loss.item() * xb.size(0)

    return total / len(dl.dataset)


@torch.no_grad()
def infer_epoch(model, dl, criterion, use_amp=False) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    preds, trues = [], []
    for xb, yb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                pred = model(xb)
                loss = criterion(pred, yb)
        else:
            pred = model(xb)
            loss = criterion(pred, yb)

        total += loss.item() * xb.size(0)
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total / len(dl.dataset), preds, trues


def main():
    print(f"[INFO] device={DEVICE} | AMP={USE_AMP} | seed={SEED}")
    print(
        f"[INFO] ATTN_AT={ATTN_AT} | FUSION_MODE={FUSION_MODE} | "
        f"ATTN_DIM={ATTN_DIM} | HEADS={N_HEADS} | NBINS={NBINS}"
    )

    # 1) 读数据
    for p in [PATH_X_GLOBAL, PATH_X_LOCAL, PATH_Y]:
        if not Path(p).exists():
            raise FileNotFoundError(f"找不到文件：{p}")

    Xg = np.load(PATH_X_GLOBAL).astype(np.float32)
    Xl = np.load(PATH_X_LOCAL ).astype(np.float32)
    y  = np.load(PATH_Y).astype(float).ravel()

    assert Xg.shape[0] == Xl.shape[0] == y.shape[0]
    S = y.shape[0]

    # 2) 分层切分
    tr_idx, val_idx, te_idx = stratified_split_by_age(y)
    take = lambda a, idx: a[idx]

    Xg_tr, Xg_val, Xg_te = take(Xg, tr_idx), take(Xg, val_idx), take(Xg, te_idx)
    Xl_tr, Xl_val, Xl_te = take(Xl, tr_idx), take(Xl, val_idx), take(Xl, te_idx)
    y_tr,  y_val,  y_te  = take(y, tr_idx),  take(y, val_idx),  take(y, te_idx)

    # 3) 可选调试子集
    if DEBUG_MAX_TRAIN_SAMPLES is not None and len(y_tr) > DEBUG_MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(SEED)
        sub = rng.choice(len(y_tr), size=DEBUG_MAX_TRAIN_SAMPLES, replace=False)
        Xg_tr, Xl_tr, y_tr = Xg_tr[sub], Xl_tr[sub], y_tr[sub]

    # 4) 标准化
    gsc = StandardScaler().fit(Xg_tr)
    lsc = StandardScaler().fit(Xl_tr)

    Xg_tr, Xg_val, Xg_te = (
        gsc.transform(Xg_tr),
        gsc.transform(Xg_val),
        gsc.transform(Xg_te),
    )
    Xl_tr, Xl_val, Xl_te = (
        lsc.transform(Xl_tr),
        lsc.transform(Xl_val),
        lsc.transform(Xl_te),
    )

    # 5) 拼接 + (可选)PCA
    X_tr  = np.concatenate([Xl_tr, Xg_tr], axis=1).astype(np.float32)
    X_val = np.concatenate([Xl_val, Xg_val], axis=1).astype(np.float32)
    X_te  = np.concatenate([Xl_te, Xg_te], axis=1).astype(np.float32)
    print(f"[INFO] S={S} | D_full={X_tr.shape[1]} | PCA={USE_PCA} (dim={PCA_DIM})")

    if USE_PCA:
        pca_dim = min(PCA_DIM, X_tr.shape[1])
        pca = PCA(n_components=pca_dim, random_state=SEED)
        X_tr  = pca.fit_transform(X_tr).astype(np.float32)
        X_val = pca.transform(X_val).astype(np.float32)
        X_te  = pca.transform(X_te ).astype(np.float32)
        print(f"[INFO] PCA 后维度: {X_tr.shape[1]}")

    D = X_tr.shape[1]

    # 6) y 标准化
    y_mu, y_sigma = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_z  = (y_tr  - y_mu) / y_sigma
    y_val_z = (y_val - y_mu) / y_sigma
    y_te_z  = (y_te  - y_mu) / y_sigma

    # 7) DataLoader
    dl_tr  = make_loader(X_tr,  y_tr_z,  BATCH, True)
    dl_val = make_loader(X_val, y_val_z, BATCH, False)
    dl_te  = make_loader(X_te,  y_te_z,  BATCH, False)

    # 8) 模型/优化器
    model = KANRegressor(
        in_dim=D,
        widths=WIDTHS,
        nbins=NBINS,
        value_range=RANGE,
        dropout=DROPOUT,
        attn_at=ATTN_AT,
        attn_dim=ATTN_DIM,
        n_heads=N_HEADS,
        fusion_mode=FUSION_MODE,
        topk_k=TOPK_K,
        col_chunk=COL_CHUNK,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.98),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    criterion = nn.MSELoss()

    # 9) 训练（早停）
    best = {"mse": float("inf"), "state": None, "es": 0}
    for ep in range(1, EPOCHS + 1):
        tr_mse = train_one_epoch(
            model,
            dl_tr,
            optimizer,
            scaler,
            criterion,
            use_amp=USE_AMP,
            clip_norm=CLIP_NORM,
        )
        val_mse, vpred_z, vtrue_z = infer_epoch(
            model, dl_val, criterion, use_amp=False
        )
        vpred = vpred_z * y_sigma + y_mu
        vtrue = vtrue_z * y_sigma + y_mu
        val_mae = mean_absolute_error(vtrue, vpred)
        val_r2  = r2_score(vtrue, vpred)

        if val_mse < best["mse"]:
            best.update(
                mse=val_mse,
                state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                es=0,
            )
        else:
            best["es"] += 1

        if ep % PRINT_EVERY == 0 or ep == 1:
            print(
                f"[EP {ep:03d}] train_MSE={tr_mse:.4f} | "
                f"VAL_MAE={val_mae:.4f} | VAL_R2={val_r2:.3f} | es={best['es']}"
            )

        if best["es"] >= PATIENCE:
            print(f"[EARLY STOP] at epoch {ep}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        model.to(DEVICE)

    # 10) 最终评估
    _, vpred_z, vtrue_z = infer_epoch(model, dl_val, criterion, use_amp=False)
    _, tpred_z, ttrue_z = infer_epoch(model, dl_te,  criterion, use_amp=False)

    vpred = vpred_z * y_sigma + y_mu
    vtrue = vtrue_z * y_sigma + y_mu
    tpred = tpred_z * y_sigma + y_mu
    ttrue = ttrue_z * y_sigma + y_mu

    print(
        f"[FINAL] VALID: MAE={mean_absolute_error(vtrue, vpred):.3f} | "
        f"R2={r2_score(vtrue, vpred):.3f}"
    )
    print(
        f"[FINAL] TEST : MAE={mean_absolute_error(ttrue, tpred):.3f} | "
        f"R2={r2_score(ttrue, tpred):.3f}"
    )
    print("[DONE]")


if __name__ == "__main__":
    main()
