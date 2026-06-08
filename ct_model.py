"""Custom Transformer architecture (Track B) — shared by the trainer
(scripts/train_transformer.py) and the live server (inference.py) so the saved
state_dict keys always match. Mirrors notebooks/AIRD_Project_Notebook.ipynb."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.bernoulli(
            torch.full((x.shape[0],) + (1,) * (x.ndim - 1), keep, device=x.device))
        return x * mask / keep


class MHSA(nn.Module):
    def __init__(self, d, h=6, p=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)

    def forward(self, x, pad_mask=None):
        return self.mha(x, x, x, key_padding_mask=pad_mask)[0]


class Block(nn.Module):
    def __init__(self, d, h, ff, p=0.1, dp=0.0):
        super().__init__()
        self.attn = MHSA(d, h, p)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(p); self.drop2 = nn.Dropout(p); self.drop_path = DropPath(dp)

    def forward(self, x, pad_mask=None):
        x = self.norm1(x + self.drop_path(self.drop1(self.attn(x, pad_mask))))
        x = self.norm2(x + self.drop_path(self.drop2(self.ffn(x))))
        return x


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, num_classes, embed_dim=300, num_heads=6,
                 ff_dim=1200, num_blocks=4, dropout=0.20, drop_path_rate=0.15,
                 pad_idx=0, glove_matrix=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if glove_matrix is not None:
            self.tok_emb.weight.data.copy_(torch.tensor(glove_matrix, dtype=torch.float32))
        else:
            nn.init.uniform_(self.tok_emb.weight, -0.05, 0.05)
        self.tok_emb.weight.data[pad_idx].fill_(0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.emb_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, ff_dim, dropout, dpr[i])
                                     for i in range(num_blocks)])
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.drop_fc = nn.Dropout(dropout + 0.1)
        self.fc2 = nn.Linear(embed_dim // 2, num_classes)

    def forward(self, x):
        pad_mask = (x == self.pad_idx)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        h = self.pool_norm(h.mean(dim=1))
        h = self.drop_fc(torch.relu(self.fc1(h)))
        return self.fc2(h)


def build_from_config(cfg, glove_matrix=None):
    return CustomTransformer(
        cfg["vocab_size"], cfg["max_len"], cfg["num_classes"],
        embed_dim=cfg.get("embed_dim", 300), num_heads=cfg.get("num_heads", 6),
        ff_dim=cfg.get("ff_dim", 1200), num_blocks=cfg.get("num_blocks", 4),
        glove_matrix=glove_matrix)
