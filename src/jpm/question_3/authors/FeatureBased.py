import torch
import torch.nn as nn
import torch.nn.functional as F


def make_valid_mask(n, lengths, device, mode="without"):
    row = torch.arange(n, device=device).unsqueeze(0)
    if mode == "with":
        return (row < lengths.unsqueeze(1)) | (row == n - 1)  # (B,n) bool
    else:
        return row < lengths.unsqueeze(1)


def masked_softmax(scores, lengths):  # scores:(B,n)
    n = scores.size(1)
    valid = make_valid_mask(n, lengths, scores.device)
    scores = scores.masked_fill(~valid, float("-inf"))
    return F.log_softmax(scores, dim=-1)


class NonlinearTransformation(nn.Module):
    def __init__(self, H, embed=128, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(embed, embed * H)
        self.fc2 = nn.Linear(embed, embed)
        self.H = H
        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.enc_norm = nn.LayerNorm(embed)

    def forward(self, X):
        B, n, _ = X.shape
        X = self.fc1(X).view(B, n, self.H, self.embed)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.enc_norm(X)
        return X


class DeepHalo(nn.Module):
    def __init__(self, n, input_dim, H, L, embed=128, dropout=0):
        super().__init__()
        self.basic_encoder = nn.Sequential(
            nn.Linear(input_dim, embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed, embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed, embed),
        )
        self.enc_norm = nn.LayerNorm(embed)
        self.aggregate_linear = nn.ModuleList([nn.Linear(embed, H) for _ in range(L)])
        self.nonlinear = nn.ModuleList(
            [NonlinearTransformation(H, embed) for _ in range(L)]
        )
        self.H = H
        self.embed = embed
        self.final_linear = nn.Linear(embed, 1)
        self.qualinear1 = nn.Linear(embed, embed)
        self.qualinear2 = nn.Linear(embed, embed)

    def forward(self, X, lengths):
        B, n, _ = X.shape
        Z = self.enc_norm(self.basic_encoder(X))
        X = Z.clone()
        for fc, nt in zip(self.aggregate_linear, self.nonlinear, strict=True):
            Z_bar = (
                (fc(Z).sum(1) / lengths.unsqueeze(1)).unsqueeze(-1).unsqueeze(1)
            )  # (B, 1, H, 1)
            phi = nt(X)
            valid = make_valid_mask(n, lengths, X.device)  # (B,n)
            # print(valid.shape, phi.shape)
            phi = phi * valid.unsqueeze(-1).unsqueeze(-1)  # (B, n, H, embed)
            Z = (phi * Z_bar).sum(2) / self.H + Z

        logits = self.final_linear(Z).squeeze(-1)  # (B,n)
        probs = masked_softmax(logits, lengths)  # log-probs
        return probs, logits
