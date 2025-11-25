import torch, torch.nn as nn, torch.nn.functional as F

class ExaResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_main = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_act  = nn.Linear(input_dim,  hidden_dim, bias=False)
    def forward(self, z_prev, z0):
        return self.linear_main(z_prev * self.linear_act(z0)) + z_prev

class QuaResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
    def forward(self, x):
        return self.linear(x.pow(2)) + x

class MainNetwork(nn.Module):
    def __init__(self, opt_size:int, depth:int,
                 resnet_width:int, block_types):
        super().__init__()
        assert len(block_types) == depth-1
        self.in_lin  = nn.Linear(opt_size, resnet_width, bias=False)
        self.out_lin = nn.Linear(resnet_width, opt_size,  bias=False)
        self.blocks  = nn.ModuleList()
        for t in block_types:
            self.blocks.append(
                ExaResBlock(opt_size, resnet_width) if t=="exa"
                else QuaResBlock(resnet_width)      if t=="qua"
                else (_ for _ in ()).throw(ValueError(f"Unknown {t}"))
            )
    def forward(self, e):
        mask = e == 1
        e0   = e.clone()
        e    = self.in_lin(e)
        for b in self.blocks:
            e = b(e, e0) if isinstance(b, ExaResBlock) else b(e)
        logits = self.out_lin(e).masked_fill(~mask, float("-inf"))
        return F.softmax(logits, dim=-1), logits


