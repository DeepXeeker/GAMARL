import torch

from gamarl.models.comm import GatedAttentionComm


def test_gates_in_range():
    B, N, D = 3, 5, 16
    comm = GatedAttentionComm(D, gate_hidden=32, attn_hidden=32, gumbel_tau=1.0, hard_gates=False)
    h = torch.randn(B, N, D)
    A = (torch.rand(B, N, N) > 0.5).float()
    c, g, a = comm(h, A)
    assert torch.all(g >= 0.0) and torch.all(g <= 1.0)
    # attention rows sum to 1 over masked entries (approximately)
    row_sum = a.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)
