import torch

from gamarl.models.gamarl_model import GAMARLModel


def test_model_shapes():
    B, N, T, df = 2, 7, 8, 11
    action_dim = 24

    model = GAMARLModel(
        df=df,
        d_model=64,
        n_heads=4,
        n_transformer_layers=1,
        history_len=T,
        gate_hidden=64,
        attn_hidden=64,
        gumbel_tau=1.0,
        hard_gates=False,
        gcn_hidden=64,
        gcn_activation="relu",
        q_hidden=128,
        action_dim=action_dim,
    )

    F_hist = torch.randn(B, N, T, df)
    F = torch.randn(B, N, df)
    A = (torch.rand(B, N, N) > 0.7).float()
    m = (torch.rand(B, N) > 0.2).float()

    Q, aux = model(F_hist, F, A, m)
    assert Q.shape == (B, N, action_dim)
    assert aux["gates"].shape == (B, N, N)
    assert aux["attn"].shape == (B, N, N)
    assert aux["Z"].shape[0] == B and aux["Z"].shape[1] == N
