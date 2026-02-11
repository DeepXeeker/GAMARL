import torch

from gamarl.utils.tensor_ops import gcn_norm, add_self_loops


def test_gcn_norm_symmetry():
    N = 6
    A = (torch.rand(N, N) > 0.7).float()
    A = ((A + A.T) > 0).float()
    A_hat = add_self_loops(A)
    An = gcn_norm(A_hat)
    # normalized matrix should be symmetric for undirected A
    assert torch.allclose(An, An.T, atol=1e-6)
