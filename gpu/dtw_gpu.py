import torch

def dtw_gpu(seqA, seqB, distance_fn, distance_args):
    m, n = seqA.shape[0], seqB.shape[0]
    C = torch.zeros((m, n), device=seqA.device)

    for i in range(m):
        for j in range(n):
            C[i, j] = distance_fn(seqA[i], seqB[j], distance_args)

    D = torch.full((m+1, n+1), float("inf"), device=seqA.device)
    D[0, 0] = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            D[i, j] = C[i-1, j-1] + torch.min(torch.stack([
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
            ]))

    return D[m, n]

