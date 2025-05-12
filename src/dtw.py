import numpy as np
def dtw(seqA, seqB, distance_fn, distance_args={}):
    """
    Procrustes-DTW distance between two sequences of landmarks.

    Parameters
    ----------
    seqA : ndarray of shape (m, no_landmarks * dim)
        Each frame is a flat array of landmarks.
    seqB : ndarray of shape (n, no_landmarks * dim)
        The second sequence of frames.
    distance_fn : callable
        Function to compute the distance between two frames.
        It should take three arguments: the first frame, the second frame and its arguments.
    distance_args : dict
        Arguments for the distance function.
        It should be a dictionary with the keys and values that the distance function expects.
        For example, if the distance function is `euclidean`, the arguments could be:
        {'normalize': True} or {'normalize': False}.
    Returns
    -------
    dtw_cost : float
        Procrustes-DTW alignment cost.
    """

    m = len(seqA)
    n = len(seqB)


    assert seqA.shape[1] == 21 and seqB.shape[2] == 3, "Invalid shape for seqA"
    assert seqA.shape[1] == 21 and seqB.shape[2] == 3, "Invalid shape for seqB"

    # cost matrix
    C = np.zeros((m, n), dtype=float)

    for i in range(m):
        X_i = seqA[i]
        for j in range(n):
            Y_j = seqB[j]
            C[i, j] = distance_fn(X_i, Y_j, distance_args)

    # DTW matrix
    D = np.full((m+1, n+1), np.inf, dtype=float)
    D[0,0] = 0.0

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_ij = C[i-1, j-1]
            D[i, j] = cost_ij + min(
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
            )

    return D[m, n]