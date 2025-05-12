import torch

# Custom quaternion geodesic distance
def quaternion_geodesic_distance(q1, q2):
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(0, 1)
    return (2 * torch.acos(dot)) ** 2

# Fast quaternion from 3x3 matrix (batched)
def rotation_matrix_to_quaternion(R):
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros((*R.shape[:-2], 4), device=R.device)
    mask = trace > 0

    S = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 3] = 0.25 * S
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
    q[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
    q[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / S
    return q  # x, y, z, w

# Quaternion distance between two hand pose frames
def quaternion_distance_torch(X, Y, args):
    Y_translated = Y + (X[0] - Y[0])

    finger_indices = {
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20]
    }

    total_distance = 0.0

    for inds in finger_indices.values():
        chain_X = X[inds]
        chain_Y = Y_translated[inds]

        diff_X = chain_X[1:] - chain_X[:-1]
        diff_Y = chain_Y[1:] - chain_Y[:-1]

        dirs_X = diff_X / (torch.norm(diff_X, dim=1, keepdim=True) + 1e-8)
        dirs_Y = diff_Y / (torch.norm(diff_Y, dim=1, keepdim=True) + 1e-8)

        # Frame computation
        ref = torch.tensor([0.0, 1.0, 0.0], device=X.device)
        refs_X = ref.expand_as(dirs_X)
        refs_Y = ref.expand_as(dirs_Y)

        parallel_X = (torch.abs(torch.sum(dirs_X * refs_X, dim=1)) > 0.99).unsqueeze(1)
        parallel_Y = (torch.abs(torch.sum(dirs_Y * refs_Y, dim=1)) > 0.99).unsqueeze(1)

        refs_X = torch.where(parallel_X, torch.tensor([1.0, 0.0, 0.0], device=X.device).expand_as(dirs_X), refs_X)
        refs_Y = torch.where(parallel_Y, torch.tensor([1.0, 0.0, 0.0], device=Y.device).expand_as(dirs_Y), refs_Y)

        x_X = torch.cross(refs_X, dirs_X, dim=1)
        x_X = x_X / (torch.norm(x_X, dim=1, keepdim=True) + 1e-8)
        y_X = torch.cross(dirs_X, x_X, dim=1)
        frames_X = torch.stack([x_X, y_X, dirs_X], dim=-1)

        x_Y = torch.cross(refs_Y, dirs_Y, dim=1)
        x_Y = x_Y / (torch.norm(x_Y, dim=1, keepdim=True) + 1e-8)
        y_Y = torch.cross(dirs_Y, x_Y, dim=1)
        frames_Y = torch.stack([x_Y, y_Y, dirs_Y], dim=-1)

        quats_X = rotation_matrix_to_quaternion(frames_X)
        quats_Y = rotation_matrix_to_quaternion(frames_Y)

        quats_X = torch.roll(quats_X, shifts=1, dims=1)
        quats_Y = torch.roll(quats_Y, shifts=1, dims=1)

        quats_X = torch.cat([quats_X, quats_X[-1:].clone()], dim=0)
        quats_Y = torch.cat([quats_Y, quats_Y[-1:].clone()], dim=0)

        lengths = torch.norm(diff_X, dim=1)
        weights = lengths ** 2 if args.get("weight_squared", False) else lengths
        weights /= weights.sum() + 1e-8

        distances = quaternion_geodesic_distance(quats_X[1:], quats_Y[1:])
        d = args.get("lambda_factor", 1.0) * torch.dot(weights, distances)
        total_distance += d

    return total_distance

