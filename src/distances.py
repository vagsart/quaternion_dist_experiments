import numpy as np
from scipy.spatial.transform import Rotation as R
from numba import njit

def procrustes_alignment(X, Y, args={'scale': True, 'rotation': True, 'mean': True}):
    """
    Helper that performs the core alignment steps.
    Returns:
      X0      : Aligned X (centered and scaled)
      Y0_rot  : Transformed Y (after rotation, etc.)
      R       : Rotation matrix used for alignment
      dist    : Basic Procrustes distance computed from the aligned frames
      add_to_dist : Additional term from the rotation (if applicable)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)


    # 1. Translation (centroid to origin)
    if args.get('mean', True):
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X0 = X - X_mean
        Y0 = Y - Y_mean
    else:
        X0 = X.copy()
        Y0 = Y.copy()

    # 2. Scale
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    if args.get('scale', True):
        X0 = X0 / (normX + 1e-15)
        Y0 = Y0 / (normY + 1e-15)

    # 3. Find best rotation
    if args.get('rotation', True):
        A = X0.T @ Y0
        U, s, Vt = np.linalg.svd(A)
        R = U @ Vt

        # Reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        
        # 4. Rotate
        Y0_rot = Y0 @ R
        add_to_dist = np.linalg.norm(np.log(1e-16 + np.abs(R - np.eye(3))), 'fro')
        # theta = np.arccos((np.trace(R) - 1) / 2)  # Rotation angle in radians (maybe this could be placed in add_to_dist)

    else:
        R = np.eye(X0.shape[1])
        Y0_rot = Y0.copy()
        add_to_dist = 0.0

    diff = X0 - Y0_rot
    dist = np.sum(np.linalg.norm(diff, axis=1))
    return dist



def quaternion_distance(X, Y, args):
    base_translation = X[0] - Y[0]
    Y_translated = Y + base_translation

    finger_indices = {
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20]
    }

    total_distance = 0.0
    Y_visualization = np.zeros_like(Y)
    Y_visualization[0] = X[0]

    for finger, inds in finger_indices.items():
        chain_X = X[inds]
        chain_Y = Y_translated[inds]

        diff_X = np.diff(chain_X, axis=0)
        diff_Y = np.diff(chain_Y, axis=0)

        norm_X = R.from_matrix(frames_X).as_quat()
        norm_Y = R.from_matrix(frames_Y).as_quat()

        dirs_X = diff_X / (norm_X + 1e-8)
        dirs_Y = diff_Y / (norm_Y + 1e-8)

        # Inlined compute_frames (dirs_X)
        ref = np.array([0.0, 1.0, 0.0])
        refs_X = np.tile(ref, (dirs_X.shape[0], 1))
        parallel_X = np.abs(np.einsum('ij,ij->i', dirs_X, refs_X)) > 0.99
        refs_X[parallel_X] = np.array([1.0, 0.0, 0.0])
        x_X = np.cross(refs_X, dirs_X)
        x_X = x_X / (np.linalg.norm(x_X, axis=1, keepdims=True) + 1e-8)
        y_X = np.cross(dirs_X, x_X)
        frames_X = np.stack([x_X, y_X, dirs_X], axis=-1)

        # Inlined compute_frames (dirs_Y)
        refs_Y = np.tile(ref, (dirs_Y.shape[0], 1))
        parallel_Y = np.abs(np.einsum('ij,ij->i', dirs_Y, refs_Y)) > 0.99
        refs_Y[parallel_Y] = np.array([1.0, 0.0, 0.0])
        x_Y = np.cross(refs_Y, dirs_Y)
        x_Y = x_Y / (np.linalg.norm(x_Y, axis=1, keepdims=True) + 1e-8)
        y_Y = np.cross(dirs_Y, x_Y)
        frames_Y = np.stack([x_Y, y_Y, dirs_Y], axis=-1)

        # Quaternion conversion and geodesic distance
        quats_X = rotation_matrix_to_quaternion(frames_X)
        quats_Y = rotation_matrix_to_quaternion(frames_Y)
        quats_X = np.roll(quats_X, shift=1, axis=1)
        quats_Y = np.roll(quats_Y, shift=1, axis=1)
        quats_X = np.vstack([quats_X, quats_X[-1]])
        quats_Y = np.vstack([quats_Y, quats_Y[-1]])

        lengths = np.linalg.norm(diff_X, axis=1)
        weights = lengths**2 if args.get("weight_squared", False) else lengths
        weights = weights / (weights.sum() + 1e-8)

        dots = np.abs(np.sum(quats_X[1:] * quats_Y[1:], axis=1))
        dots = np.clip(dots, 0, 1)
        angles = 2 * np.arccos(dots)
        distances = angles**2

        lambda_factor = args.get("lambda_factor", 1.0)
        d = lambda_factor * np.dot(weights, distances)
        total_distance += d

    return total_distance

