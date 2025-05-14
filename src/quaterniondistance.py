#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

def align_base(X, Y):
    return Y + (X[0] - Y[0])

def compute_link_directions(chain):
    diff = np.diff(chain, axis=0)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = np.divide(diff, norms, out=np.zeros_like(diff), where=norms >= 1e-8)
    return directions

def compute_local_frames(vectors, ref=np.array([0, 1, 0])):
    refs = np.tile(ref, (vectors.shape[0], 1))
    parallel = np.abs(np.einsum('ij,ij->i', vectors, refs)) > 0.99
    refs[parallel] = np.array([1, 0, 0])

    x = np.cross(refs, vectors)
    norms_x = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms_x, out=np.tile([1.0, 0.0, 0.0], (len(vectors), 1)), where=norms_x >= 1e-8)

    y = np.cross(vectors, x)
    return np.stack((x, y, vectors), axis=-1)

def frames_to_quaternions(frames):
    rotations = R.from_matrix(frames)
    quats = rotations.as_quat()
    quats_wxyz = np.roll(quats, shift=1, axis=1)
    return quats_wxyz

def compute_chain_quaternions(chain, ref_vector=np.array([0, 1, 0])):
    if len(chain) < 2:
        return np.array([[1.0, 0.0, 0.0, 0.0]])

    directions = compute_link_directions(chain)
    frames = compute_local_frames(directions, ref=ref_vector)
    quats = frames_to_quaternions(frames)
    quats = np.vstack([quats, quats[-1]])
    return quats

def quaternion_geodesic_distance(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, 0, 1)
    theta = 2 * np.arccos(dot)
    return theta**2

def compute_link_lengths(chain):
    diff = np.diff(chain, axis=0)
    return np.linalg.norm(diff, axis=1)

def compute_weighted_quaternion_distance(X, Y, lambda_factor=1.0, weight_squared=False, scale=False):
    
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if scale:
        X = X / (normX + 1e-15)
        Y = Y / (normY + 1e-15)

    # 1: Align Y so that its base matches X.
    Y_aligned = align_base(X, Y)
    
    # 2: Compute quaternions for each chain.
    quats_X = compute_chain_quaternions(X)
    quats_Y = compute_chain_quaternions(Y_aligned)

    # 3: Compute per-link weights based on link lengths (using chain X).
    lengths = compute_link_lengths(X)
    weights = lengths**2 if weight_squared else lengths
    weights /= weights.sum()

    # 4: Compute the weighted geodesic distance.
    distances = quaternion_geodesic_distance(quats_X[1:], quats_Y[1:])
    return lambda_factor * np.dot(weights, distances)

def compute_hand_weighted_quaternion_distance(X_hand, Y_hand, lambda_factor=1.0, weight_squared=False):
    return sum(compute_weighted_quaternion_distance(X, Y, lambda_factor, weight_squared) for X, Y in zip(X_hand, Y_hand))

def plot_hand_chains(X_hand, Y_hand, title="Hand Kinematic Chains"):
    colors = ["blue", "red", "green", "purple", "orange"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i, (X, Y) in enumerate(zip(X_hand, Y_hand)):
        ax.plot(X[:, 0], X[:, 1], X[:, 2], marker="o", color=colors[i], label=f"Chain X (Finger {i + 1})")
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], marker="^", linestyle="--", color=colors[i], label=f"Chain Y (Finger {i + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    n_joints, n_fingers = 5, 5
    t = np.linspace(0, 2 * np.pi, n_joints)
    
    base_chain = np.column_stack((np.cos(t), np.sin(t), t / (2 * np.pi)))

    X_hand = [base_chain + np.array([0, i * 0.2, 0]) for i in range(n_fingers)]

    angle_rad = np.radians(10)
    R_z = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])

    Y_hand = []
    for chain in X_hand:
        Y = (R_z @ chain.T).T + 0.05 * np.random.randn(*chain.shape)
        Y_hand.append(align_base(chain, Y))

    n_iter = 1000
    t_start = time.perf_counter()
    for _ in range(n_iter):
        _ = compute_hand_weighted_quaternion_distance(X_hand, Y_hand)
    t_end = time.perf_counter()

    avg_time = (t_end - t_start) / n_iter
    print(f"Average time per iteration: {avg_time:.6f} seconds")

    hand_distance = compute_hand_weighted_quaternion_distance(X_hand, Y_hand)
    print("Weighted Quaternion Distance for the Hand:", hand_distance)

    plot_hand_chains(X_hand, Y_hand)
