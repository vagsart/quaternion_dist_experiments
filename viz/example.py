import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from src.distances import quaternion_distance

# from gpu.distances_gpu import quaternion_distance_torch as quaternion_distance
# import torch

# Paths and data loading
data_path = '/home/vagsart/code/signum/signum/sample_signum/'
X_train = np.load(osp.join(data_path, 'X_train.npy'))
X_test = np.load(osp.join(data_path, 'X_test.npy'))

# Reshape a sample to (21, 3)
X0 = X_train[0][10].reshape(21, 3)
X1 = X_train[0][11].reshape(21, 3)

print("X0 shape:", X0.shape)
print("X1 shape:", X1.shape)

# X0 = torch.from_numpy(X0).float()
# X1 = torch.from_numpy(X1).float()

# Compute quaternion distance
dist = quaternion_distance(X0, X1)
print("Quaternion distance:", dist)


# Define MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (17, 5)
]


# Plotting helper
def plot_hand(ax, landmarks, title="Hand Skeleton", color='blue'):
    ax.scatter(landmarks[:, 0], landmarks[:, 1], color=color, s=50, zorder=2)
    for i, j in HAND_CONNECTIONS:
        ax.plot([landmarks[i, 0], landmarks[j, 0]], 
                [landmarks[i, 1], landmarks[j, 1]], 
                color='black', linewidth=1.5, zorder=1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.invert_yaxis()  # To match MediaPipe view
    ax.axis('off')


# Show both samples
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_hand(axs[0], X0, title="Sample X0")
plot_hand(axs[1], X1, title="Sample X1")
plt.suptitle(f"Quaternion Distance: {dist:.4f}")
plt.tight_layout()
plt.show()
