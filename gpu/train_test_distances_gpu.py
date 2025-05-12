import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from distances_gpu import quaternion_distance_torch
from dtw_gpu import dtw_gpu

# Load data
X_train = torch.from_numpy(np.load("/home/vagsart/code/signum/signum/sample_signum/X_train.npy")).float().cuda()
X_test = torch.from_numpy(np.load("/home/vagsart/code/signum/signum/sample_signum/X_test.npy")).float().cuda()

n_landmarks = 21
dim = 3
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_landmarks, dim)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_landmarks, dim)

args = {"lambda_factor": 1.0, "weight_squared": False}
distances = torch.zeros((X_train.shape[0], X_test.shape[0]), device="cuda")

for i in tqdm(range(X_train.shape[0])):
    for j in range(X_test.shape[0]):
        dist = dtw_gpu(X_train[i], X_test[j], quaternion_distance_torch, args)
        distances[i, j] = dist

# Save to disk (optionally move back to CPU)
np.save("distances_train_test_gpu.npy", distances.cpu().numpy())

