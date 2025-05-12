from os import path as osp
import numpy as np
import argparse
from tqdm import tqdm
import importlib

from src.dtw import dtw



if __name__ == "__main__":
    # Load Files

    # args = argparse.ArgumentParser()
    # args.add_argument("--data_path", type=str, default="sample_data", help="Path to the data directory")
    # args.add_argument("--output_path", type=str, default="data/results", help="Path to the output directory")
    # args.add_argument("--distance_name", type=str, default="procrustes_alignment", help="Distance function name")
    # args.add_argument("--distance_args", type=dict, default={'scale': True, 'mean': True, 'Rotation': True, 's1': 0, 's2': 0, 's3': 0}, help="Distance function arguments")
    # args = args.parse_args()

    # Add this lines for debugging
    args = {'data_path': '/home/vagsart/code/signum/signum/sample_signum/',
            'output_path': 'data/results',
            'distance_name': 'quaternion_distance',
            # 'distance_args': {'scale': True, 'mean': True, 'Rotation': True, 's1': 0, 's2': 0, 's3': 0}}
            'distance_args': {'lambda_factor': 1.0, 'weight_squared': False, 'scale': True}}
    args = argparse.Namespace(**args)

    
    module = importlib.import_module('src.distances')

    # Get the function by name
    distance_fn = getattr(module, args.distance_name)



    X_train = np.load(osp.join(args.data_path, 'X_train.npy'))#[0:10]
    X_test = np.load(osp.join(args.data_path, 'X_test.npy'))#[0:10]
    
    n_landmarks = 21
    dim = 3

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_landmarks, dim)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_landmarks, dim)

    assert X_train.shape[2] == n_landmarks and X_train.shape[3] == dim, "X_train shape mismatch"
    assert X_test.shape[2] == n_landmarks and X_test.shape[3] == dim, "X_test shape mismatch"

    
    distances = np.zeros((X_train.shape[0], X_test.shape[0]))

    for i in tqdm(range(X_train.shape[0]), desc="Calculating distances"):
        x1 = X_train[i] 
        for j in range(X_test.shape[0]):
            print(j)
            x2 = X_test[j]
            dtw_cost = dtw(x1, x2, distance_fn=distance_fn, distance_args=args.distance_args)

            distances[i][j] = dtw_cost
    
    
    np.save(osp.join(args.output_path, 'distances_train_test_proc.npy'), distances)


