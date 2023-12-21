import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def cutout(pointcloud, severity):
    # 只是传入一个点云
    N, C = pointcloud.shape
    c = [(2, 30), (3, 60), (5, 30), (7, 30), (10, 80)][severity - 1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud

def random_dropout(dropout_prob=0.9):
    mask = np.random.choice([False, True], size=2048, p=[dropout_prob, 1 - dropout_prob])
    # print('--', data.shape[1])
    return mask
    # return data[:, mask]

def add_noise_and_save(input_file, output_folder):
    with h5py.File(input_file, 'r') as f:
        original_data = f['data'][:]  # Assuming 'data' is the dataset name, update accordingly
        labels = f['label'][:]  # Assuming 'label' is the dataset name, update accordingly

    # Randomly dropout points
    # dropout_data = random_dropout(original_data)

    dropout_data = []
    for i in range(original_data.shape[0]):
        dropout_data.append(cutout(original_data[i], 5))
    # dropout_data = original_data[:, mask]
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=dropout_data)
        f.create_dataset('label', data=labels)

# Example usage
input_files = ['ply_data_test0.h5', 'ply_data_test1.h5']
output_folder = 'modelnet40_ply_hdf5_2048'
# os.makedirs(output_folder, exist_ok=True)
# mask = random_dropout()
for input_file in input_files:
    add_noise_and_save(input_file, output_folder)



