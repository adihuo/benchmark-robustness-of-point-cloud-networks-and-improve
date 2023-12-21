import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def random_rotation(data, max_angle=30):
    # Create a 4x4 identity matrix for the transformation
    transform_1 = np.identity(4, dtype=np.float32)
    # Set the rotation angle
    theta = np.pi / 10
    # Set the rotation matrix elements
    transform_1[0, 0] = np.cos(theta)
    transform_1[0, 1] = -np.sin(theta)
    transform_1[1, 0] = np.sin(theta)
    transform_1[1, 1] = np.cos(theta)

    # Set the translation values
    # transform_1[0, 3] = 5.5
    # transform_1[1, 3] = -6.5
    # transform_1[2, 3] = 0

    # Apply the transformation to each point cloud in original_data
    transformed_data = np.empty_like(data)

    for i in range(data.shape[0]):  # Iterate over the first dimension (419)
        # Extract the i-th point cloud
        point_cloud = data[i, :, :]

        # Apply the transformation
        transformed_point_cloud = np.dot(point_cloud, transform_1[:3, :3].T) + transform_1[:3, 3]

        # Save the transformed point cloud
        transformed_data[i, :, :] = transformed_point_cloud
    return transformed_data

def add_noise_and_save(input_file, output_folder):
    with h5py.File(input_file, 'r') as f:
        original_data = f['data'][:]  # Assuming 'data' is the dataset name, update accordingly
        labels = f['label'][:]  # Assuming 'label' is the dataset name, update accordingly

    # Randomly dropout points
    rotated_data = random_rotation(original_data)
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=rotated_data)
        f.create_dataset('label', data=labels)


def main():
    # Example usage
    input_files = ['ply_data_test0.h5', 'ply_data_test1.h5']
    output_folder = 'modelnet40_ply_hdf5_2048'
    # os.makedirs(output_folder, exist_ok=True)
    for input_file in input_files:
        add_noise_and_save(input_file, output_folder)

if __name__ == '__main__':
    main()

