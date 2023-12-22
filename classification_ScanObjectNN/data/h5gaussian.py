import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def add_gaussian_noise(data, mean=0, std=0.03):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def add_noise_and_save(input_file, output_folder):
    with h5py.File(input_file, 'r') as f:
        original_data = f['data'][:]  # Assuming 'data' is the dataset name, update accordingly
        labels = f['label'][:]  # Assuming 'label' is the dataset name, update accordingly

    # Add Gaussian noise
    gaussian_noisy_data = add_gaussian_noise(original_data)

    output_file = os.path.join(output_folder, os.path.basename(input_file))

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=gaussian_noisy_data)
        f.create_dataset('label', data=labels)

def main():
    # Example usage
    # os.makedirs(output_folder, exist_ok=True)
    input_files = ['test_objectdataset_augmentedrot_scale75.h5']
    output_folder = 'h5_files/main_split'
    for input_file in input_files:
        add_noise_and_save(input_file, output_folder)

if __name__ == '__main__':
    main()

