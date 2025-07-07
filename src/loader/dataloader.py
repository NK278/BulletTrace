import os
from typing import Tuple
import numpy as np
from scipy.io import loadmat


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all .mat files from the given directory, extract Ez_refl, epsilon_r, and sigma,
    and stack them into 3D arrays.

    Args:
        data_dir: Path to the directory containing .mat files.

    Returns:
        A tuple of three numpy arrays:
            - X_array: stacked Ez_refl with shape (N, H, W)
            - y_eps_array: stacked epsilon_r with shape (N, H, W)
            - y_sig_array: stacked sigma with shape (N, H, W)
    """
    # find all files ending with '.mat'
    mat_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith('.mat')
    ]

    X_list = []
    eps_list = []
    sig_list = []

    for matfile in mat_files:
        mat = loadmat(matfile, squeeze_me=True, struct_as_record=False)
        X_list.append(mat.get('Ez_refl'))
        eps_list.append(mat.get('epsilon_r'))
        sig_list.append(mat.get('sigma'))

    # Stack into 3D arrays of shape (N, H, W)
    X_array = np.stack(X_list, axis=0)
    y_eps_array = np.stack(eps_list, axis=0)
    y_sig_array = np.stack(sig_list, axis=0)
    print("X_array   shape:", X_array.shape)
    print("y_eps_array shape:", y_eps_array.shape)
    print("y_sig_array shape:", y_sig_array.shape) 

    return X_array, y_eps_array, y_sig_array
