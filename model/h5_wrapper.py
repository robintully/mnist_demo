"""Module to interact with h5 files"""
import h5py


def save_to_h5(f_name, data):
    """Saves af ile to h5"""
    h5f = h5py.File(str(f_name) + '.h5', 'w')
    h5f.create_dataset('dataset_1', data=data)
    h5f.close()
