from sklearn.datasets import fetch_covtype
from sklearn.utils import check_array
import numpy as np





def load_data(dtype=np.float32, order='C', random_state=13):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(download_if_missing=True, shuffle=True,
                         random_state=random_state)
    X = check_array(data['data'], dtype=dtype, order=order)
    y = (data['target'] != 1).astype(np.int)

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test




if __name__ == "__main__":
    X_train, X_test, y_train, y_test  = load_data()
    print(len(X_train), len(X_test))

