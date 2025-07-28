import numpy as np
from sklearn.model_selection import train_test_split

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data

def load_UCR(Path='../../archives/UCR_UEA/Multivariate_arff/', folder='Cricket', test_size=0.2, random_state=42):
    # Load full data
    X = np.load("/content/drive/MyDrive/FD/X.npy")  # shape (N, C, T)
    y = np.load("/content/drive/MyDrive/FD/y.npy")  # shape (N,)

    # Transpose to (N, T, C)
    X = X.transpose(0, 2, 1)

    # Optional: truncate sequence length
    MAX_TIME_STEPS = 30000
    X = X[:, :MAX_TIME_STEPS, :]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Ensure correct types
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return (X_train, y_train), (X_test, y_test)
