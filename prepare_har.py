import numpy as np
import os

def load_signals(signal_folder):
    """
    signal_folder: 'UCI HAR Dataset/train/Inertial Signals' 등
    각 파일마다 shape (n_samples, 128) 인 신호가 있고,
    총 9개의 축/센서 파일을 concat → (n_samples, 128*9)
    """
    files = sorted([
        os.path.join(signal_folder, f) 
        for f in os.listdir(signal_folder) 
        if f.endswith('.txt')
    ])
    # 각각 (n_samples, 128)
    arrays = [np.loadtxt(f) for f in files]
    # concat axis=1 → (n_samples, 128 * len(files))
    return np.concatenate(arrays, axis=1)

def load_har_dataset(base_path='./UCI_HAR_Dataset/UCI HAR Dataset'):
    # train
    X_train = load_signals(os.path.join(base_path, 'train', 'Inertial Signals'))
    y_train = np.loadtxt(os.path.join(base_path, 'train', 'y_train.txt'), dtype=int) - 1  # 0~5 로 만들기

    # test
    X_test  = load_signals(os.path.join(base_path, 'test', 'Inertial Signals'))
    y_test  = np.loadtxt(os.path.join(base_path, 'test', 'y_test.txt'),  dtype=int) - 1

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_har_dataset()

    # 합쳐서 저장
    np.save('har_X.npy', np.vstack([X_train, X_test]))
    np.save('har_y.npy', np.hstack([y_train, y_test]))
    print("Saved har_X.npy (shape", np.load('har_X.npy').shape, ")",
          "and har_y.npy (shape", np.load('har_y.npy').shape, ")")
