# data/preprocessing.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def prepare_datasets(train_df, test_df, seq_length, train_ratio=0.8):
    # 合并数据归一化
    combined = np.vstack([train_df.values, test_df.values])
    scaler = MinMaxScaler().fit(combined)

    # 处理训练数据
    train_scaled = scaler.transform(train_df)
    X_train, y_train = create_sequences(train_scaled, seq_length)

    # 划分训练验证集
    train_size = int(len(X_train) * train_ratio)
    X_val, y_val = X_train[train_size:], y_train[train_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    # 处理测试数据
    test_scaled = scaler.transform(test_df)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler