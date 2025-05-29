# evaluators/metrics.py
import numpy as np


def calculate_metrics(true, pred):
    # 确保输入为NumPy数组
    true = np.asarray(true)
    pred = np.asarray(pred)

    # 统一计算坐标差值
    delta = true - pred

    # MSE（均方误差）
    squared_errors = np.sum(delta ** 2, axis=1)  # 每个点的欧氏距离平方
    metrics = {'mse': np.mean(squared_errors)}

    # MAE（平均绝对误差）
    abs_errors = np.sum(np.abs(delta), axis=1)  # 每个点的曼哈顿距离
    metrics['mae'] = np.mean(abs_errors)

    # AED（地理平均距离误差）
    euclidean_errors = np.linalg.norm(delta, axis=1)  # 每个点的欧氏距离
    metrics['aed'] = np.mean(euclidean_errors)

    # SMAPE（对称平均绝对百分比误差）
    pred_norms = np.linalg.norm(pred, axis=1)
    true_norms = np.linalg.norm(true, axis=1)
    denominators = (pred_norms + true_norms) / 2 + 1e-8  # 防除零
    metrics['smape'] = np.mean(2 * euclidean_errors / denominators)

    # FDE（最终位移误差）
    metrics['fde'] = euclidean_errors[-1]

    return metrics