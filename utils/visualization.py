# utils/visualization.py
import matplotlib.pyplot as plt
import os

def plot_trajectories(true, pred, save_path=None, route_number=0):
    plt.figure(figsize=(12, 6))
    plt.scatter(true[:, 0], true[:, 1], c='r', label='True Trajectory', alpha=0.6)
    plt.scatter(pred[:, 0], pred[:, 1], c='b', label='Predicted Trajectory', alpha=0.6)
    plt.legend()
    plt.title(f"route{route_number} Trajectories Comparison")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()