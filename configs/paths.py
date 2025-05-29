# configs/paths.py
import os

def get_paths(model_number, time_solution, test_number=None):
    dataset_path = r".\datasets"
    output_path = r".\outputs"
    return {
        "train_data": os.path.join(
            dataset_path, "train_dataset", f"route{model_number}",
            f"route{model_number}_interpolation_{time_solution}.csv"
        ),
        "test_data": os.path.join(
            dataset_path, "test_dataset", f"route{model_number}",
            f"route{model_number}_interpolation_{time_solution}.csv"
        ),
        "outputs": {
            "models": os.path.join(output_path, "models"),
            "results": os.path.join(output_path, "predicted_position"),
            "plots": os.path.join(output_path, "plots"),
            "accuracy": os.path.join(output_path, "accuracy")
        }
    }