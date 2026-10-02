import os, csv
from typing import List, Dict, Union, Optional
import pandas as pd

from collision_avoidance.config import REPO_ROOT


def get_model(
    model_name: str,
    custom_objects: Optional[dict] = None,
    safe_mode: bool = True
):
    """
    Loads a Keras model from the nn/models folder.

    TensorFlow is imported lazily so core train/demo paths can avoid TF.

    Args:
        model_name (str): Name of the model (without .keras extension)
        custom_objects (dict, optional): Dictionary of custom objects (e.g. custom losses/layers)
        safe_mode (bool): Whether to enable safe_mode in load_model (default True)

    Returns:
        tf.keras.Model: The loaded model
    """
    from tensorflow.keras.models import load_model

    model_path = REPO_ROOT / "nn" / "models" / f"{model_name}.keras"

    model = load_model(model_path, custom_objects=custom_objects, safe_mode=safe_mode)
    return model


def save_model(
    model,
    model_name: str,
    overwrite: bool = True
):
    """
    Saves a Keras model to the nn/models folder.

    Args:
        model: The Keras model to save
        model_name (str): Name of the model (without .keras extension)
        overwrite (bool): Whether to overwrite existing model (default False)

    Returns:
        str: Path where the model was saved
    """
    model_path = REPO_ROOT / "nn" / "models" / f"{model_name}.keras"

    os.makedirs(model_path.parent, exist_ok=True)

    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Model {model_name} already exists. Set overwrite=True to overwrite.")

    model.save(str(model_path))
    return str(model_path)


def get_data(
    data_name: str
):
    """Load a CSV from the repo ``data/`` folder."""
    data_path = REPO_ROOT / "data" / f"{data_name}.csv"
    return pd.read_csv(data_path)


def save_data(
    data: Union[List[Dict], Dict],
    path: str = 'data/train_raw.csv',
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Append rows of data to a CSV file, creating it with a header if needed.

    Args:
        data: A list of dictionaries or a single dictionary representing rows to append.
        path: File path where the CSV will be saved (default: 'data/train_raw.csv').
        fieldnames: Optional list of field names for the CSV header. If not provided,
                    they will be inferred from the first row of data.

    Example:
        save_data(
            [{'episode':1, 'frame':0, 'item':'A', 'x':0.1,'y':0.2,'vx':1.0,'vy':0.0}],
            'data/train_raw.csv'
        )
    """
    # Ensure data is a list of dicts
    if isinstance(data, dict):
        data = [data]

    if not data:
        return  # Nothing to write

    # Infer fieldnames if not provided
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.isfile(path)

    # Write the data
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)




def point_in_polygon(x, y, polygon):
    # polygon is a list of (x,y)
    num = len(polygon)
    j = num - 1
    inside = False
    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside













