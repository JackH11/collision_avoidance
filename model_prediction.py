from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils import get_model, save_data, get_data
from nn.nn import nll_gaussian, ClippedLogVar

# Thread pool for async predictions
_model_cache = None

def agent_predict(model, item):
    """Synchronous prediction function"""
    X = item.get_history(lag=5)
    y_pred = model.predict(np.array([X]), verbose=0)
    pred_x = y_pred[0][0]
    pred_y = y_pred[0][1]
    return pred_x, pred_y

def agent_uncertain_predict(model, item):
    X = item.get_history(lag=10,window=5)
    y_pred = model.predict(np.array([X]), verbose=0)
    pred_x = y_pred[0][0]
    pred_y = y_pred[0][1]
    std_x = y_pred[0][2]
    std_y = y_pred[0][3]
    return pred_x, pred_y, std_x, std_y

def update_item_async(item):
    """Asynchronous update function"""
    item.update()
    return item

def get_cached_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = get_model("j_10_5",
                                 custom_objects={'nll_gaussian': nll_gaussian, 'ClippedLogVar': ClippedLogVar},
                                 safe_mode=False)
    return _model_cache

def make_nn_prediction(item):


    model = get_cached_model()

    X = item.get_history(lag=10, window=5)
    y_pred = model.predict(np.array([X]), verbose=0)
    pred_x = y_pred[0][0]
    pred_y = y_pred[0][1]
    std_x = y_pred[0][2]
    std_y = y_pred[0][3]
    return pred_x, pred_y, std_x, std_y

def make_simple_prediction(item):

    pred_x = item.x + item.vx * 8
    pred_y = item.y + item.vy * 8
    speed = (item.vx ** 2 + item.vy ** 2) ** 0.5
    std_x = 0.5 * speed  # tweak multiplier
    std_y = 0.5 * speed

    return pred_x, pred_y, std_x, std_y