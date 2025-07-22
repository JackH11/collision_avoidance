from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Thread pool for async predictions


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