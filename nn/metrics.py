from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer

@register_keras_serializable()
def nll_gaussian(y_true, y_pred):
    """Negative log-likelihood loss for Gaussian predictions"""
    mean = y_pred[:, :2]
    log_var = y_pred[:, 2:]
    precision = K.exp(-log_var)
    return K.mean(K.sum(0.5 * log_var + 0.5 * precision * K.square(y_true - mean), axis=1)) 

def nll_gaussian_np(y_true, y_pred):
    """Negative log-likelihood loss for Gaussian predictions"""
    mean = y_pred[:, :2]
    log_var = y_pred[:, 2:]
    precision = np.exp(-log_var)
    nll = 0.5 * (log_var + precision * (y_true - mean)**2)
    return np.mean(np.sum(nll, axis=1))




