import pandas as pd
import numpy as np
import os, argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
from .metrics import nll_gaussian, nll_gaussian_np


@register_keras_serializable()
class ClippedLogVar(Layer):
    def call(self, x):
        mean = x[:, :2]
        log_var = K.clip(x[:, 2:], -5.0, 5.0)
        return K.concatenate([mean, log_var], axis=1)

def build_uncertainty_model(input_dim):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(4, activation='linear'),  # Output: [μ_x, μ_y, logσ_x², logσ_y²]
        ClippedLogVar()
    ])
    model.compile(optimizer='adam', loss=nll_gaussian)
    return model

def build_model(input_dim):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_and_save_model(X, y, model_path, epochs, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("model_type", model_type)
    
    if model_type == "c":
       model = build_uncertainty_model(X_train.shape[1])
    elif model_type == "d":
       model = build_model(X_train.shape[1])
    else:
       raise ValueError(f"Invalid model type: {model_type}")

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)


    model.save(model_path)
    print(f"Model saved to {model_path}")
    if model_type == "d":
        loss, mae = model.evaluate(X_test, y_test)
        print(f"Train/Test evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")
    elif model_type == "c":
        y_pred = model.predict(X_test)
        nll = nll_gaussian_np(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred[:, :2]))

        print(f"Test Evaluation - Negative Log-Likelihood: {nll:.4f}, MAE (mean only): {mae:.4f}")
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def load_and_test_model(X, y, model_path, model_type):
    loaded_model = load_model(model_path)
    if model_type == "d":
        loss, mae = loaded_model.evaluate(X, y)
        print(f"Loaded model evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")
    elif model_type == "c":
        y_pred = loaded_model.predict(X)
        nll = nll_gaussian_np(y, y_pred)
        mae = np.mean(np.abs(y - y_pred[:, :2]))
        print(f"Loaded model evaluation - Negative Log-Likelihood: {nll:.4f}, MAE (mean only): {mae:.4f}")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return y_pred

def main():

    parser = argparse.ArgumentParser(description="Train or load model")
    parser.add_argument('--mode', choices=['train', 'load'], required=True,
                        help="Mode: 'train' to train and save, 'load' to load and test")
    # Get the script directory and construct absolute path to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'data', 'train_lag.csv')
    parser.add_argument('--data', type=str, default=default_data_path, help="CSV data path")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of epochs for training (only for train mode)")
    parser.add_argument('--lag', type=int, default=5,
                        help="Number of lag steps")
    parser.add_argument('--type', choices=['c','d'], default='d',
                        help="Confidence or deterministic model")
    
    args = parser.parse_args()

    df = pd.read_csv('../data/train_lag.csv')

    lag_features = []
    for lag in range(args.lag, args.lag + 15):

        if f'x_lag_{lag}' not in df.columns:
            print(df.columns)
            raise ValueError(f"Lag {lag} not found in data")

        lag_features.append(f'x_lag_{lag}')
        lag_features.append(f'y_lag_{lag}')
        lag_features.append(f'vx_lag_{lag}')
        lag_features.append(f'vy_lag_{lag}')

    X = df[lag_features]
    y = df[['x', 'y']]

    # Create models directory if it doesn't exist
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'model_{args.epochs}_{args.lag}_{args.type}.keras')

    if args.mode == 'train':
        train_and_save_model(X, y, model_path, args.epochs, args.type)
    elif args.mode == 'load':
        y_pred = load_and_test_model(X, y, model_path, args.type)

if __name__ == '__main__':

    # Example usage:
    #   Train a model for 500 epochs:
    #   python nn.py --mode train --epochs 500
    #
    #   Load and test a previously trained model:
    #   python nn.py --mode load --epochs 500
    #
    #   Use custom data file:
    #   python nn.py --mode train --data path/to/data.csv --epochs 100
    #
    #   Change number of lag steps:
    #   python nn.py --mode train --lag 10 --epochs 100
    
    main()