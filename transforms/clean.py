import pandas as pd



def create_lag_features(df: pd.DataFrame, lag_steps: int = 20):

    df = df.sort_values(['item', 'frame','episode'])

    for lag in range(1, lag_steps + 1):
        df[f'x_lag_{lag}'] = df.groupby(['item','episode']).x.shift(lag).round(2)
        df[f'y_lag_{lag}'] = df.groupby(['item','episode']).y.shift(lag).round(2)
        df[f'vx_lag_{lag}'] = df.groupby(['item','episode']).vx.shift(lag).round(2)
        df[f'vy_lag_{lag}'] = df.groupby(['item','episode']).vy.shift(lag).round(2)

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == '__main__':
    df = pd.read_csv('data/train_raw.csv')
    df = create_lag_features(df)
    df.to_csv('data/train_lag.csv', index=False)