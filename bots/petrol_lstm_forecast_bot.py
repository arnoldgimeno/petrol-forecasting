import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


class PetrolLSTMForecastBot:
    """
    Petrol LSTM Forecast Bot

    This bot creates a time-series forecast using an LSTM neural network.

    Best use:
        - Petrol price forecasting
        - Time-series sequence learning
        - Portfolio demonstration of deep learning forecasting

    Main method:
        forecast_next_days(df, date_col, value_col, days=30, lookback=30, epochs=50, batch_size=16)
    """

    def _prepare_series(self, df, date_col, value_col):
        data = df.copy()

        data[date_col] = pd.to_datetime(data[date_col])
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

        data = data.dropna(subset=[date_col, value_col])
        data = data.sort_values(date_col)

        # Make sure there is one value per date.
        data = data.groupby(date_col, as_index=False)[value_col].mean()

        return data

    def _create_sequences(self, values, lookback):
        X, y = [], []

        for i in range(lookback, len(values)):
            X.append(values[i - lookback:i])
            y.append(values[i])

        return np.array(X), np.array(y)

    def forecast_next_days(
        self,
        df,
        date_col="date",
        value_col="price",
        days=30,
        lookback=30,
        epochs=50,
        batch_size=16,
        verbose=0
    ):
        """
        Forecast future petrol prices using LSTM.

        Parameters:
            df (pd.DataFrame): Cleaned petrol dataset.
            date_col (str): Date column name.
            value_col (str): Petrol price column name.
            days (int): Number of future days to forecast.
            lookback (int): Number of previous days used to predict the next day.
            epochs (int): Training rounds for the LSTM model.
            batch_size (int): Number of samples per training batch.
            verbose (int): 0 = silent, 1 = progress bar.

        Returns:
            pd.DataFrame: Forecast dataframe with date, forecast, and model columns.
        """

        try:
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Please install TensorFlow and scikit-learn first:\n"
                "pip install tensorflow scikit-learn"
            ) from e

        data = self._prepare_series(df, date_col, value_col)

        if len(data) <= lookback:
            raise ValueError(
                f"Not enough rows for LSTM. You have {len(data)} rows, "
                f"but lookback is {lookback}. Use a smaller lookback or provide more data."
            )

        values = data[[value_col]].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)

        X, y = self._create_sequences(scaled_values, lookback)

        # LSTM expects 3D input: samples, time steps, features.
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")

        early_stop = EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop]
        )

        # Forecast future days recursively.
        last_sequence = scaled_values[-lookback:].reshape(1, lookback, 1)
        future_scaled_predictions = []

        for _ in range(days):
            next_scaled_value = model.predict(last_sequence, verbose=0)[0][0]
            future_scaled_predictions.append(next_scaled_value)

            next_input = np.array(next_scaled_value).reshape(1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], next_input, axis=1)

        future_predictions = scaler.inverse_transform(
            np.array(future_scaled_predictions).reshape(-1, 1)
        ).flatten()

        last_date = data[date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            date_col: future_dates,
            "forecast": future_predictions,
            "model": "LSTM"
        })

        return forecast_df
