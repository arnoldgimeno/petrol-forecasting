import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


class PetrolAutoKerasForecastBot:
    """
    Petrol AutoKeras Forecast Bot

    This bot creates a forecast using AutoKeras.

    Important:
        AutoKeras can be heavy and may take longer to run than Linear Regression or ARIMA.

    Best use:
        - Automated machine learning experiment
        - Advanced portfolio demonstration
        - Comparing AutoML against traditional forecasting models

    Main method:
        forecast_next_days(df, date_col, value_col, days=30, lags=14, max_trials=5, epochs=30)
    """

    def _prepare_data(self, df, date_col, value_col):
        data = df.copy()

        data[date_col] = pd.to_datetime(data[date_col])
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

        data = data.dropna(subset=[date_col, value_col])
        data = data.sort_values(date_col)

        # Make sure there is one value per date.
        data = data.groupby(date_col, as_index=False)[value_col].mean()

        # Create daily date index and fill missing dates.
        data = data.set_index(date_col).asfreq("D")
        data[value_col] = data[value_col].interpolate(method="time")
        data[value_col] = data[value_col].ffill().bfill()
        data = data.reset_index()

        return data

    def _create_lag_features(self, data, value_col, lags):
        df_lagged = data.copy()

        for lag in range(1, lags + 1):
            df_lagged[f"lag_{lag}"] = df_lagged[value_col].shift(lag)

        df_lagged = df_lagged.dropna().reset_index(drop=True)

        feature_cols = [f"lag_{lag}" for lag in range(1, lags + 1)]

        X = df_lagged[feature_cols].values
        y = df_lagged[value_col].values

        return X, y, feature_cols

    def forecast_next_days(
        self,
        df,
        date_col="date",
        value_col="price",
        days=30,
        lags=14,
        max_trials=5,
        epochs=30,
        overwrite=True
    ):
        """
        Forecast future petrol prices using AutoKeras.

        Parameters:
            df (pd.DataFrame): Cleaned petrol dataset.
            date_col (str): Date column name.
            value_col (str): Petrol price column name.
            days (int): Number of future days to forecast.
            lags (int): Number of previous days used as features.
            max_trials (int): Number of model attempts AutoKeras will test.
            epochs (int): Training rounds.
            overwrite (bool): Whether AutoKeras can overwrite previous trial files.

        Returns:
            pd.DataFrame: Forecast dataframe with date, forecast, and model columns.
        """

        try:
            import autokeras as ak
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Please install AutoKeras first:\n"
                "pip install autokeras"
            ) from e

        data = self._prepare_data(df, date_col, value_col)

        if len(data) <= lags:
            raise ValueError(
                f"Not enough rows for AutoKeras. You have {len(data)} rows, "
                f"but lags is {lags}. Use a smaller lag value or provide more data."
            )

        X, y, feature_cols = self._create_lag_features(data, value_col, lags)

        regressor = ak.StructuredDataRegressor(
            max_trials=max_trials,
            overwrite=overwrite
        )

        regressor.fit(
            X,
            y,
            epochs=epochs,
            verbose=0
        )

        # Recursive forecasting.
        history = list(data[value_col].values)
        predictions = []

        for _ in range(days):
            lag_values = history[-lags:]
            X_future = np.array(lag_values).reshape(1, -1)

            next_prediction = regressor.predict(X_future, verbose=0)[0][0]
            predictions.append(float(next_prediction))

            history.append(float(next_prediction))

        last_date = data[date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            date_col: future_dates,
            "forecast": predictions,
            "model": "AutoKeras"
        })

        return forecast_df
