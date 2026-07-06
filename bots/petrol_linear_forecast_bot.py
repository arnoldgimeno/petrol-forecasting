import pandas as pd
from sklearn.linear_model import LinearRegression


class PetrolLinearForecastBot:
    """
    Creates simple time-series forecasts using lag features
    and Linear Regression.
    """

    def prepare_time_series(self, df, date_col, value_col):
        """
        Cleans and prepares the time-series dataframe.
        """

        out = df.copy()

        # Make sure date column exists as a normal column
        if date_col not in out.columns:
            out = out.reset_index()

        # Convert date column to datetime
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

        # Convert value column to numeric
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

        # Remove rows with missing date or value
        out = out.dropna(subset=[date_col, value_col])

        # Sort by date
        out = out.sort_values(date_col)

        # Reset index
        out = out.reset_index(drop=True)

        return out

    def create_lag_features(self, df, value_col, lags=7):
        """
        Creates lag columns from previous days.

        Example:
        lag_1 = yesterday's price
        lag_2 = price 2 days ago
        """

        out = df.copy()

        for lag in range(1, lags + 1):
            out[f"lag_{lag}"] = out[value_col].shift(lag)

        # Remove rows where lag values are missing
        out = out.dropna().reset_index(drop=True)

        return out

    def forecast_next_days(self, df, date_col, value_col, days=30, lags=7, model_name="Linear Regression"):
        """
        Forecasts future values.

        Parameters:
        df         = dataframe with date and price/value columns
        date_col   = name of date column
        value_col  = name of value column to forecast
        days       = number of future days to forecast
        lags       = number of past days used as features
        model_name = label stored in the "model" column of the output,
                     kept consistent with the other forecast bots
                     (ARIMA / LSTM / AutoKeras) so results can be
                     concatenated and charted together.
        """

        # Prepare clean time-series data
        ts = self.prepare_time_series(df, date_col, value_col)

        # Create lag features
        model_data = self.create_lag_features(ts, value_col, lags=lags)

        # Define feature columns
        feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]

        # X = previous values
        X = model_data[feature_cols]

        # y = current value
        y = model_data[value_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Get latest known values
        last_values = list(ts[value_col].tail(lags).values)

        # Last known date
        last_date = ts[date_col].max()

        future_dates = []
        future_predictions = []

        # Predict future days one by one
        for i in range(1, days + 1):

            # Create prediction input with same feature names used during training
            input_data = pd.DataFrame(
                [last_values[-lags:]],
                columns=feature_cols
            )

            # Predict next value
            next_prediction = model.predict(input_data)[0]

            # Create next date
            next_date = last_date + pd.Timedelta(days=i)

            # Store result
            future_dates.append(next_date)
            future_predictions.append(next_prediction)

            # Add prediction so the next forecast can use it
            last_values.append(next_prediction)

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            date_col: future_dates,
            "forecast": future_predictions,
            "model": model_name
        })

        return forecast_df
