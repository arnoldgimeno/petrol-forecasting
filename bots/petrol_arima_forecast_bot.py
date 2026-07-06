import warnings
warnings.filterwarnings("ignore")

import pandas as pd


class PetrolARIMAForecastBot:
    """
    Petrol ARIMA Forecast Bot

    This bot creates a time-series forecast using ARIMA.

    Best use:
        - Traditional time-series forecasting
        - Petrol price forecasting
        - Business-friendly model comparison

    Main method:
        forecast_next_days(df, date_col, value_col, days=30, order=(5, 1, 0))
    """

    def _prepare_series(self, df, date_col, value_col):
        data = df.copy()

        data[date_col] = pd.to_datetime(data[date_col])
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

        data = data.dropna(subset=[date_col, value_col])
        data = data.sort_values(date_col)

        # Make sure there is one value per date.
        data = data.groupby(date_col, as_index=False)[value_col].mean()

        # Create daily date index.
        series = data.set_index(date_col)[value_col].asfreq("D")

        # Fill missing daily values using time interpolation.
        series = series.interpolate(method="time")
        series = series.ffill().bfill()

        return series

    def forecast_next_days(
        self,
        df,
        date_col="date",
        value_col="price",
        days=30,
        order=(5, 1, 0)
    ):
        """
        Forecast future petrol prices using ARIMA.

        Parameters:
            df (pd.DataFrame): Cleaned petrol dataset.
            date_col (str): Date column name.
            value_col (str): Petrol price column name.
            days (int): Number of future days to forecast.
            order (tuple): ARIMA order in the form (p, d, q).

        Returns:
            pd.DataFrame: Forecast dataframe with date, forecast, and model columns.
        """

        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Please install statsmodels first:\n"
                "pip install statsmodels"
            ) from e

        series = self._prepare_series(df, date_col, value_col)

        if len(series) < 20:
            raise ValueError(
                f"Not enough rows for ARIMA. You have {len(series)} rows. "
                "ARIMA works better with more time-series observations."
            )

        model = ARIMA(series, order=order)
        fitted_model = model.fit()

        forecast_values = fitted_model.forecast(steps=days)

        forecast_df = pd.DataFrame({
            date_col: forecast_values.index,
            "forecast": forecast_values.values,
            "model": f"ARIMA{order}"
        })

        return forecast_df
