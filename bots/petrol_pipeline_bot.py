import warnings
warnings.filterwarnings("ignore")

import pandas as pd


class PetrolPipelineBot:
    """
    Petrol Pipeline Bot

    This bot runs the complete petrol forecasting workflow:

        1. Load raw petrol dataset
        2. Clean petrol dataset
        3. Run selected forecasting models
        4. Combine forecast results
        5. Optionally create an integrated actual vs. forecast comparison chart

    Supported models:
        - linear
        - arima
        - lstm
        - autokeras

    Main method:
        run_pipeline(file_path, models=["linear", "arima"], days=30)
    """

    def __init__(self):
        """
        Import project bots only when the pipeline is created.

        This keeps the file cleaner and gives clearer error messages
        if one of the bot files is missing.
        """

        try:
            from bots.ingestion_bot import IngestionBot
            from bots.petrol_transform_bot import PetrolTransformBot
            from bots.petrol_visualization_bot import PetrolVisualizationBot
        except ImportError as e:
            raise ImportError(
                "One or more required base bot files are missing.\n\n"
                "Please check that these files exist inside your bots folder:\n"
                "- ingestion_bot.py\n"
                "- petrol_transform_bot.py\n"
                "- petrol_visualization_bot.py"
            ) from e

        self.ingestion_bot = IngestionBot()
        self.transform_bot = PetrolTransformBot()
        self.visualization_bot = PetrolVisualizationBot()

    def _run_linear_model(
        self,
        df,
        date_col,
        value_col,
        days,
        lags
    ):
        try:
            from bots.petrol_linear_forecast_bot import PetrolLinearForecastBot
        except ImportError as e:
            raise ImportError(
                "Missing file: petrol_linear_forecast_bot.py"
            ) from e

        model = PetrolLinearForecastBot()

        return model.forecast_next_days(
            df=df,
            date_col=date_col,
            value_col=value_col,
            days=days,
            lags=lags,
            model_name="Linear Regression"
        )

    def _run_arima_model(
        self,
        df,
        date_col,
        value_col,
        days,
        arima_order
    ):
        try:
            from bots.petrol_arima_forecast_bot import PetrolARIMAForecastBot
        except ImportError as e:
            raise ImportError(
                "Missing file: petrol_arima_forecast_bot.py"
            ) from e

        model = PetrolARIMAForecastBot()

        return model.forecast_next_days(
            df=df,
            date_col=date_col,
            value_col=value_col,
            days=days,
            order=arima_order
        )

    def _run_lstm_model(
        self,
        df,
        date_col,
        value_col,
        days,
        lookback,
        lstm_epochs,
        lstm_batch_size
    ):
        try:
            from bots.petrol_lstm_forecast_bot import PetrolLSTMForecastBot
        except ImportError as e:
            raise ImportError(
                "Missing file: petrol_lstm_forecast_bot.py"
            ) from e

        model = PetrolLSTMForecastBot()

        return model.forecast_next_days(
            df=df,
            date_col=date_col,
            value_col=value_col,
            days=days,
            lookback=lookback,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size
        )

    def _run_autokeras_model(
        self,
        df,
        date_col,
        value_col,
        days,
        autokeras_lags,
        autokeras_max_trials,
        autokeras_epochs
    ):
        try:
            from bots.petrol_autokeras_forecast_bot import PetrolAutoKerasForecastBot
        except ImportError as e:
            raise ImportError(
                "Missing file: petrol_autokeras_forecast_bot.py"
            ) from e

        model = PetrolAutoKerasForecastBot()

        return model.forecast_next_days(
            df=df,
            date_col=date_col,
            value_col=value_col,
            days=days,
            lags=autokeras_lags,
            max_trials=autokeras_max_trials,
            epochs=autokeras_epochs
        )

    def run_pipeline(
        self,
        file_path,
        models=None,
        date_col="date",
        value_col="price",
        days=30,
        linear_lags=7,
        arima_order=(5, 1, 0),
        lstm_lookback=30,
        lstm_epochs=50,
        lstm_batch_size=16,
        autokeras_lags=14,
        autokeras_max_trials=5,
        autokeras_epochs=30,
        show_chart=True
    ):
        """
        Run the complete petrol forecasting pipeline.

        Parameters:
            file_path (str): Path to the raw petrol dataset.
            models (list): Forecast models to run.
                Example: ["linear", "arima", "lstm", "autokeras"]
            date_col (str): Cleaned date column name.
            value_col (str): Cleaned petrol price column name.
            days (int): Number of future days to forecast.
            linear_lags (int): Number of lag days for linear regression.
            arima_order (tuple): ARIMA order as (p, d, q).
            lstm_lookback (int): Number of previous days for LSTM.
            lstm_epochs (int): Training rounds for LSTM.
            lstm_batch_size (int): Batch size for LSTM.
            autokeras_lags (int): Number of lag features for AutoKeras.
            autokeras_max_trials (int): Number of AutoKeras model attempts.
            autokeras_epochs (int): Training rounds for AutoKeras.
            show_chart (bool): If True, display the integrated actual vs.
                forecast comparison chart across all models that were run.

        Returns:
            dict:
                {
                    "raw_data": df_raw,
                    "clean_data": df_petrol,
                    "forecast_data": forecast_all,
                    "chart": chart
                }
        """

        if models is None:
            models = ["linear", "arima"]

        valid_models = {"linear", "arima", "lstm", "autokeras"}
        selected_models = [model.lower().strip() for model in models]

        invalid_models = [model for model in selected_models if model not in valid_models]

        if invalid_models:
            raise ValueError(
                f"Unsupported model(s): {invalid_models}. "
                f"Please choose from: {sorted(valid_models)}"
            )

        print("Step 1: Loading raw petrol dataset...")
        df_raw = self.ingestion_bot.load_file(file_path)

        print("Step 2: Cleaning petrol dataset...")
        df_petrol = self.transform_bot.clean_petrol_data(df_raw)

        forecast_results = []

        print("Step 3: Running selected forecast models...")

        if "linear" in selected_models:
            print(" - Running Linear Regression forecast...")
            linear_forecast = self._run_linear_model(
                df=df_petrol,
                date_col=date_col,
                value_col=value_col,
                days=days,
                lags=linear_lags
            )
            forecast_results.append(linear_forecast)

        if "arima" in selected_models:
            print(" - Running ARIMA forecast...")
            arima_forecast = self._run_arima_model(
                df=df_petrol,
                date_col=date_col,
                value_col=value_col,
                days=days,
                arima_order=arima_order
            )
            forecast_results.append(arima_forecast)

        if "lstm" in selected_models:
            print(" - Running LSTM forecast...")
            lstm_forecast = self._run_lstm_model(
                df=df_petrol,
                date_col=date_col,
                value_col=value_col,
                days=days,
                lookback=lstm_lookback,
                lstm_epochs=lstm_epochs,
                lstm_batch_size=lstm_batch_size
            )
            forecast_results.append(lstm_forecast)

        if "autokeras" in selected_models:
            print(" - Running AutoKeras forecast...")
            autokeras_forecast = self._run_autokeras_model(
                df=df_petrol,
                date_col=date_col,
                value_col=value_col,
                days=days,
                autokeras_lags=autokeras_lags,
                autokeras_max_trials=autokeras_max_trials,
                autokeras_epochs=autokeras_epochs
            )
            forecast_results.append(autokeras_forecast)

        forecast_all = pd.concat(forecast_results, ignore_index=True)

        chart = None

        if show_chart:
            print("Step 4: Creating integrated actual vs. forecast comparison chart...")
            chart = self.visualization_bot.integrated_forecast_comparison_chart(
                actual_df=df_petrol,
                all_forecasts_df=forecast_all,
                date_col=date_col,
                value_col=value_col,
                forecast_col="forecast",
                model_col="model"
            )

        print("Pipeline completed successfully.")

        return {
            "raw_data": df_raw,
            "clean_data": df_petrol,
            "forecast_data": forecast_all,
            "chart": chart
        }
