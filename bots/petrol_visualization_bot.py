import pandas as pd
import plotly.graph_objects as go


class PetrolVisualizationBot:
    """
    Petrol Visualization Bot

    Creates premium dark-theme interactive charts for:
    - Single-model actual vs forecast charts
    - Integrated multi-model comparison charts
    """

    def __init__(self):
        self.colors = {
            "actual": "#4FC3F7",              # light blue
            "Linear Regression": "#FF9800",  # orange
            "ARIMA": "#00E676",              # green
            "LSTM": "#E040FB",               # purple/magenta
            "AutoKeras": "#FFD740"           # yellow
        }

    def _apply_dark_theme(self, fig, title):
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white", size=13),
            title_font=dict(size=20, color="white"),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=False
            ),
            yaxis=dict(
                title="Price per Litre (€)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=False
            ),
            legend=dict(
                title="Legend",
                font=dict(color="white")
            ),
            hovermode="x unified"
        )

        return fig

    def single_model_forecast_chart(
        self,
        actual_df,
        forecast_df,
        date_col="date",
        value_col="price",
        forecast_col="forecast",
        model_name="Forecast Model"
    ):
        """
        Create a dark-theme actual vs forecast chart for one model.
        """

        actual_df = actual_df.copy()
        forecast_df = forecast_df.copy()

        actual_df[date_col] = pd.to_datetime(actual_df[date_col])
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=actual_df[date_col],
                y=actual_df[value_col],
                mode="lines",
                name="Actual E10 Price",
                line=dict(
                    color=self.colors["actual"],
                    width=2
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df[date_col],
                y=forecast_df[forecast_col],
                mode="lines+markers",
                name=f"{model_name} Forecast",
                line=dict(
                    color=self.colors.get(model_name, "#FF5252"),
                    width=3
                ),
                marker=dict(size=5)
            )
        )

        title = f"Germany E10 Petrol Price: Actual vs {model_name} Forecast"

        fig = self._apply_dark_theme(fig, title)

        fig.show()

        return fig

    def integrated_forecast_comparison_chart(
        self,
        actual_df,
        all_forecasts_df,
        date_col="date",
        value_col="price",
        forecast_col="forecast",
        model_col="model"
    ):
        """
        Create a dark-theme chart comparing actual prices with all forecast models.
        """

        actual_df = actual_df.copy()
        all_forecasts_df = all_forecasts_df.copy()

        actual_df[date_col] = pd.to_datetime(actual_df[date_col])
        all_forecasts_df[date_col] = pd.to_datetime(all_forecasts_df[date_col])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=actual_df[date_col],
                y=actual_df[value_col],
                mode="lines",
                name="Actual E10 Price",
                line=dict(
                    color=self.colors["actual"],
                    width=2
                )
            )
        )

        for model in all_forecasts_df[model_col].unique():
            model_df = all_forecasts_df[
                all_forecasts_df[model_col] == model
            ]

            fig.add_trace(
                go.Scatter(
                    x=model_df[date_col],
                    y=model_df[forecast_col],
                    mode="lines+markers",
                    name=f"{model} Forecast",
                    line=dict(
                        color=self.colors.get(model, "#FFFFFF"),
                        width=3
                    ),
                    marker=dict(size=4)
                )
            )

        title = "Germany E10 Petrol Price: Integrated Forecast Comparison"

        fig = self._apply_dark_theme(fig, title)

        fig.show()

        return fig
