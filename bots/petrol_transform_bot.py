import pandas as pd


class PetrolTransformBot:
    """
    Cleans German petrol price data for forecasting.
    """

    def clean_petrol_data(self, df: pd.DataFrame) -> pd.DataFrame:

        out = df.copy()

        # Clean column names
        out.columns = out.columns.str.strip().str.lower()

        # Convert date
        out["date"] = pd.to_datetime(
            out["tag"],
            format="%d-%m-%y",
            errors="coerce"
        )

        # Use E10 as petrol price
        out["price"] = pd.to_numeric(out["avg_e10"], errors="coerce")

        # Keep only forecasting columns
        out = out[["date", "price"]]

        # Remove invalid rows
        out = out.dropna(subset=["date", "price"])

        # Sort by date
        out = out.sort_values("date")

        # Remove duplicate dates, just in case
        out = out.drop_duplicates(subset=["date"])

        # Set date as index
        out = out.set_index("date")

        # Create complete daily date range
        out = out.asfreq("D")

        # Fill missing daily prices using interpolation
        out["price"] = out["price"].interpolate(method="linear")

        # Reset index
        out = out.reset_index()

        return out
