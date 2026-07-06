import pandas as pd
from pathlib import Path


class IngestionBot:
    """
    A reusable bot for loading CSV and Excel files.
    It also standardizes column names for easier analysis.
    """

    def load_file(self, path: str, **kwargs):
        """
        Load a CSV or Excel file into a pandas DataFrame.

        Parameters:
            path (str): File path of the dataset.
            **kwargs: Optional settings for pandas read functions.

        Returns:
            pd.DataFrame: Loaded dataset with cleaned column names.
        """

        file_path = Path(path)
        file_extension = file_path.suffix.lower()

        if file_extension == ".csv":
            df = pd.read_csv(file_path, **kwargs)

        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)

        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                "Supported file types are .csv, .xlsx, and .xls."
            )

        df.columns = (
            df.columns.astype(str)
                      .str.strip()
                      .str.lower()
                      .str.replace(" ", "_", regex=False)
        )

        return df
