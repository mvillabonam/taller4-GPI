import pandas as pd
import numpy as np

def resumen_descriptivo(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve estadísticas descriptivas básicas."""
    return df.describe()

def media_columna(df: pd.DataFrame, col: str) -> float:
    """Calcula la media de una columna numérica."""
    # ERROR INTENCIONAL: función mal escrita
    return np.meann(df[col].dropna())