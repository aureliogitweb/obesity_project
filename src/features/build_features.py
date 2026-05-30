"""
src/features/build_features.py
==============================
Feature engineering específica do projeto.

Por enquanto, a única feature engineered é o IMC (Modelo A).

"""

import pandas as pd


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o IMC (Weight / Height²) e adiciona como coluna.

    Parameters
    ----------
    df : pd.DataFrame
        Deve conter as colunas 'Weight' (kg) e 'Height' (m).

    Returns
    -------
    pd.DataFrame
        Com coluna adicional 'IMC'.

    Raises
    ------
    KeyError
        Se Weight ou Height não estiverem presentes.
    """
    if 'Weight' not in df.columns or 'Height' not in df.columns:
        raise KeyError(
            "add_bmi requer as colunas 'Weight' e 'Height' no DataFrame."
        )
    df = df.copy()
    df['IMC'] = df['Weight'] / (df['Height'] ** 2)
    return df
