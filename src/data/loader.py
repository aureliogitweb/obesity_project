"""
src/data/loader.py
==================
Carregamento e validação de schema do dataset bruto.

Responsabilidades:
- Ler o CSV de `data/raw/obesity.csv` (path resolvido via config.py)
- Validar shape, presença de colunas esperadas e domínio das categóricas
- Falhar cedo (raise) se o arquivo estiver corrompido ou fora do schema

Princípio: validação na entrada evita propagar erros silenciosos pela pipeline.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import (
    BINARY_MAP,
    CAEC_CALC_MAP,
    CLASS_ORDER,
    EXPECTED_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COL,
)


class SchemaValidationError(Exception):
    """Erro de schema do dataset bruto — disparado por validate_schema."""


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carrega o CSV bruto e devolve um DataFrame.

    Parameters
    ----------
    path : Path, opcional
        Caminho customizado. Se None, usa RAW_DATA_PATH do config.

    Returns
    -------
    pd.DataFrame
        Dataset bruto, sem qualquer transformação.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    path = Path(path) if path is not None else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f'Dataset não encontrado em {path}. '
            f'Coloque obesity.csv em data/raw/ na raiz do projeto.'
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Valida o schema do DataFrame contra o esperado.

    Verifica:
    - Presença e ordem das colunas
    - Domínio dos valores categóricos (yes/no, Male/Female, classes do target)
    - Tipos numéricos onde se espera numérico

    Não retorna nada — apenas levanta SchemaValidationError se algo não bater.
    Não modifica o DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto a validar.

    Raises
    ------
    SchemaValidationError
        Se qualquer expectativa de schema for violada.
    """
    # 1. Presença de colunas
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(f'Colunas ausentes: {sorted(missing)}')

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        raise SchemaValidationError(f'Colunas inesperadas: {sorted(extra)}')

    # 2. Domínios das categóricas binárias / target
    expected_domains = {
        'Gender':         {'Male', 'Female'},
        'family_history': set(BINARY_MAP) - {'Male', 'Female'},     # {'yes', 'no'}
        'FAVC':           set(BINARY_MAP) - {'Male', 'Female'},
        'SMOKE':          set(BINARY_MAP) - {'Male', 'Female'},
        'SCC':            set(BINARY_MAP) - {'Male', 'Female'},
        'CAEC':           set(CAEC_CALC_MAP),
        'CALC':           set(CAEC_CALC_MAP),
        'MTRANS':         {'Public_Transportation', 'Walking',
                           'Automobile', 'Motorbike', 'Bike'},
        TARGET_COL:       set(CLASS_ORDER),
    }

    for col, expected in expected_domains.items():
        actual = set(df[col].dropna().unique())
        unexpected = actual - expected
        if unexpected:
            raise SchemaValidationError(
                f'Coluna {col!r} contém valores fora do domínio esperado: '
                f'{sorted(unexpected)}. Esperados: {sorted(expected)}'
            )

    # 3. Tipos numéricos
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaValidationError(
                f'Coluna {col!r} deveria ser numérica, mas é {df[col].dtype}.'
            )


def load_and_validate(path: Optional[Path] = None) -> pd.DataFrame:
    """Conveniência: carrega e valida em uma única chamada."""
    df = load_raw_data(path)
    validate_schema(df)
    return df
