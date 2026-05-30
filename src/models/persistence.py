"""
src/models/persistence.py
========================
Persistência de modelos e métricas.

Centraliza I/O de artefatos para que notebook e app usem os mesmos paths/formatos.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_FILENAME,
    METRICS_FILENAME,
    METRICS_JSON_FILENAME,
    PROCESSED_DATA_DIR,
    SPLITS_FILENAME,
)


def load_splits() -> dict:
    """Carrega os splits persistidos pelo notebook 02."""
    path = PROCESSED_DATA_DIR / SPLITS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f'Splits não encontrados em {path}. Rode o notebook 02 primeiro.'
        )
    return joblib.load(path)


def save_best_model(pipeline: Pipeline, filename: str = BEST_MODEL_FILENAME) -> Path:
    """Salva a pipeline final completa (preprocessing + modelo)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(pipeline, path)
    return path


def load_best_model(filename: str = BEST_MODEL_FILENAME) -> Pipeline:
    """Carrega a pipeline final (para o app Streamlit, etapas futuras)."""
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f'Modelo não encontrado em {path}.')
    return joblib.load(path)


def save_metrics_table(df: pd.DataFrame, filename: str = METRICS_FILENAME) -> Path:
    """Salva a tabela consolidada de métricas em CSV."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_metrics_detailed(data: dict, filename: str = METRICS_JSON_FILENAME) -> Path:
    """Salva métricas detalhadas (recall por classe, best_params) em JSON."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename

    def _coerce(o: Any):
        """Torna numpy types serializáveis."""
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f'Não serializável: {type(o)}')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_coerce)
    return path
