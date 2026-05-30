"""
src/features/pipeline.py
========================
Construção da pipeline sklearn end-to-end.

Esta é a peça que une cleaning + feature engineering + encoding + (futuramente) modelo.
A pipeline final aceita um DataFrame "bruto" (com colunas string em yes/no,
Female/Male etc.) e produz uma matriz numérica pronta para treinar/predizer.
"""

from typing import Literal, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build_features import add_bmi
from src.features.cleaning import prepare_features
from src.utils.config import (
    BINARY_COLS,
    CONTINUOUS_COLS_MODEL_A,
    CONTINUOUS_COLS_MODEL_B,
    NOMINAL_COLS,
    NUMERIC_ORDINAL_COLS,
    ORDINAL_STRING_COLS,
)

ModelType = Literal['A', 'B']


# -----------------------------------------------------------------------------
# Construtor do ColumnTransformer
# -----------------------------------------------------------------------------
def build_preprocessor(
    model_type: ModelType = 'A',
    scale_continuous: bool = False,
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer apropriado para o modelo escolhido.

    -------
    ColumnTransformer não-fitado.
    """
    if model_type not in ('A', 'B'):
        raise ValueError(f"model_type deve ser 'A' ou 'B', recebido {model_type!r}")

    # Contínuas dependem do modelo
    if model_type == 'A':
        continuous_cols = CONTINUOUS_COLS_MODEL_A      # Age, Height, Weight, IMC
    else:
        continuous_cols = CONTINUOUS_COLS_MODEL_B      # Age, Height

    # Já estão como int após prepare_features — só passam
    passthrough_cols = (
        BINARY_COLS              # Gender, family_history, FAVC, SMOKE, SCC (0/1)
        + ORDINAL_STRING_COLS    # CAEC, CALC (0..3)
        + NUMERIC_ORDINAL_COLS   # FCVC, NCP, CH2O, FAF, TUE (inteiros)
    )

    continuous_transformer = (
        StandardScaler() if scale_continuous else 'passthrough'
    )

    return ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_cols),
            ('passthrough', 'passthrough', passthrough_cols),
            ('nominal', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False,
            ), NOMINAL_COLS),
        ],
        remainder='drop',          # explícito: qualquer coluna não listada é descartada
        verbose_feature_names_out=False,
    )


# -----------------------------------------------------------------------------
# Construtor da pipeline completa
# -----------------------------------------------------------------------------
def build_pipeline(
    model_type: ModelType = 'A',
    estimator: Optional[BaseEstimator] = None,
    scale_continuous: bool = False,
) -> Pipeline:
    """
    Constrói a pipeline sklearn completa.
    -------
    sklearn.pipeline.Pipeline não-fitada.
    """
    steps = [
        # 1. Limpeza determinística (arredondamento + mapeamento)
        ('prepare', FunctionTransformer(prepare_features, validate=False)),
    ]

    # 2. Feature engineering — IMC só no Modelo A
    if model_type == 'A':
        steps.append(('add_bmi', FunctionTransformer(add_bmi, validate=False)))

    # 3. Encoding final (numéricas → opcionalmente escaladas, MTRANS → one-hot)
    steps.append(('preprocessor', build_preprocessor(model_type, scale_continuous)))

    # 4. Modelo final, se fornecido
    if estimator is not None:
        steps.append(('model', estimator))

    return Pipeline(steps)


# -----------------------------------------------------------------------------
# Inspeção (útil em notebooks / debugging)
# -----------------------------------------------------------------------------
def get_feature_names_after_transform(pipeline: Pipeline) -> list[str]:
    """
    Retorna os nomes das colunas após o preprocessor (one-hot expandido).
    ------
    ValueError
        Se a pipeline não tiver step 'preprocessor' ou não estiver fitada.
    """
    if 'preprocessor' not in pipeline.named_steps:
        raise ValueError("Pipeline não tem step 'preprocessor'.")
    preprocessor = pipeline.named_steps['preprocessor']
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        raise ValueError(
            'Não foi possível obter feature names. '
            'Garanta que a pipeline foi fitada com pipeline.fit(X_train) antes.'
        ) from e
