"""
src/features/cleaning.py
========================
Operações de limpeza e preparação determinística de features.

"""

from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config import (
    BINARY_COLS,
    BINARY_MAP,
    CAEC_CALC_MAP,
    NUMERIC_ORDINAL_COLS,
    ORDINAL_STRING_COLS,
)


# -----------------------------------------------------------------------------
# Operações training-only (alteram número de linhas)
# -----------------------------------------------------------------------------
def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas exatamente duplicadas.

    Em dataset SMOTE-augmented, duplicatas exatas frequentemente vêm de
    interpolação degenerada (λ≈0 ou λ≈1). Mantê-las inflaria classes minoritárias
    e geraria leakage de validação. A EDA identificou 24 duplicatas no dataset.

    Aplicar APENAS no split de treino, ANTES do train_test_split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset bruto.

    Returns
    -------
    pd.DataFrame
        Sem duplicatas exatas, index resetado.
    """
    return df.drop_duplicates().reset_index(drop=True)


# -----------------------------------------------------------------------------
# Operações in-pipeline (preservam shape, determinísticas, sem fit)
# -----------------------------------------------------------------------------
def round_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arredonda Age para inteiro.

    A EDA mostrou que 66% das idades têm decimais, oriundos do SMOTE.
    Em produção, o app coleta idade como inteiro — arredondar garante
    coerência entre distribuição treino e distribuição inferência.

    Returns DataFrame novo (não modifica o original).
    """
    df = df.copy()
    df['Age'] = np.round(df['Age']).astype(int)
    return df


def round_ordinals(
    df: pd.DataFrame,
    cols: Iterable[str] = NUMERIC_ORDINAL_COLS,
) -> pd.DataFrame:
    """
    Arredonda variáveis ordinais numéricas (FCVC, NCP, CH2O, FAF, TUE).

    Decisão fundamentada pela ablação empírica do notebook 01:
    - F1-macro contínuo:   0.9709 ± 0.0066
    - F1-macro arredondado: 0.9659 ± 0.0058
    - Diferença +0.005 — dentro do desvio dos folds, estatisticamente equivalente.

    Argumento decisivo: coerência treino-inferência. O app só pode receber
    inteiros (questionário discreto), então treinar em decimal cria distribution
    shift estrutural.
    """
    df = df.copy()
    for col in cols:
        df[col] = np.round(df[col]).astype(int)
    return df


def map_binary_cols(
    df: pd.DataFrame,
    cols: Iterable[str] = BINARY_COLS,
) -> pd.DataFrame:
    """
    Aplica BINARY_MAP em colunas binárias (yes/no → 1/0, Male/Female → 1/0).

    Encoding determinístico — não precisa de fit. Faz sentido aqui (e não
    num sklearn encoder) porque o mapping é universal e clínico, não derivado
    de estatísticas do treino.
    """
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(BINARY_MAP).astype(int)
    return df


def map_ordinal_string_cols(
    df: pd.DataFrame,
    cols: Iterable[str] = ORDINAL_STRING_COLS,
) -> pd.DataFrame:
    """
    Aplica CAEC_CALC_MAP em CAEC e CALC (no/Sometimes/Frequently/Always → 0..3).

    Tratamos como ordinal (numérico crescente) em vez de one-hot porque:
    - Há ordem natural inequívoca
    - One-hot perderia a relação de ordem que XGBoost pode explorar
    - Reduz dimensionalidade (1 coluna vs. 4 colunas)
    """
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(CAEC_CALC_MAP).astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestrador determinístico de preparação de features.

    Esta é a função que vai DENTRO do sklearn Pipeline (via FunctionTransformer).
    Funciona tanto para treino (lote completo) quanto para inferência (uma linha
    do app), porque todas as transformações são pontuais e determinísticas.

    Etapas (nesta ordem):
    1. Arredondar Age (decimais SMOTE → inteiro)
    2. Arredondar ordinais numéricas (FCVC, NCP, CH2O, FAF, TUE)
    3. Mapear binárias yes/no e Gender → 0/1
    4. Mapear ordinais string (CAEC, CALC) → 0..3

    Após isso, todas as colunas exceto MTRANS são numéricas e prontas para o
    ColumnTransformer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame "bruto" (já sem duplicatas e sem target, se for treino).

    Returns
    -------
    pd.DataFrame
        DataFrame com features prontas para encoding final.
    """
    df = round_age(df)
    df = round_ordinals(df)
    df = map_binary_cols(df)
    df = map_ordinal_string_cols(df)
    return df
