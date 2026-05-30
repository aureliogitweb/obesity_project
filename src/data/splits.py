"""
src/data/splits.py
==================
Lógica de split estratificado treino/validação/teste.

Encapsula a divisão 70/15/15 para garantir consistência entre notebooks e
execuções. Estratificado pelo target, com RANDOM_STATE fixo para
reprodutibilidade.

Sequência (two-step split):
1. Separa 15% para teste (`test`) — nunca mais tocado até a avaliação final
2. Do restante (85%), separa ~17.6% para validação (= 15% do total)
3. Sobra 70% para treino

Por que dois steps?
- train_test_split do sklearn não tem split 3-way nativo.
- Manter test isolado é a regra mais importante de toda a pipeline.
"""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE


@dataclass
class DataSplits:
    """
    Container nomeado dos splits.

    Atributos
    ---------
    X_train, y_train : treino (70%)
    X_val, y_val     : validação (15%)
    X_test, y_test   : teste (15%)
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series

    def summary(self) -> pd.DataFrame:
        """Retorna um DataFrame com tamanhos e proporções de cada split."""
        sizes = {
            'train': len(self.X_train),
            'val':   len(self.X_val),
            'test':  len(self.X_test),
        }
        total = sum(sizes.values())
        return pd.DataFrame({
            'split':     list(sizes.keys()),
            'n_rows':    list(sizes.values()),
            'pct_total': [100 * n / total for n in sizes.values()],
        })


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    val_size:  float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """
    Split estratificado 70/15/15.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset já limpo (sem duplicatas). Deve conter o target_col.
    target_col : str
        Nome da coluna alvo.
    test_size : float
        Fração do total para o conjunto de teste.
    val_size : float
        Fração do total para validação.
    random_state : int
        Semente fixa para reprodutibilidade.

    Returns
    -------
    DataSplits
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Passo 1: separa test final (sempre isolado a partir daqui)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Passo 2: divide o restante em train / val
    # val_size é fração do TOTAL
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=random_state,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
