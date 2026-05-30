"""
src/models/registry.py
======================
Registro central dos estimators e seus espaços de busca de hiperparâmetros.

Define os 3 algoritmos do projeto:
- Logistic Regression — baseline interpretável (precisa de scaling)
- Random Forest — comparativo robusto
- XGBoost — modelo de produção (único que recebe RandomizedSearchCV)

Centralizar isso aqui mantém o notebook limpo (só orquestra) e permite
reusar os mesmos estimators no app/scripts futuros.
"""

from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.config import N_CLASSES, RANDOM_STATE


# -----------------------------------------------------------------------------
# Factories dos estimators (cada chamada cria uma instância nova e limpa)
# -----------------------------------------------------------------------------
def make_logistic_regression() -> LogisticRegression:
    """
    Logistic Regression multinomial.

    `max_iter` alto para garantir convergência com features escaladas.
    Requer `scale_continuous=True` na pipeline (sensível a escala).
    """
    return LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        random_state=RANDOM_STATE,
    )


def make_random_forest() -> RandomForestClassifier:
    """
    Random Forest com defaults robustos.

    Insensível a escala — não precisa de scaling na pipeline.
    Usado como comparativo; não recebe tuning pesado.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgboost() -> XGBClassifier:
    """
    XGBoost multiclasse — modelo de produção.

    Hiperparâmetros base (serão refinados via RandomizedSearchCV).
    `tree_method='hist'` para velocidade; insensível a escala.
    """
    return XGBClassifier(
        objective='multi:softprob',
        num_class=N_CLASSES,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# Mapa nome → factory. `scale` indica se a pipeline precisa de StandardScaler.
ESTIMATOR_FACTORIES: Dict[str, Dict[str, object]] = {
    'LogisticRegression': {'factory': make_logistic_regression, 'scale': True},
    'RandomForest':       {'factory': make_random_forest,       'scale': False},
    'XGBoost':            {'factory': make_xgboost,             'scale': False},
}


# -----------------------------------------------------------------------------
# Espaço de busca do XGBoost (RandomizedSearchCV)
# -----------------------------------------------------------------------------
# Prefixo 'model__' porque o estimator é o step 'model' dentro do Pipeline.
# Faixas conservadoras: max_depth limitado a 7 (dataset de ~2k linhas, evita overfit).
XGB_PARAM_DISTRIBUTIONS: Dict[str, list] = {
    'model__n_estimators':     [100, 200, 300, 500],
    'model__max_depth':        [3, 4, 5, 6, 7],
    'model__learning_rate':    [0.01, 0.05, 0.1, 0.15],
    'model__subsample':        [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__reg_alpha':        [0, 0.1, 0.5, 1.0],
    'model__reg_lambda':       [0.1, 1.0, 5.0, 10.0],
    'model__min_child_weight': [1, 3, 5],
}
