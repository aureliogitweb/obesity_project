"""
src/models/train.py
===================
Lógica de treino e validação cruzada.

Todas usam RANDOM_STATE fixo e operam sobre raw DataFrames (a pipeline cuida
do preprocessing). O notebook apenas chama estas funções.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_pipeline
from src.models.registry import (
    ESTIMATOR_FACTORIES,
    XGB_PARAM_DISTRIBUTIONS,
    make_xgboost,
)
from src.utils.config import (
    CV_FOLDS,
    N_ITER_RANDOM_SEARCH,
    RANDOM_STATE,
)


def _make_cv() -> StratifiedKFold:
    """StratifiedKFold padrão do projeto — sempre o mesmo, reprodutível."""
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def cross_validate_model(
    model_name: str,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict:
    """
    Avalia um modelo via StratifiedKFold CV.

    Constrói a pipeline (com scaling se o estimator pedir), roda CV de 5 folds
    e retorna métricas médias.

    Parameters
    ----------
    model_name : str
        Chave em ESTIMATOR_FACTORIES ('LogisticRegression', 'RandomForest', 'XGBoost').
    model_type : str
        'A' ou 'B'.
    X_train : pd.DataFrame
        Features de treino (raw — a pipeline preprocessa).
    y_train : np.ndarray
        Target encodado (inteiros ordinais).

    Returns
    -------
    dict
        {model, variant, f1_macro_mean, f1_macro_std, accuracy_mean, accuracy_std}
    """
    spec = ESTIMATOR_FACTORIES[model_name]
    estimator = spec['factory']()
    scale = spec['scale']

    pipeline = build_pipeline(
        model_type=model_type,
        estimator=estimator,
        scale_continuous=scale,
    )

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=_make_cv(),
        scoring=['f1_macro', 'accuracy'],
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        'model': model_name,
        'variant': model_type,
        'f1_macro_mean':  cv_results['test_f1_macro'].mean(),
        'f1_macro_std':   cv_results['test_f1_macro'].std(),
        'accuracy_mean':  cv_results['test_accuracy'].mean(),
        'accuracy_std':   cv_results['test_accuracy'].std(),
    }


def tune_xgboost(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = N_ITER_RANDOM_SEARCH,
) -> Tuple[Pipeline, dict]:
    """
    RandomizedSearchCV sobre o XGBoost de um model_type.

    Apenas o XGBoost recebe tuning (decisão da revisão crítica). Máximo de
    `n_iter` combinações, F1-macro como scoring, 5-fold estratificado.
        Retorna a pipeline fitada com os melhores hiperparâmetros e um dicionário
    """
    pipeline = build_pipeline(
        model_type=model_type,
        estimator=make_xgboost(),
        scale_continuous=False,
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=_make_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    search_info = {
        'best_params': search.best_params_,
        'best_cv_f1_macro': search.best_score_,
        'n_iter': n_iter,
    }
    return search.best_estimator_, search_info


def fit_full_pipeline(
    model_name: str,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Fita a pipeline completa (preprocessing + modelo) no treino.

    Usado para treinar os modelos não-tunados (LogReg, RF) para a avaliação
    final, ou qualquer modelo com defaults.

    Returns
    -------
    Pipeline fitada.
    """
    spec = ESTIMATOR_FACTORIES[model_name]
    pipeline = build_pipeline(
        model_type=model_type,
        estimator=spec['factory'](),
        scale_continuous=spec['scale'],
    )
    pipeline.fit(X_train, y_train)
    return pipeline
