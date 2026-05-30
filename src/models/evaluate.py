"""
src/models/evaluate.py
=====================
Métricas de avaliação e visualizações para os modelos.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.features.pipeline import get_feature_names_after_transform
from src.features.target_encoder import OrdinalTargetEncoder
from src.utils.config import CLASS_ORDER, CRITICAL_CLASSES


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_int: np.ndarray,
    target_encoder: OrdinalTargetEncoder,
    model_name: str,
    model_type: str,
) -> dict:
    """
    Avalia uma pipeline fitada no conjunto de teste.

    Parameters
    ----------

    -------
    dict
        Métricas consolidadas incluindo recall por classe crítica.
    """
    y_pred = pipeline.predict(X_test)

    f1_macro = f1_score(y_test_int, y_pred, average='macro')
    accuracy = accuracy_score(y_test_int, y_pred)

    # Recall por classe (índice → nome)
    recall_per_class = recall_score(
        y_test_int, y_pred, average=None,
        labels=list(range(len(CLASS_ORDER))),
        zero_division=0,
    )
    recall_by_name = {
        target_encoder.inverse_transform([i])[0]: recall_per_class[i]
        for i in range(len(CLASS_ORDER))
    }

    result = {
        'model': model_name,
        'variant': model_type,
        'test_f1_macro': f1_macro,
        'test_accuracy': accuracy,
    }
    # Recall das classes clinicamente críticas
    for cls in CRITICAL_CLASSES:
        result[f'recall_{cls}'] = recall_by_name.get(cls, np.nan)

    result['recall_by_class'] = recall_by_name
    return result


def get_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_int: np.ndarray,
) -> pd.DataFrame:
    """Matriz de confusão como DataFrame rotulado em ordem clínica."""
    y_pred = pipeline.predict(X_test)
    labels = list(range(len(CLASS_ORDER)))
    cm = confusion_matrix(y_test_int, y_pred, labels=labels)
    return pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)


def get_classification_report_df(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_int: np.ndarray,
) -> pd.DataFrame:
    """classification_report como DataFrame, com nomes de classe."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test_int, y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def get_xgboost_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extrai feature importance de uma pipeline XGBoost fitada.

    Mapeia as importâncias aos nomes de feature pós-encoding.

    Returns
    -------
    pd.DataFrame
        Colunas: feature, importance — ordenado desc.
    """
    model = pipeline.named_steps['model']
    feature_names = get_feature_names_after_transform(pipeline)
    importances = model.feature_importances_

    return (
        pd.DataFrame({'feature': feature_names, 'importance': importances})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = 'Matriz de Confusão',
    normalize: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Heatmap da matriz de confusão."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    data = cm_df.copy()
    fmt = 'd'
    if normalize:
        data = data.div(data.sum(axis=1), axis=0).fillna(0)
        fmt = '.2f'

    sns.heatmap(
        data, annot=True, fmt=fmt, cmap='Blues',
        cbar_kws={'label': 'proporção' if normalize else 'contagem'},
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title(title, fontweight='bold')
    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = 'Feature Importance — XGBoost',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Barh das top-N feature importances."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    top = importance_df.head(top_n).iloc[::-1]
    ax.barh(top['feature'], top['importance'], color='steelblue',
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Importância (gain)')
    ax.set_title(title, fontweight='bold')
    return ax
