"""Model training, evaluation, and persistence."""
from src.models.evaluate import (
    evaluate_on_test,
    get_classification_report_df,
    get_confusion_matrix,
    get_xgboost_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.models.persistence import (
    load_best_model,
    load_splits,
    save_best_model,
    save_metrics_detailed,
    save_metrics_table,
)
from src.models.registry import (
    ESTIMATOR_FACTORIES,
    XGB_PARAM_DISTRIBUTIONS,
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
)
from src.models.train import (
    cross_validate_model,
    fit_full_pipeline,
    tune_xgboost,
)

__all__ = [
    'ESTIMATOR_FACTORIES',
    'XGB_PARAM_DISTRIBUTIONS',
    'cross_validate_model',
    'evaluate_on_test',
    'fit_full_pipeline',
    'get_classification_report_df',
    'get_confusion_matrix',
    'get_xgboost_feature_importance',
    'load_best_model',
    'load_splits',
    'make_logistic_regression',
    'make_random_forest',
    'make_xgboost',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'save_best_model',
    'save_metrics_detailed',
    'save_metrics_table',
    'tune_xgboost',
]
