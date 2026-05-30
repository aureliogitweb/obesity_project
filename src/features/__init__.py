"""Feature engineering, cleaning, and sklearn pipeline construction."""
from src.features.build_features import add_bmi
from src.features.cleaning import (
    drop_exact_duplicates,
    map_binary_cols,
    map_ordinal_string_cols,
    prepare_features,
    round_age,
    round_ordinals,
)
from src.features.pipeline import (
    build_pipeline,
    build_preprocessor,
    get_feature_names_after_transform,
)
from src.features.target_encoder import OrdinalTargetEncoder

__all__ = [
    'OrdinalTargetEncoder',
    'add_bmi',
    'build_pipeline',
    'build_preprocessor',
    'drop_exact_duplicates',
    'get_feature_names_after_transform',
    'map_binary_cols',
    'map_ordinal_string_cols',
    'prepare_features',
    'round_age',
    'round_ordinals',
]
