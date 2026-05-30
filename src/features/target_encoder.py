"""
src/features/target_encoder.py
==============================
Encoder para a variável alvo `Obesity`.
"""

from typing import Iterable, Optional

import numpy as np

from src.utils.config import CLASS_ORDER


class OrdinalTargetEncoder:
    """
    Encoder para variável alvo `Obesity`, convertendo rótulos string → inteiros ordinais.
    """

    def __init__(self, class_order: Iterable[str] = CLASS_ORDER):
        self.classes_ = np.array(list(class_order))
        self._to_int = {c: i for i, c in enumerate(self.classes_)}
        self._to_label = {i: c for c, i in self._to_int.items()}

    def fit(self, y: Optional[Iterable[str]] = None) -> 'OrdinalTargetEncoder':
        """No-op — o mapeamento é fixo via CLASS_ORDER. Mantido por compatibilidade."""
        if y is not None:
            unknown = set(y) - set(self.classes_)
            if unknown:
                raise ValueError(
                    f'Encontradas classes não previstas em CLASS_ORDER: {sorted(unknown)}'
                )
        return self

    def transform(self, y: Iterable[str]) -> np.ndarray:
        """Converte rótulos string → inteiros ordinais."""
        try:
            return np.array([self._to_int[v] for v in y], dtype=int)
        except KeyError as e:
            raise ValueError(f'Classe desconhecida: {e.args[0]!r}') from e

    def inverse_transform(self, y: Iterable[int]) -> np.ndarray:
        """Converte inteiros ordinais → rótulos string."""
        try:
            return np.array([self._to_label[int(v)] for v in y])
        except KeyError as e:
            raise ValueError(f'Inteiro fora do range válido: {e.args[0]!r}') from e

    def fit_transform(self, y: Iterable[str]) -> np.ndarray:
        return self.fit(y).transform(y)

    def __repr__(self) -> str:
        return f'OrdinalTargetEncoder(classes={list(self.classes_)})'
