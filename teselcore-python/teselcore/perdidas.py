"""
teselcore.perdidas — Funciones de pérdida (envolturas delgadas sobre C).
"""
from . import _lib as L
from .tensor import Tensor


def entropia_cruzada(logits: Tensor, objetivos: Tensor) -> Tensor:
    return Tensor(L.tc_entropia_cruzada(logits._ptr, objetivos._ptr))


def error_cuadratico_medio(pred: Tensor, objetivo: Tensor) -> Tensor:
    return Tensor(L.tc_error_cuadratico_medio(pred._ptr, objetivo._ptr))


def entropia_cruzada_binaria(pred: Tensor, objetivo: Tensor) -> Tensor:
    return Tensor(L.tc_entropia_cruzada_binaria(pred._ptr, objetivo._ptr))
