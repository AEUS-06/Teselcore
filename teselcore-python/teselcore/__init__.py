"""
TeselCore — bindings Python (ctypes) sobre el núcleo C libteselcore.so
========================================================================
Ninguna operación matemática se reimplementa aquí: cada llamada delega
en la librería compartida compilada desde teselcore-nucleo/.

Uso:
    from teselcore import Tensor, Teselacion, KernelPenrose, Modelo, SGD, Adam
"""
from . import _lib
from .tensor import Tensor
from .penrose import Teselacion, KernelPenrose
from .modelo import Modelo
from .optimizador import SGD, Adam, AdamW
from . import perdidas

TC_PHI = (1.0 + 5 ** 0.5) / 2.0
TC_PENROSE_KITE = _lib.TC_PENROSE_KITE
TC_PENROSE_DART = _lib.TC_PENROSE_DART


def imprimir_info():
    _lib.tc_imprimir_info()


def semilla_aleatoria(semilla: int):
    _lib.tc_semilla_aleatoria(semilla)


__all__ = [
    "Tensor", "Teselacion", "KernelPenrose", "Modelo",
    "SGD", "Adam", "AdamW", "perdidas",
    "TC_PHI", "TC_PENROSE_KITE", "TC_PENROSE_DART",
    "imprimir_info", "semilla_aleatoria",
]
