"""
teselcore.optimizador — Envolturas Python para SGD / Adam / AdamW.
Cada paso de actualización se ejecuta en C (tc_paso_optimizador). El struct
C tc_optimizador guarda el puntero crudo "tc_tensor** parametros" que le
pasamos, así que Python debe retener TANTO los Tensor (para que sus punteros
no mueran) COMO el propio array ctypes que empaqueta esos punteros (o el GC
de Python lo recolecta y opt->parametros queda apuntando a memoria liberada).
"""
import ctypes
from . import _lib as L
from .tensor import Tensor


def _empacar_parametros(parametros):
    n = len(parametros)
    arr = (L.TcTensorP * n)(*[p._ptr for p in parametros])
    return arr, n


class _OptimizadorBase:
    def __init__(self, ptr, parametros, arr_parametros):
        self._ptr = ptr
        self._parametros = list(parametros)      # mantiene vivos los Tensor
        self._arr_parametros = arr_parametros     # mantiene vivo el array ctypes

    def paso(self):
        L.tc_paso_optimizador(self._ptr)

    def cero_gradientes(self):
        L.tc_cero_gradientes_optimizador(self._ptr)

    def __del__(self):
        try:
            if getattr(self, "_ptr", None):
                L.tc_liberar_optimizador(self._ptr)
        except Exception:
            pass


class SGD(_OptimizadorBase):
    def __init__(self, parametros, lr=0.01, momento=0.9):
        arr, n = _empacar_parametros(parametros)
        ptr = L.tc_sgd(arr, n, lr, momento)
        super().__init__(ptr, parametros, arr)


class Adam(_OptimizadorBase):
    def __init__(self, parametros, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        arr, n = _empacar_parametros(parametros)
        ptr = L.tc_adam(arr, n, lr, beta1, beta2, eps)
        super().__init__(ptr, parametros, arr)


class AdamW(_OptimizadorBase):
    def __init__(self, parametros, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, decaimiento_pesos=0.01):
        arr, n = _empacar_parametros(parametros)
        ptr = L.tc_adamw(arr, n, lr, beta1, beta2, eps, decaimiento_pesos)
        super().__init__(ptr, parametros, arr)
