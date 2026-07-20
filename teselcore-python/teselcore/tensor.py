"""
teselcore.tensor — Envoltura Python de tc_tensor
=================================================
Cada Tensor es un puntero real a memoria administrada por libteselcore.so.
Toda operación (+, -, *, matmul, activaciones, etc.) invoca la función C
correspondiente; NumPy solo se usa para exponer una vista de los datos,
nunca para recalcular la operación en Python.
"""
import ctypes
import numpy as np
from . import _lib as L

_NP_DTYPE = {L.TC_FLOAT32: np.float32, L.TC_INT32: np.int32}


class Tensor:
    __slots__ = ("_ptr", "_dueno")

    def __init__(self, ptr, dueno=True):
        """ptr: ctypes.POINTER(TcTensor) ya creado por una función tc_*."""
        if not ptr:
            raise ValueError("Puntero de tensor nulo (la llamada C falló)")
        self._ptr = ptr
        self._dueno = dueno

    # ── Constructores que delegan 100% en C ──────────────────────────
    @classmethod
    def ceros(cls, forma):
        f = (ctypes.c_int * len(forma))(*forma)
        return cls(L.tc_ceros(f, len(forma)))

    @classmethod
    def unos(cls, forma):
        f = (ctypes.c_int * len(forma))(*forma)
        return cls(L.tc_unos(f, len(forma)))

    @classmethod
    def aleatorio_uniforme(cls, forma):
        f = (ctypes.c_int * len(forma))(*forma)
        return cls(L.tc_aleatorio_uniforme(f, len(forma)))

    @classmethod
    def aleatorio_normal(cls, forma):
        f = (ctypes.c_int * len(forma))(*forma)
        return cls(L.tc_aleatorio_normal(f, len(forma)))

    @classmethod
    def escalar(cls, valor):
        return cls(L.tc_escalar(ctypes.c_float(valor)))

    @classmethod
    def desde_numpy(cls, arr: np.ndarray, requiere_grad=False):
        """Copia los datos de un ndarray a un tc_tensor real (float32)."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        forma = (ctypes.c_int * arr.ndim)(*arr.shape)
        ptr_datos = arr.ctypes.data_as(ctypes.c_void_p)
        t = cls(L.tc_desde_datos(ptr_datos, forma, arr.ndim, L.TC_FLOAT32))
        t.requiere_grad = requiere_grad
        return t

    # ── Propiedades ────────────────────────────────────────────────────
    @property
    def forma(self):
        s = self._ptr.contents
        return tuple(s.forma[i] for i in range(s.ndim))

    @property
    def ndim(self):
        return self._ptr.contents.ndim

    @property
    def total(self):
        return self._ptr.contents.total

    @property
    def requiere_grad(self):
        return bool(self._ptr.contents.requiere_grad)

    @requiere_grad.setter
    def requiere_grad(self, valor):
        self._ptr.contents.requiere_grad = 1 if valor else 0

    def elemento(self) -> float:
        return L.tc_elemento(self._ptr)

    def numpy(self) -> np.ndarray:
        """Vista SIN COPIA sobre el buffer de datos administrado por C.
        Válida mientras el Tensor no haya sido liberado."""
        s = self._ptr.contents
        dtype = _NP_DTYPE.get(s.tipo, np.float32)
        buf_p = ctypes.cast(s.datos, ctypes.POINTER(ctypes.c_float * s.total))
        return np.ctypeslib.as_array(buf_p.contents).reshape(self.forma).astype(dtype, copy=False)

    def gradiente(self) -> "np.ndarray | None":
        """Vista sin copia sobre el gradiente acumulado en C, o None si aún no existe."""
        s = self._ptr.contents
        if not s.gradiente:
            return None
        buf_p = ctypes.cast(s.gradiente, ctypes.POINTER(ctypes.c_float * s.total))
        return np.ctypeslib.as_array(buf_p.contents).reshape(self.forma)

    def cero_gradiente(self):
        L.tc_cero_gradiente(self._ptr)

    def imprimir(self):
        L.tc_imprimir(self._ptr)

    def __repr__(self):
        return f"Tensor(forma={self.forma}, requiere_grad={self.requiere_grad})"

    # ── Operadores — cada uno llama directamente al núcleo C ─────────
    def __add__(self, o): return Tensor(L.tc_sumar(self._ptr, o._ptr))
    def __sub__(self, o): return Tensor(L.tc_restar(self._ptr, o._ptr))
    def __mul__(self, o): return Tensor(L.tc_multiplicar(self._ptr, o._ptr))
    def __truediv__(self, o): return Tensor(L.tc_dividir(self._ptr, o._ptr))
    def __neg__(self): return Tensor(L.tc_negar(self._ptr))
    def potencia(self, exp: float): return Tensor(L.tc_potencia(self._ptr, exp))
    def logaritmo(self): return Tensor(L.tc_logaritmo(self._ptr))
    def exponencial(self): return Tensor(L.tc_exponencial(self._ptr))
    def raiz(self): return Tensor(L.tc_raiz(self._ptr))
    def valor_absoluto(self): return Tensor(L.tc_valor_absoluto(self._ptr))

    # ── Reducción ──────────────────────────────────────────────────────
    def suma(self, dim=-1, mantener_dim=0): return Tensor(L.tc_suma_dimension(self._ptr, dim, mantener_dim))
    def media(self, dim=-1, mantener_dim=0): return Tensor(L.tc_media_dimension(self._ptr, dim, mantener_dim))
    def maximo(self, dim=-1, mantener_dim=0): return Tensor(L.tc_maximo_dimension(self._ptr, dim, mantener_dim))
    def minimo(self, dim=-1, mantener_dim=0): return Tensor(L.tc_minimo_dimension(self._ptr, dim, mantener_dim))

    # ── Álgebra lineal ─────────────────────────────────────────────────
    def matmul(self, o): return Tensor(L.tc_multiplicacion_matricial(self._ptr, o._ptr))
    def transponer(self, dim0=0, dim1=1): return Tensor(L.tc_transponer(self._ptr, dim0, dim1))

    def reformar(self, *nueva_forma):
        f = (ctypes.c_int * len(nueva_forma))(*nueva_forma)
        return Tensor(L.tc_reformar(self._ptr, f, len(nueva_forma)))

    def aplanar(self, dim_inicio=0): return Tensor(L.tc_aplanar(self._ptr, dim_inicio))

    # ── Activaciones ───────────────────────────────────────────────────
    def relu(self): return Tensor(L.tc_relu(self._ptr))
    def relu_con_fuga(self, alfa=0.01): return Tensor(L.tc_relu_con_fuga(self._ptr, alfa))
    def sigmoide(self): return Tensor(L.tc_sigmoide(self._ptr))
    def tangente_hiperbolica(self): return Tensor(L.tc_tangente_hiperbolica(self._ptr))
    def softmax(self, dim=-1): return Tensor(L.tc_softmax(self._ptr, dim))
    def gelu(self): return Tensor(L.tc_gelu(self._ptr))
    def silu(self): return Tensor(L.tc_silu(self._ptr))

    # ── Capas ──────────────────────────────────────────────────────────
    def lineal(self, pesos: "Tensor", sesgo: "Tensor" = None):
        sp = sesgo._ptr if sesgo is not None else None
        return Tensor(L.tc_lineal(self._ptr, pesos._ptr, sp))

    def conv2d(self, kernel: "Tensor", sesgo: "Tensor" = None, paso=1, relleno=0):
        sp = sesgo._ptr if sesgo is not None else None
        return Tensor(L.tc_conv2d(self._ptr, kernel._ptr, sp, paso, relleno))

    def agrupacion_max(self, kernel, paso): return Tensor(L.tc_agrupacion_max(self._ptr, kernel, paso))
    def agrupacion_promedio(self, kernel, paso): return Tensor(L.tc_agrupacion_promedio(self._ptr, kernel, paso))

    def normalizacion_lote(self, gamma, beta, media_mov, var_mov, eps=1e-5, momento=0.1, entrenando=True):
        return Tensor(L.tc_normalizacion_lote(self._ptr, gamma._ptr, beta._ptr,
                      media_mov._ptr, var_mov._ptr, eps, momento, int(entrenando)))

    def abandono(self, p, entrenando=True):
        return Tensor(L.tc_abandono(self._ptr, p, int(entrenando)))

    # ── Autograd ───────────────────────────────────────────────────────
    def retropropagar(self):
        L.tc_retropropagar(self._ptr)

    # ── Memoria ────────────────────────────────────────────────────────
    def liberar(self):
        if self._dueno and self._ptr:
            L.tc_liberar(self._ptr)
            self._ptr = None

    def __del__(self):
        try:
            self.liberar()
        except Exception:
            pass  # evitar excepciones durante el cierre del intérprete
