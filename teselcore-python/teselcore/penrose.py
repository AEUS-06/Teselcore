"""
teselcore.penrose — Envolturas Python para teselación y kernel de Penrose.
Toda la geometría (subdivisión, vecindad, convolución) ocurre en C;
aquí solo se exponen punteros y se facilita el acceso a atributos.
"""
import ctypes
from . import _lib as L
from .tensor import Tensor


class Teselacion:
    def __init__(self, nivel: int, escala: float = 1.0):
        self._ptr = L.tc_crear_teselacion(nivel, escala)
        if not self._ptr:
            raise RuntimeError("tc_crear_teselacion falló")

    @property
    def num_tejas(self) -> int:
        return self._ptr.contents.num_tejas

    @property
    def nivel(self) -> int:
        return self._ptr.contents.nivel

    def exportar_svg(self, ruta: str) -> bool:
        return L.tc_exportar_svg_teselacion(self._ptr, ruta.encode("utf-8")) == 0

    def imagen_a_teselacion(self, imagen: Tensor) -> Tensor:
        return Tensor(L.tc_imagen_a_teselacion(imagen._ptr, self._ptr))

    def teselacion_a_imagen(self, caracteristicas: Tensor, alto: int, ancho: int) -> Tensor:
        return Tensor(L.tc_teselacion_a_imagen(caracteristicas._ptr, self._ptr, alto, ancho))

    def __del__(self):
        try:
            if getattr(self, "_ptr", None):
                L.tc_liberar_teselacion(self._ptr)
        except Exception:
            pass


class KernelPenrose:
    def __init__(self, canales_entrada: int, canales_salida: int, nivel: int, escala: float = 1.0):
        self._ptr = L.tc_crear_kernel_penrose(canales_entrada, canales_salida, nivel, escala)
        if not self._ptr:
            raise RuntimeError("tc_crear_kernel_penrose falló")

    @property
    def pesos(self) -> Tensor:
        return Tensor(self._ptr.contents.pesos, dueno=False)

    @property
    def sesgo(self) -> Tensor:
        return Tensor(self._ptr.contents.sesgo, dueno=False)

    @property
    def teselacion_ptr(self):
        return self._ptr.contents.teselacion

    @property
    def num_tejas(self) -> int:
        return self._ptr.contents.teselacion.contents.num_tejas

    def conv_grafo(self, entrada: Tensor) -> Tensor:
        """Convolución en modo grafo: usa la conectividad de vecindad de Penrose."""
        salida = L.tc_conv_penrose_grafo(entrada._ptr, self.pesos._ptr, self.sesgo._ptr,
                                          self.teselacion_ptr)
        return Tensor(salida)

    def conv_imagen(self, entrada: Tensor, paso: int = 1, modo_relleno: int = 0) -> Tensor:
        """Convolución en modo imagen: interpolación bilineal sobre la teselación."""
        return Tensor(L.tc_conv_penrose(entrada._ptr, self._ptr, paso, modo_relleno))

    def __del__(self):
        try:
            if getattr(self, "_ptr", None):
                L.tc_liberar_kernel_penrose(self._ptr)
        except Exception:
            pass
