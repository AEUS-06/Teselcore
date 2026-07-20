"""
teselcore.modelo — Envoltura Python del formato binario .ax (guardar/cargar).
La serialización, el CRC32 y el layout binario viven enteramente en C;
este módulo solo administra el ciclo de vida del struct tc_modelo.
"""
from . import _lib as L
from .tensor import Tensor


class Modelo:
    def __init__(self, ptr=None, metadatos: str = ""):
        self._ptr = ptr if ptr is not None else L.tc_modelo_nuevo(metadatos.encode("utf-8"))
        if not self._ptr:
            raise RuntimeError("No se pudo crear/cargar el modelo")
        # Si se agrega un tensor prestado (dueño=False, p.ej. k.pesos de un
        # KernelPenrose), este modelo ya no puede liberar sus tensores al
        # morir — solo puede liberar su propia estructura, o habría un
        # double-free con el dueño original.
        self._tiene_tensores_prestados = False

    @property
    def num_tensores(self) -> int:
        return self._ptr.contents.num_tensores

    def agregar_tensor(self, nombre: str, tensor: Tensor):
        if not tensor._dueno:
            self._tiene_tensores_prestados = True
        L.tc_modelo_agregar_tensor(self._ptr, nombre.encode("utf-8"), tensor._ptr)

    def obtener_tensor(self, nombre: str) -> "Tensor | None":
        p = L.tc_modelo_obtener_tensor(self._ptr, nombre.encode("utf-8"))
        return Tensor(p, dueno=False) if p else None

    def nombres_tensores(self):
        n = self.num_tensores
        return [self._ptr.contents.tensores[i].nombre.decode("utf-8") for i in range(n)]

    def guardar(self, ruta: str) -> bool:
        return L.tc_guardar(self._ptr, ruta.encode("utf-8")) == 0

    @classmethod
    def cargar(cls, ruta: str) -> "Modelo":
        ptr = L.tc_cargar(ruta.encode("utf-8"))
        if not ptr:
            raise IOError(f"No se pudo cargar el modelo: {ruta}")
        m = cls(ptr=ptr)
        m._tiene_tensores_prestados = False  # tc_cargar crea tensores nuevos y propios
        return m

    def __del__(self):
        try:
            if getattr(self, "_ptr", None):
                if self._tiene_tensores_prestados:
                    L.tc_liberar_modelo_estructura(self._ptr)
                else:
                    L.tc_liberar_modelo(self._ptr)
        except Exception:
            pass
