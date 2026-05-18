"""
teselcore.py — Binding Python para TeselCore (via ctypes)
=========================================================
Expone la API de TeselCore como clases Python puras.
Si la librería .so no está disponible, activa el modo
"numpy puro" que reimplementa todas las operaciones en Python
para poder probar la exportabilidad del framework sin compilar C.

Uso:
    from teselcore import Tensor, red_cnn, red_penrose
"""

import ctypes
import os
import math
import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────
#  Intenta cargar la librería compartida
# ─────────────────────────────────────────────
_LIB: Optional[ctypes.CDLL] = None
_SEARCH_PATHS = [
    Path(__file__).parent / "libteselcore.so",
    Path(__file__).parent.parent / "teselcore-nucleo" / "libteselcore.so",
    Path("/usr/local/lib/libteselcore.so"),
]
for _p in _SEARCH_PATHS:
    if _p.exists():
        try:
            _LIB = ctypes.CDLL(str(_p))
            print(f"[TeselCore] Librería C cargada: {_p}")
            break
        except OSError:
            pass

if _LIB is None:
    print("[TeselCore] Librería .so no encontrada — modo NumPy puro activado.")

# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
TC_PHI = (1.0 + math.sqrt(5.0)) / 2.0
TC_PENROSE_KITE = 0
TC_PENROSE_DART = 1


# ═══════════════════════════════════════════════════════════════════════
#  MODO NUMPY PURO  (referencia portátil, idéntica semántica al núcleo C)
# ═══════════════════════════════════════════════════════════════════════

class Tensor:
    """Tensor diferenciable — idéntico en semántica al tc_tensor de C."""

    def __init__(self, data: np.ndarray, requiere_grad: bool = False):
        self.data: np.ndarray = np.array(data, dtype=np.float32)
        self.requiere_grad = requiere_grad
        self.grad: Optional[np.ndarray] = None
        self._backward_fn = None   # callable que acumula gradientes
        self._parents: List["Tensor"] = []

    # ── Constructores estáticos ──────────────────────────────────────
    @staticmethod
    def ceros(forma) -> "Tensor":
        return Tensor(np.zeros(forma, dtype=np.float32))

    @staticmethod
    def unos(forma) -> "Tensor":
        return Tensor(np.ones(forma, dtype=np.float32))

    @staticmethod
    def aleatorio_normal(forma, requiere_grad=False) -> "Tensor":
        t = Tensor(np.random.randn(*forma).astype(np.float32), requiere_grad)
        return t

    @staticmethod
    def desde_numpy(arr: np.ndarray, requiere_grad=False) -> "Tensor":
        return Tensor(arr.astype(np.float32), requiere_grad)

    @staticmethod
    def escalar(valor: float) -> "Tensor":
        return Tensor(np.array([valor], dtype=np.float32))

    # ── Propiedades ──────────────────────────────────────────────────
    @property
    def forma(self):
        return self.data.shape

    @property
    def total(self):
        return self.data.size

    def elemento(self) -> float:
        return float(self.data.flat[0])

    def numpy(self) -> np.ndarray:
        return self.data

    # ── Internos de autograd ─────────────────────────────────────────
    def _asegurar_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def cero_gradiente(self):
        if self.grad is not None:
            self.grad[:] = 0.0

    def retropropagar(self):
        """Retropropagación hacia atrás (recorre el grafo en orden topológico)."""
        self._asegurar_grad()
        self.grad[:] = 1.0

        # Orden topológico con DFS
        orden, visitados = [], set()
        def _visitar(t):
            if id(t) not in visitados:
                visitados.add(id(t))
                for p in t._parents:
                    _visitar(p)
                orden.append(t)
        _visitar(self)

        for nodo in reversed(orden):
            if nodo._backward_fn is not None and nodo.grad is not None:
                nodo._backward_fn(nodo.grad)

    # ── Operaciones elementales ──────────────────────────────────────
    def __add__(self, otro: "Tensor") -> "Tensor":
        sal = Tensor(self.data + otro.data,
                     self.requiere_grad or otro.requiere_grad)
        sal._parents = [self, otro]
        if sal.requiere_grad:
            a, b = self, otro
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g
                if b.requiere_grad:
                    b._asegurar_grad(); b.grad += g
            sal._backward_fn = bwd
        return sal

    def __sub__(self, otro: "Tensor") -> "Tensor":
        sal = Tensor(self.data - otro.data,
                     self.requiere_grad or otro.requiere_grad)
        sal._parents = [self, otro]
        if sal.requiere_grad:
            a, b = self, otro
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g
                if b.requiere_grad:
                    b._asegurar_grad(); b.grad -= g
            sal._backward_fn = bwd
        return sal

    def __mul__(self, otro: "Tensor") -> "Tensor":
        sal = Tensor(self.data * otro.data,
                     self.requiere_grad or otro.requiere_grad)
        sal._parents = [self, otro]
        if sal.requiere_grad:
            a, b = self, otro
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g * b.data
                if b.requiere_grad:
                    b._asegurar_grad(); b.grad += g * a.data
            sal._backward_fn = bwd
        return sal

    def __neg__(self) -> "Tensor":
        sal = Tensor(-self.data, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad -= g
            sal._backward_fn = bwd
        return sal

    def __truediv__(self, otro: "Tensor") -> "Tensor":
        sal = Tensor(self.data / (otro.data + 1e-8),
                     self.requiere_grad or otro.requiere_grad)
        sal._parents = [self, otro]
        if sal.requiere_grad:
            a, b = self, otro
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g / (b.data + 1e-8)
                if b.requiere_grad:
                    b._asegurar_grad()
                    b.grad -= g * a.data / (b.data ** 2 + 1e-8)
            sal._backward_fn = bwd
        return sal

    # ── Álgebra lineal ───────────────────────────────────────────────
    def matmul(self, otro: "Tensor") -> "Tensor":
        sal = Tensor(self.data @ otro.data,
                     self.requiere_grad or otro.requiere_grad)
        sal._parents = [self, otro]
        if sal.requiere_grad:
            a, b = self, otro
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g @ b.data.T
                if b.requiere_grad:
                    b._asegurar_grad(); b.grad += a.data.T @ g
            sal._backward_fn = bwd
        return sal

    def transponer(self) -> "Tensor":
        sal = Tensor(self.data.T, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g.T
            sal._backward_fn = bwd
        return sal

    def reformar(self, *nueva_forma) -> "Tensor":
        sal = Tensor(self.data.reshape(nueva_forma), self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a, forma_orig = self, self.data.shape
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g.reshape(forma_orig)
            sal._backward_fn = bwd
        return sal

    # ── Activaciones ─────────────────────────────────────────────────
    def relu(self) -> "Tensor":
        sal = Tensor(np.maximum(0, self.data), self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a, mask = self, self.data > 0
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g * mask
            sal._backward_fn = bwd
        return sal

    def sigmoide(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data.clip(-80, 80)))
        sal = Tensor(s, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g * s * (1 - s)
            sal._backward_fn = bwd
        return sal

    def tangente_hiperbolica(self) -> "Tensor":
        t_val = np.tanh(self.data)
        sal = Tensor(t_val, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g * (1 - t_val ** 2)
            sal._backward_fn = bwd
        return sal

    def softmax(self, eje=-1) -> "Tensor":
        x = self.data - self.data.max(axis=eje, keepdims=True)
        ex = np.exp(x)
        s = ex / ex.sum(axis=eje, keepdims=True)
        sal = Tensor(s, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad()
                    dot = (g * s).sum(axis=eje, keepdims=True)
                    a.grad += s * (g - dot)
            sal._backward_fn = bwd
        return sal

    def gelu(self) -> "Tensor":
        sqrt2pi = 0.7978845608
        x = self.data
        inner = sqrt2pi * (x + 0.044715 * x**3)
        tanh_val = np.tanh(inner)
        out = 0.5 * x * (1.0 + tanh_val)
        sal = Tensor(out, self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad()
                    sech2 = 1.0 - tanh_val ** 2
                    dgelu = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * sqrt2pi * (1 + 3*0.044715*x**2)
                    a.grad += g * dgelu
            sal._backward_fn = bwd
        return sal

    # ── Reducción ────────────────────────────────────────────────────
    def suma(self) -> "Tensor":
        sal = Tensor(np.array([self.data.sum()], dtype=np.float32), self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a, forma_orig = self, self.data.shape
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += np.full(forma_orig, g.flat[0])
            sal._backward_fn = bwd
        return sal

    def media(self) -> "Tensor":
        n = self.data.size
        sal = Tensor(np.array([self.data.mean()], dtype=np.float32), self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a, forma_orig = self, self.data.shape
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += np.full(forma_orig, g.flat[0] / n)
            sal._backward_fn = bwd
        return sal

    # ── Logaritmo / exponencial ──────────────────────────────────────
    def logaritmo(self) -> "Tensor":
        sal = Tensor(np.log(self.data + 1e-8), self.requiere_grad)
        sal._parents = [self]
        if sal.requiere_grad:
            a = self
            def bwd(g):
                if a.requiere_grad:
                    a._asegurar_grad(); a.grad += g / (a.data + 1e-8)
            sal._backward_fn = bwd
        return sal

    def __repr__(self):
        return f"Tensor(forma={self.data.shape}, req_grad={self.requiere_grad})"


# ── Funciones sueltas (API funcional como en C) ──────────────────────

def tc_conv2d(x: Tensor, peso: Tensor, sesgo: Optional[Tensor],
              paso: int = 1, relleno: int = 0) -> Tensor:
    """Conv2D con backprop via im2col/NumPy."""
    B, Cin, H, W = x.data.shape
    Cout, _, kH, kW = peso.data.shape
    assert peso.data.shape[1] == Cin

    if relleno > 0:
        xp = np.pad(x.data, ((0,0),(0,0),(relleno,relleno),(relleno,relleno)))
    else:
        xp = x.data

    H_out = (H + 2*relleno - kH) // paso + 1
    W_out = (W + 2*relleno - kW) // paso + 1

    # im2col
    cols = np.zeros((B, Cin, kH, kW, H_out, W_out), dtype=np.float32)
    for h in range(kH):
        for w in range(kW):
            cols[:, :, h, w, :, :] = xp[:, :, h:h+H_out*paso:paso, w:w+W_out*paso:paso]

    cols_flat = cols.reshape(B, Cin*kH*kW, H_out*W_out)
    k_flat    = peso.data.reshape(Cout, Cin*kH*kW)

    # (B, Cout, H_out*W_out)
    out = np.tensordot(k_flat, cols_flat, axes=([1],[1])).transpose(1,0,2)
    out = out.reshape(B, Cout, H_out, W_out)

    if sesgo is not None:
        out += sesgo.data[None, :, None, None]

    sal = Tensor(out, x.requiere_grad or peso.requiere_grad)
    sal._parents = [x, peso] + ([sesgo] if sesgo else [])

    if sal.requiere_grad:
        def bwd(g):  # g: (B, Cout, H_out, W_out)
            if peso.requiere_grad:
                peso._asegurar_grad()
                g_flat = g.reshape(B, Cout, -1)
                dw = np.tensordot(g_flat, cols_flat, axes=([0,2],[0,2]))
                peso.grad += dw.reshape(peso.data.shape)
            if sesgo is not None and sesgo.requiere_grad:
                sesgo._asegurar_grad()
                sesgo.grad += g.sum(axis=(0,2,3))
            if x.requiere_grad:
                x._asegurar_grad()
                g_flat = g.reshape(B, Cout, -1)
                dcols = np.tensordot(k_flat, g_flat, axes=([0],[1])).transpose(1,0,2)
                dcols = dcols.reshape(B, Cin, kH, kW, H_out, W_out)
                dxp = np.zeros_like(xp)
                for h in range(kH):
                    for w in range(kW):
                        dxp[:,:,h:h+H_out*paso:paso,w:w+W_out*paso:paso] += dcols[:,:,h,w,:,:]
                if relleno > 0:
                    x.grad += dxp[:,:,relleno:-relleno,relleno:-relleno]
                else:
                    x.grad += dxp
        sal._backward_fn = bwd

    return sal


def tc_agrupacion_max(x: Tensor, kernel: int, paso: int) -> Tensor:
    B, C, H, W = x.data.shape
    H_out = (H - kernel) // paso + 1
    W_out = (W - kernel) // paso + 1
    out   = np.zeros((B, C, H_out, W_out), dtype=np.float32)
    mask  = np.zeros_like(x.data, dtype=bool)

    for h in range(H_out):
        for w in range(W_out):
            region = x.data[:, :, h*paso:h*paso+kernel, w*paso:w*paso+kernel]
            flat   = region.reshape(B, C, -1)
            idx    = flat.argmax(axis=2)
            out[:, :, h, w] = flat[np.arange(B)[:,None], np.arange(C)[None,:], idx]
            local = np.zeros_like(flat, dtype=bool)
            local[np.arange(B)[:,None], np.arange(C)[None,:], idx] = True
            mask[:, :, h*paso:h*paso+kernel, w*paso:w*paso+kernel] |= local.reshape(B,C,kernel,kernel)

    sal = Tensor(out, x.requiere_grad)
    sal._parents = [x]
    if sal.requiere_grad:
        def bwd(g):
            if x.requiere_grad:
                x._asegurar_grad()
                dx = np.zeros_like(x.data)
                for h in range(H_out):
                    for w in range(W_out):
                        region_mask = mask[:,:,h*paso:h*paso+kernel,w*paso:w*paso+kernel]
                        dx[:,:,h*paso:h*paso+kernel,w*paso:w*paso+kernel] += \
                            region_mask * g[:,:,h:h+1,w:w+1]
                x.grad += dx
        sal._backward_fn = bwd
    return sal


def tc_normalizacion_lote(x: Tensor, gamma: Tensor, beta: Tensor,
                           eps: float = 1e-5, entrenando: bool = True) -> Tensor:
    if entrenando:
        mu  = x.data.mean(axis=(0,2,3), keepdims=True)
        var = x.data.var(axis=(0,2,3), keepdims=True)
        xhat = (x.data - mu) / np.sqrt(var + eps)
    else:
        xhat = x.data  # simplificado

    out = gamma.data[None,:,None,None] * xhat + beta.data[None,:,None,None]
    sal = Tensor(out, x.requiere_grad or gamma.requiere_grad or beta.requiere_grad)
    sal._parents = [x, gamma, beta]

    if sal.requiere_grad:
        def bwd(g):
            N, C, H, W = x.data.shape
            m = N * H * W
            if gamma.requiere_grad:
                gamma._asegurar_grad()
                gamma.grad += (g * xhat).sum(axis=(0,2,3))
            if beta.requiere_grad:
                beta._asegurar_grad()
                beta.grad += g.sum(axis=(0,2,3))
            if x.requiere_grad:
                x._asegurar_grad()
                gx = gamma.data[None,:,None,None] * g
                std_inv = 1.0 / np.sqrt(var + eps)
                dx = (1.0/m) * std_inv * (m*gx - gx.sum(axis=(0,2,3),keepdims=True)
                     - xhat * (gx*xhat).sum(axis=(0,2,3),keepdims=True))
                x.grad += dx
        sal._backward_fn = bwd
    return sal


def tc_abandono(x: Tensor, p: float, entrenando: bool = True) -> Tensor:
    if not entrenando or p <= 0.0:
        return Tensor(x.data.copy(), x.requiere_grad)
    mask  = (np.random.rand(*x.data.shape) > p).astype(np.float32) / (1.0 - p)
    out   = x.data * mask
    sal   = Tensor(out, x.requiere_grad)
    sal._parents = [x]
    if sal.requiere_grad:
        def bwd(g):
            if x.requiere_grad:
                x._asegurar_grad(); x.grad += g * mask
        sal._backward_fn = bwd
    return sal


# ── Pérdidas ─────────────────────────────────────────────────────────

def tc_entropia_cruzada(logits: Tensor, etiquetas: np.ndarray) -> Tensor:
    """Cross-entropy con backprop directo en logits (más estable)."""
    B, C = logits.data.shape
    x = logits.data - logits.data.max(axis=1, keepdims=True)
    ex = np.exp(x); sm = ex / ex.sum(axis=1, keepdims=True)
    perdida = -np.log(sm[np.arange(B), etiquetas] + 1e-8).mean()

    sal = Tensor(np.array([perdida], dtype=np.float32), logits.requiere_grad)
    sal._parents = [logits]

    if sal.requiere_grad:
        def bwd(g):
            if logits.requiere_grad:
                logits._asegurar_grad()
                ds = sm.copy()
                ds[np.arange(B), etiquetas] -= 1.0
                logits.grad += (g.flat[0] / B) * ds
        sal._backward_fn = bwd
    return sal


def tc_error_cuadratico_medio(pred: Tensor, objetivo: Tensor) -> Tensor:
    diff = pred.data - objetivo.data
    perdida = (diff ** 2).mean()
    sal = Tensor(np.array([perdida], dtype=np.float32), pred.requiere_grad)
    sal._parents = [pred, objetivo]
    if sal.requiere_grad:
        def bwd(g):
            if pred.requiere_grad:
                pred._asegurar_grad()
                pred.grad += (2.0 * g.flat[0] / pred.data.size) * diff
        sal._backward_fn = bwd
    return sal


# ── Optimizadores ─────────────────────────────────────────────────────

class SGD:
    def __init__(self, params: List[Tensor], lr=0.01, momento=0.9):
        self.params  = params
        self.lr      = lr
        self.momento = momento
        self.v       = [np.zeros_like(p.data) for p in params]

    def paso(self):
        for i, p in enumerate(self.params):
            if not p.requiere_grad or p.grad is None:
                continue
            self.v[i] = self.momento * self.v[i] + p.grad
            p.data   -= self.lr * self.v[i]

    def cero_gradientes(self):
        for p in self.params:
            p.cero_gradiente()


class Adam:
    def __init__(self, params: List[Tensor], lr=1e-3,
                 beta1=0.9, beta2=0.999, eps=1e-8, decaimiento=0.0):
        self.params    = params
        self.lr        = lr
        self.beta1     = beta1
        self.beta2     = beta2
        self.eps       = eps
        self.decaimiento = decaimiento
        self.m1 = [np.zeros_like(p.data) for p in params]
        self.m2 = [np.zeros_like(p.data) for p in params]
        self.t  = 0

    def paso(self):
        self.t += 1
        corr1 = 1.0 - self.beta1 ** self.t
        corr2 = 1.0 - self.beta2 ** self.t
        lr_t  = self.lr * math.sqrt(corr2) / corr1

        for i, p in enumerate(self.params):
            if not p.requiere_grad or p.grad is None:
                continue
            g = p.grad
            self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * g
            self.m2[i] = self.beta2 * self.m2[i] + (1 - self.beta2) * g * g
            upd = lr_t * self.m1[i] / (np.sqrt(self.m2[i]) + self.eps)
            if self.decaimiento > 0:
                upd += self.lr * self.decaimiento * p.data
            p.data -= upd

    def cero_gradientes(self):
        for p in self.params:
            p.cero_gradiente()


# ── Guardado / carga formato .ax ─────────────────────────────────────

MAGIC = b"AXON"

def _crc32(data: bytes) -> int:
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF

def tc_guardar(tensores_nombrados: dict, ruta: str,
               metadatos: str = "", version=(1, 0)):
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<HH", version[0], version[1])
    buf += struct.pack("<I", 0)  # banderas
    meta_bytes = metadatos.encode("utf-8")
    buf += struct.pack("<I", len(meta_bytes))
    buf += meta_bytes
    buf += struct.pack("<I", len(tensores_nombrados))

    for nombre, tensor in tensores_nombrados.items():
        nb = nombre.encode("utf-8")
        buf += struct.pack("<H", len(nb)) + nb
        arr = tensor.data if isinstance(tensor, Tensor) else tensor
        arr = arr.astype(np.float32)
        buf += struct.pack("<BB", 0, arr.ndim)  # tipo TC_FLOAT32=0
        for d in arr.shape:
            buf += struct.pack("<I", d)
        raw = arr.tobytes()
        buf += struct.pack("<Q", len(raw))
        buf += raw

    crc = _crc32(bytes(buf))
    buf += struct.pack("<I", crc)

    with open(ruta, "wb") as f:
        f.write(buf)
    print(f"tc_guardar: '{ruta}' guardado ({len(buf)/1024:.2f} KB)")


def tc_cargar(ruta: str) -> dict:
    with open(ruta, "rb") as f:
        contenido = f.read()

    crc_guardado = struct.unpack("<I", contenido[-4:])[0]
    crc_calc     = _crc32(contenido[:-4])
    if crc_guardado != crc_calc:
        raise ValueError("tc_cargar: CRC inválido — archivo corrupto")

    pos = 0
    assert contenido[pos:pos+4] == MAGIC, "No es un archivo .ax válido"
    pos += 4

    v_may, v_men = struct.unpack_from("<HH", contenido, pos); pos += 4
    banderas,     = struct.unpack_from("<I",  contenido, pos); pos += 4
    meta_len,     = struct.unpack_from("<I",  contenido, pos); pos += 4
    metadatos = contenido[pos:pos+meta_len].decode("utf-8"); pos += meta_len
    num,       = struct.unpack_from("<I",  contenido, pos); pos += 4

    resultado = {}
    for _ in range(num):
        nb_len, = struct.unpack_from("<H", contenido, pos); pos += 2
        nombre  = contenido[pos:pos+nb_len].decode("utf-8"); pos += nb_len
        tipo, ndim = struct.unpack_from("<BB", contenido, pos); pos += 2
        forma = tuple(struct.unpack_from("<I", contenido, pos + i*4)[0] for i in range(ndim))
        pos += ndim * 4
        raw_len, = struct.unpack_from("<Q", contenido, pos); pos += 8
        arr = np.frombuffer(contenido[pos:pos+raw_len], dtype=np.float32).copy()
        pos += raw_len
        arr = arr.reshape(forma)
        resultado[nombre] = Tensor.desde_numpy(arr)

    print(f"tc_cargar: '{ruta}' cargado ({num} tensores)")
    return resultado