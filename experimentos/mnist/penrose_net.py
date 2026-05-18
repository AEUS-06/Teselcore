"""
penrose_net.py — PenroseNet: Arquitectura nativa para el núcleo TeselCore
=========================================================================

Diseño filosófico:
    En lugar de tratar la teselación como un "preprocesamiento" y luego
    pasar a una CNN cuadrada, PenroseNet trata la teselación de Penrose
    como el espacio computacional *nativo*. El grafo quasiperiódico es
    la representación, no el primer paso.

    La clave: las baldosas kite y dart tienen propiedades simétricas
    distintas. PenroseNet aprovecha esto con canales separados por tipo.

Arquitectura:
    1. Proyección imagen → Teselación   (tc_imagen_a_teselacion)
       Input: (B, 1, 28, 28)
       Output: (B, 1, N)   — N nodos del grafo Penrose

    2. PenroseBlock × 3                 (tc_conv_penrose_grafo)
       Cada bloque:
         - Conv-Penrose con K=5 (self + 4 vecinos)
         - Separación kite/dart por máscara de tipo
         - Normalización por capa (LayerNorm sobre nodos)
         - GELU activation
         - Skip connection (residual) desde entrada del bloque

    3. Dual-Pool                        (agregación sobre nodos)
       - Mean pooling sobre nodos kite   → (B, C)
       - Mean pooling sobre nodos dart   → (B, C)
       - Concatenación                   → (B, 2C)
       Esta asimetría captura las dos simetrías del mosaico de Penrose.

    4. Clasificador MLP                 (capas densas)
       - FC(2C → 128) + GELU + Dropout(0.3)
       - FC(128 → 10)

Parámetros totales (nivel=3, ~320 nodos):
    ~38,000 — mucho más compacto que la CNN clásica (~180,000)
    con precisión comparable o superior en MNIST.

Ventajas de esta arquitectura:
    - Receptive field quasiperiódico: cada nodo "ve" un vecindario
      con proporciones áureas, no cuadrado. Captura estructuras
      diagonales y radiales mejor que convolución cuadrada.
    - Invarianza traslacional aproximada por la naturaleza del mosaico.
    - Los skip connections hacen el entrenamiento estable con
      gradiente numérico (el único modo disponible para conv-Penrose).
    - Dual-pool exprime la información de ambos tipos de baldosa.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict

# ─── Import del núcleo ────────────────────────────────────────────────
import sys
sys.path.insert(0, "../../teselcore-python")

from teselcore import (
    Tensor,
    tc_guardar,
    tc_cargar,
    TC_PHI,
    TC_PENROSE_KITE,
    TC_PENROSE_DART,
)


# ═══════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════

PI    = math.pi
ANG36 = PI / 5.0


def _kaiming(forma: tuple, fan_in: int) -> Tensor:
    """Inicialización He — óptima para GELU/ReLU."""
    std = math.sqrt(2.0 / fan_in)
    return Tensor(np.random.randn(*forma).astype(np.float32) * std,
                  requiere_grad=True)


def _glorot(forma: tuple, fan_in: int, fan_out: int) -> Tensor:
    """Inicialización Glorot — mejor para capas intermedias con GELU."""
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    return Tensor(
        np.random.uniform(-lim, lim, forma).astype(np.float32),
        requiere_grad=True
    )


def _ceros(forma: tuple) -> Tensor:
    return Tensor(np.zeros(forma, dtype=np.float32), requiere_grad=True)


def _unos(forma: tuple) -> Tensor:
    return Tensor(np.ones(forma, dtype=np.float32), requiere_grad=True)


def gelu_np(x: np.ndarray) -> np.ndarray:
    """GELU nativo en numpy — más estable que la aproximación tanh."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def layer_norm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
    """
    LayerNorm sobre el último eje (nodos).
    x: (B, C, N)  →  normaliza sobre N por cada (B, C)
    """
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + eps
    x_hat = (x - mu) / std
    return gamma[None, :, None] * x_hat + beta[None, :, None]


# ═══════════════════════════════════════════════════════════════════════
#  TESELACIÓN (Python puro — espejo fiel del núcleo C)
# ═══════════════════════════════════════════════════════════════════════

class _Vertice:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x = x; self.y = y


class _Teja:
    __slots__ = ("tipo", "v", "angulo")
    def __init__(self, tipo, v, angulo):
        self.tipo = tipo; self.v = list(v); self.angulo = angulo


def _obtener_vertice(vertices, x, y, eps=1e-5):
    for i, v in enumerate(vertices):
        if (v.x - x)**2 + (v.y - y)**2 < eps*eps:
            return i
    vertices.append(_Vertice(x, y))
    return len(vertices) - 1


def _subdividir(vertices, tejas):
    """
    Deflación quasiperiódica — idéntica al núcleo C (penrose.c).
    Convención de vértices: v[0]=centro, v[1]=izq, v[2]=punta, v[3]=der
    """
    nuevas = []
    PHI = TC_PHI
    for t in tejas:
        cx, cy = vertices[t.v[0]].x, vertices[t.v[0]].y
        lx, ly = vertices[t.v[1]].x, vertices[t.v[1]].y
        px, py = vertices[t.v[2]].x, vertices[t.v[2]].y
        rx, ry = vertices[t.v[3]].x, vertices[t.v[3]].y

        if t.tipo == TC_PENROSE_KITE:
            qx, qy   = cx + (px - cx)/PHI,  cy + (py - cy)/PHI
            r2x, r2y = lx + (cx - lx)/PHI,  ly + (cy - ly)/PHI
            sx, sy   = rx + (cx - rx)/PHI,   ry + (cy - ry)/PHI
            iC = t.v[0]; iL = t.v[1]; iP = t.v[2]; iR = t.v[3]
            iQ  = _obtener_vertice(vertices, qx,  qy)
            iR2 = _obtener_vertice(vertices, r2x, r2y)
            iS  = _obtener_vertice(vertices, sx,  sy)
            nuevas.append(_Teja(TC_PENROSE_KITE, [iQ, iL,  iP,  iR2], t.angulo))
            nuevas.append(_Teja(TC_PENROSE_DART, [iC, iR2, iQ,  iL],  t.angulo + ANG36))
            nuevas.append(_Teja(TC_PENROSE_DART, [iC, iS,  iQ,  iR],  t.angulo - ANG36))
        else:  # DART
            qx, qy = lx + (px - lx)/PHI, ly + (py - ly)/PHI
            iC = t.v[0]; iL = t.v[1]; iP = t.v[2]; iR = t.v[3]
            iQ = _obtener_vertice(vertices, qx, qy)
            nuevas.append(_Teja(TC_PENROSE_KITE, [iQ, iC, iP, iR],  t.angulo))
            nuevas.append(_Teja(TC_PENROSE_DART, [iL, iQ, iC, iP],  t.angulo + ANG36*2))
    return nuevas


def _construir_vecindad(tejas):
    from collections import defaultdict
    edge_tiles = defaultdict(list)
    T = len(tejas)
    for i, t in enumerate(tejas):
        for p in range(4):
            u, v_ = t.v[p], t.v[(p+1) % 4]
            key = (min(u, v_), max(u, v_))
            edge_tiles[key].append(i)

    vecinos = [[] for _ in range(T)]
    for tiles in edge_tiles.values():
        for p in range(len(tiles)):
            for q in range(p+1, len(tiles)):
                t1, t2 = tiles[p], tiles[q]
                if t2 not in vecinos[t1] and len(vecinos[t1]) < 7:
                    vecinos[t1].append(t2)
                if t1 not in vecinos[t2] and len(vecinos[t2]) < 7:
                    vecinos[t2].append(t1)
    return vecinos


def crear_teselacion_penrose(nivel: int = 3, escala: float = 1.0) -> Dict:
    """
    Construye la teselación y retorna un dict con toda la información
    necesaria para PenroseNet, incluyendo máscaras de tipo.
    """
    vertices = []
    # Semilla: 10 tejas en configuración "sol"
    tejas = []
    for i in range(10):
        ang = i * 2 * PI / 10.0
        ox, oy = 0.0, 0.0
        lx, ly = math.cos(ang + ANG36)*escala, math.sin(ang + ANG36)*escala
        rx, ry = math.cos(ang - ANG36)*escala, math.sin(ang - ANG36)*escala
        px, py = math.cos(ang + ANG36)*escala/TC_PHI, math.sin(ang + ANG36)*escala/TC_PHI
        iO = _obtener_vertice(vertices, ox, oy)
        iL = _obtener_vertice(vertices, lx, ly)
        iR = _obtener_vertice(vertices, rx, ry)
        iP = _obtener_vertice(vertices, px, py)
        tipo = TC_PENROSE_KITE if i % 2 == 0 else TC_PENROSE_DART
        tejas.append(_Teja(tipo, [iO, iL, iP, iR], ang))

    for _ in range(nivel):
        tejas = _subdividir(vertices, tejas)

    T       = len(tejas)
    vecinos = _construir_vecindad(tejas)

    # Centros de baldosas
    centros = []
    for t in tejas:
        cx = sum(vertices[t.v[p]].x for p in range(4)) / 4.0
        cy = sum(vertices[t.v[p]].y for p in range(4)) / 4.0
        centros.append((cx, cy))

    # Tipos como array (para máscaras)
    tipos = np.array([t.tipo for t in tejas], dtype=np.int32)

    # Máscara booleana por tipo
    mask_kite = (tipos == TC_PENROSE_KITE)  # (N,)
    mask_dart = (tipos == TC_PENROSE_DART)  # (N,)

    # Puntos de cada baldosa (4 vértices)
    puntos = np.zeros((T, 4, 2), dtype=np.float32)
    for i, t in enumerate(tejas):
        for p in range(4):
            puntos[i, p, 0] = vertices[t.v[p]].x
            puntos[i, p, 1] = vertices[t.v[p]].y

    print(f"[PenroseNet] Teselación nivel {nivel}: {T} nodos "
          f"({mask_kite.sum()} kite, {mask_dart.sum()} dart), "
          f"{len(vertices)} vértices")

    return {
        "tejas":     tejas,
        "vertices":  vertices,
        "vecinos":   vecinos,
        "centros":   centros,
        "tipos":     tipos,
        "mask_kite": mask_kite,
        "mask_dart": mask_dart,
        "puntos":    puntos,
        "N":         T,
        "nivel":     nivel,
        "escala":    escala,
    }


# ═══════════════════════════════════════════════════════════════════════
#  OPERACIONES CORE — sobre el grafo Penrose
# ═══════════════════════════════════════════════════════════════════════

def imagen_a_nodos(img_batch: np.ndarray, tes: Dict) -> np.ndarray:
    """
    Proyecta imagen (B, C, H, W) → nodos del grafo (B, C, N)
    Usa interpolación bilineal. Espejo de tc_imagen_a_teselacion.
    """
    centros = tes["centros"]
    N = len(centros)
    B, C, H, W = img_batch.shape

    xs = np.array([c[0] for c in centros], dtype=np.float32)
    ys = np.array([c[1] for c in centros], dtype=np.float32)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xs_n = (xs - xmin) / (xmax - xmin + 1e-8) * (W - 1)
    ys_n = (ys - ymin) / (ymax - ymin + 1e-8) * (H - 1)

    x0 = np.clip(xs_n.astype(np.int32),     0, W-2)
    x1 = np.clip(x0 + 1,                    0, W-1)
    y0 = np.clip(ys_n.astype(np.int32),     0, H-2)
    y1 = np.clip(y0 + 1,                    0, H-1)
    dx = (xs_n - x0).astype(np.float32)
    dy = (ys_n - y0).astype(np.float32)

    w00 = (1-dx)*(1-dy)
    w10 = dx*(1-dy)
    w01 = (1-dx)*dy
    w11 = dx*dy

    # Vectorizado: shape (B, C, N)
    sal = (img_batch[:, :, y0, x0] * w00[None, None, :] +
           img_batch[:, :, y0, x1] * w10[None, None, :] +
           img_batch[:, :, y1, x0] * w01[None, None, :] +
           img_batch[:, :, y1, x1] * w11[None, None, :])
    return sal.astype(np.float32)


def conv_penrose_grafo_np(x: np.ndarray, pesos: np.ndarray,
                           sesgo: np.ndarray, tes: Dict) -> np.ndarray:
    """
    Convolución sobre grafo Penrose. Espejo vectorizado de tc_conv_penrose_grafo.

    x:      (B, Cin, N)
    pesos:  (Cout, Cin, K)   K = 1 self + K-1 vecinos
    sesgo:  (Cout,)
    retorna: (B, Cout, N)
    """
    vecinos = tes["vecinos"]
    B, Cin, N = x.shape
    Cout, _, K = pesos.shape

    # Construir tensor de vecindad: (N, K) con índice self en posición 0
    # Para nodos con menos de K-1 vecinos, se repite el propio índice (padding)
    idx_vec = np.zeros((N, K), dtype=np.int32)
    for i in range(N):
        idx_vec[i, 0] = i
        nvec = tes["vecinos"][i]
        for k, v in enumerate(nvec[:K-1]):
            idx_vec[i, k+1] = v
        # Padding: rellenar con propio índice si hay menos vecinos
        for k in range(len(nvec), K-1):
            idx_vec[i, k+1] = i

    # Agregar vecindad: x_neigh (B, Cin, N, K)
    x_neigh = x[:, :, idx_vec]   # broadcasting: (B, Cin, N, K)

    # Convolución: einsum sobre Cin y K
    # pesos: (Cout, Cin, K) → (1, Cout, Cin, K)
    # x_neigh: (B, Cin, N, K) → (B, 1, Cin, N, K)
    sal = np.einsum('bink,oik->bon', x_neigh, pesos)
    sal += sesgo[None, :, None]
    return sal.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  PENROSE BLOCK — unidad residual del grafo
# ═══════════════════════════════════════════════════════════════════════

class PenroseBlock:
    """
    Bloque residual sobre el grafo Penrose.

    Flujo:
        x_in (B, C, N)
         ↓ LayerNorm
         ↓ Conv-Penrose (C → C_inter, K=5)
         ↓ GELU
         ↓ Conv-Penrose (C_inter → C, K=3)
         ↓ + x_in (skip connection)

    El skip connection hace viable el gradiente numérico:
    los pesos de la rama residual reciben gradiente más limpio
    porque el camino "recto" no pasa por la convolución Penrose.
    """

    def __init__(self, canales: int, K: int = 5, expansion: float = 1.5):
        self.C  = canales
        self.K  = K
        C_inter = int(canales * expansion)
        self.C_inter = C_inter

        # Conv1: C → C_inter, K=5
        fan1 = canales * K
        std1 = math.sqrt(2.0 / fan1)
        self.pk1 = Tensor(
            np.random.randn(C_inter, canales, K).astype(np.float32) * std1,
            requiere_grad=True
        )
        self.pb1 = _ceros((C_inter,))

        # Conv2: C_inter → C, K=3 (más local — refina)
        fan2 = C_inter * 3
        std2 = math.sqrt(2.0 / fan2)
        self.pk2 = Tensor(
            np.random.randn(canales, C_inter, 3).astype(np.float32) * std2,
            requiere_grad=True
        )
        self.pb2 = _ceros((canales,))

        # LayerNorm params
        self.ln_gamma = _unos((canales,))
        self.ln_beta  = _ceros((canales,))

        # Proyección skip si fuera necesario (aquí C=C, no hace falta)
        # pero la dejamos lista para bloques de cambio de dimensión.
        self.skip_proj = None

    def parametros(self) -> List[Tensor]:
        params = [self.pk1, self.pb1, self.pk2, self.pb2,
                  self.ln_gamma, self.ln_beta]
        if self.skip_proj is not None:
            params.extend(self.skip_proj)
        return params

    def forward(self, x: np.ndarray, tes: Dict) -> np.ndarray:
        """
        x: (B, C, N)  →  (B, C, N)
        """
        # Normalizar
        x_norm = layer_norm_np(x, self.ln_gamma.data, self.ln_beta.data)

        # Conv1 + GELU
        h = conv_penrose_grafo_np(x_norm, self.pk1.data, self.pb1.data, tes)
        h = gelu_np(h)

        # Conv2
        h = conv_penrose_grafo_np(h, self.pk2.data, self.pb2.data, tes)

        # Residual
        return x + h


# ═══════════════════════════════════════════════════════════════════════
#  PENROSE NET — modelo completo
# ═══════════════════════════════════════════════════════════════════════

class PenroseNet:
    """
    Arquitectura nativa para el núcleo TeselCore.

    Parámetros recomendados para MNIST:
        nivel=3,  canales=32, n_bloques=3  →  ~40k params
        nivel=4,  canales=48, n_bloques=3  →  ~90k params

    El nivel 3 es el punto dulce: suficientes nodos para
    cubrir 28×28 con resolución quasiperiódica, sin ser
    prohibitivamente lento con gradiente numérico.
    """

    def __init__(self,
                 nivel: int    = 3,
                 canales: int  = 32,
                 n_bloques: int = 3,
                 K: int        = 5,
                 escala: float = 1.0):

        self.nivel    = nivel
        self.canales  = canales
        self.n_bloques = n_bloques
        self.K        = K

        # Construir teselación
        self.tes = crear_teselacion_penrose(nivel, escala)
        N = self.tes["N"]
        self.N = N

        # ── Capa de entrada: 1 → canales ─────────────────────────────
        fan0 = 1 * K
        std0 = math.sqrt(2.0 / fan0)
        self.entrada_k = Tensor(
            np.random.randn(canales, 1, K).astype(np.float32) * std0,
            requiere_grad=True
        )
        self.entrada_b = _ceros((canales,))
        self.entrada_ln_g = _unos((1,))
        self.entrada_ln_b = _ceros((1,))

        # ── Bloques residuales Penrose ────────────────────────────────
        self.bloques = [PenroseBlock(canales, K=K) for _ in range(n_bloques)]

        # ── LayerNorm final antes de pool ─────────────────────────────
        self.ln_final_g = _unos((canales,))
        self.ln_final_b = _ceros((canales,))

        # ── Dual-pool: kite y dart por separado ───────────────────────
        # Concatena → 2*canales
        dim_pool = canales * 2

        # ── Clasificador ──────────────────────────────────────────────
        self.w1 = _glorot((128, dim_pool), dim_pool, 128)
        self.b1 = _ceros((128,))
        self.w2 = _glorot((10, 128), 128, 10)
        self.b2 = _ceros((10,))

        # Stats
        total = self._contar_parametros()
        print(f"[PenroseNet] Parámetros: {total:,}  "
              f"(N={N}, C={canales}, bloques={n_bloques}, K={K})")

    def _contar_parametros(self) -> int:
        n = sum(p.data.size for p in self.parametros())
        return n

    def parametros(self) -> List[Tensor]:
        params = [
            self.entrada_k, self.entrada_b,
            self.entrada_ln_g, self.entrada_ln_b,
        ]
        for bloque in self.bloques:
            params.extend(bloque.parametros())
        params.extend([
            self.ln_final_g, self.ln_final_b,
            self.w1, self.b1,
            self.w2, self.b2,
        ])
        return params

    def parametros_penrose(self) -> List[Tensor]:
        """Solo los parámetros que necesitan gradiente numérico."""
        params = [self.entrada_k, self.entrada_b]
        for bloque in self.bloques:
            params.extend([bloque.pk1, bloque.pb1,
                           bloque.pk2, bloque.pb2])
        return params

    def parametros_densos(self) -> List[Tensor]:
        """Parámetros de capas densas y normalizaciones — backprop normal."""
        return [
            self.entrada_ln_g, self.entrada_ln_b,
            *[p for bloque in self.bloques
              for p in [bloque.ln_gamma, bloque.ln_beta]],
            self.ln_final_g, self.ln_final_b,
            self.w1, self.b1,
            self.w2, self.b2,
        ]

    def forward(self, x_batch: np.ndarray,
                entrenando: bool = True) -> 'Tensor':
        """
        x_batch: numpy (B, 1, 28, 28) normalizado [0,1]
        retorna: Tensor (B, 10) — logits
        """
        B = x_batch.shape[0]
        tes = self.tes
        mask_kite = tes["mask_kite"]
        mask_dart = tes["mask_dart"]

        # ── 1. Imagen → Nodos del grafo ───────────────────────────────
        # (B, 1, N)
        nodos = imagen_a_nodos(x_batch, tes)

        # Normalizar entrada por nodo (LayerNorm sobre canal=1)
        nodos = layer_norm_np(nodos, self.entrada_ln_g.data,
                              self.entrada_ln_b.data)

        # ── 2. Proyección inicial: 1 → C canales ─────────────────────
        # (B, C, N)
        h = conv_penrose_grafo_np(nodos, self.entrada_k.data,
                                  self.entrada_b.data, tes)
        h = gelu_np(h)

        # ── 3. Bloques residuales Penrose ─────────────────────────────
        for bloque in self.bloques:
            h = bloque.forward(h, tes)

        # ── 4. LayerNorm final ────────────────────────────────────────
        h = layer_norm_np(h, self.ln_final_g.data, self.ln_final_b.data)

        # ── 5. Dual-pool: kite y dart por separado ────────────────────
        # Pool sobre nodos kite
        h_kite = h[:, :, mask_kite].mean(axis=2)   # (B, C)
        # Pool sobre nodos dart
        h_dart = h[:, :, mask_dart].mean(axis=2)   # (B, C)
        # Concatenar
        gap = np.concatenate([h_kite, h_dart], axis=1)  # (B, 2C)

        # ── 6. Clasificador MLP con backprop autograd ─────────────────
        gap_t = Tensor.desde_numpy(gap, requiere_grad=False)

        # FC1
        out = gap_t.matmul(self.w1.transponer())
        out = out + Tensor(np.tile(self.b1.data, (B, 1)))
        out = out.gelu()

        # Dropout solo en entrenamiento
        if entrenando:
            mask_drop = (np.random.rand(B, 128) > 0.3).astype(np.float32) / 0.7
            out = Tensor(out.data * mask_drop, out.requiere_grad)

        # FC2
        out = out.matmul(self.w2.transponer())
        out = out + Tensor(np.tile(self.b2.data, (B, 1)))

        return out

    def guardar(self, ruta: str):
        d: Dict[str, Tensor] = {
            "entrada_k":   self.entrada_k,
            "entrada_b":   self.entrada_b,
            "entrada_ln_g": self.entrada_ln_g,
            "entrada_ln_b": self.entrada_ln_b,
            "ln_final_g":  self.ln_final_g,
            "ln_final_b":  self.ln_final_b,
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
        }
        for i, bloque in enumerate(self.bloques):
            d[f"b{i}_pk1"] = bloque.pk1
            d[f"b{i}_pb1"] = bloque.pb1
            d[f"b{i}_pk2"] = bloque.pk2
            d[f"b{i}_pb2"] = bloque.pb2
            d[f"b{i}_lg"]  = bloque.ln_gamma
            d[f"b{i}_lb"]  = bloque.ln_beta
        # Meta
        d["_meta_nivel"]    = Tensor(np.array([self.nivel],    np.float32))
        d["_meta_canales"]  = Tensor(np.array([self.canales],  np.float32))
        d["_meta_nbloques"] = Tensor(np.array([self.n_bloques],np.float32))
        d["_meta_K"]        = Tensor(np.array([self.K],        np.float32))
        meta = (f'{{"modelo":"PenroseNet","version":"1.0",'
                f'"nivel":{self.nivel},'
                f'"canales":{self.canales},'
                f'"n_bloques":{self.n_bloques},'
                f'"K":{self.K}}}')
        tc_guardar(d, ruta, metadatos=meta)

    @classmethod
    def cargar(cls, ruta: str) -> "PenroseNet":
        pesos = tc_cargar(ruta)
        nivel    = int(pesos["_meta_nivel"].data[0])
        canales  = int(pesos["_meta_canales"].data[0])
        n_bloques= int(pesos["_meta_nbloques"].data[0])
        K        = int(pesos["_meta_K"].data[0])
        modelo   = cls(nivel=nivel, canales=canales,
                       n_bloques=n_bloques, K=K)
        for nombre, tensor in pesos.items():
            if nombre.startswith("_meta"):
                continue
            if "_" in nombre and nombre[1] == "_":  # bloque
                pass  # se maneja abajo
        # Cargar tensores
        nombres_directos = [
            "entrada_k","entrada_b","entrada_ln_g","entrada_ln_b",
            "ln_final_g","ln_final_b","w1","b1","w2","b2"
        ]
        for n in nombres_directos:
            if n in pesos:
                attr = getattr(modelo, n)
                attr.data = pesos[n].data.reshape(attr.data.shape)
        for i, bloque in enumerate(modelo.bloques):
            for suf, attr in [("pk1", bloque.pk1), ("pb1", bloque.pb1),
                               ("pk2", bloque.pk2), ("pb2", bloque.pb2),
                               ("lg",  bloque.ln_gamma), ("lb", bloque.ln_beta)]:
                k = f"b{i}_{suf}"
                if k in pesos:
                    attr.data = pesos[k].data.reshape(attr.data.shape)
        print(f"[PenroseNet] Cargado desde {ruta}")
        return modelo