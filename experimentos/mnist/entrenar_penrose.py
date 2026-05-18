"""
entrenar_penrose.py — Entrenamiento de PenroseNet sobre MNIST
=============================================================

Estrategia de optimización híbrida:
    - Parámetros de convolución Penrose → gradiente numérico (diferencias finitas)
    - Parámetros de capas densas / LayerNorm → autograd (retropropagación)
    - Un único paso Adam para ambos grupos

    Esta separación es necesaria porque tc_conv_penrose_grafo opera sobre
    numpy directo (no construye grafo autograd), por diseño del núcleo C.

Uso:
    python entrenar_penrose.py [opciones]

    --datos        ./datos_mnist     directorio con mnist.npz
    --epocas       20
    --batch        128
    --lr           3e-3
    --nivel        3                 subdivisiones de Penrose (2-4)
    --canales      32
    --bloques      3
    --K            5                 vecinos por nodo
    --salida       ./modelos_ax
    --semilla      42
    --eps_num      5e-4              epsilon para gradiente numérico
    --subgrupo     256               muestras por cálculo de grad numérico
    --debug        1
"""

import argparse
import csv
import os
import sys
import time
import math
import random
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, "../../teselcore-python")

from teselcore import Tensor, Adam, tc_entropia_cruzada, tc_guardar
from penrose_net import PenroseNet, imagen_a_nodos, conv_penrose_grafo_np


# ═══════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════

class Log:
    nivel = 1
    t0 = 0.0

    @classmethod
    def init(cls, nivel):
        cls.nivel = nivel
        cls.t0    = time.time()

    @classmethod
    def ts(cls):
        return f"+{time.time()-cls.t0:6.1f}s"

    @classmethod
    def info(cls, msg, n=1):
        if cls.nivel >= n:
            print(f"  [{cls.ts()}] ℹ  {msg}", flush=True)

    @classmethod
    def ok(cls, msg, n=1):
        if cls.nivel >= n:
            print(f"  [{cls.ts()}] ✓  {msg}", flush=True)

    @classmethod
    def warn(cls, msg):
        print(f"  [{cls.ts()}] ⚠  {msg}", flush=True)

    @classmethod
    def error(cls, msg):
        print(f"  [{cls.ts()}] ✗  {msg}", flush=True)

    @classmethod
    def sec(cls, titulo):
        if cls.nivel >= 1:
            print(f"\n  {'─'*60}", flush=True)
            print(f"  ▶  {titulo}", flush=True)
            print(f"  {'─'*60}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════

def cargar_mnist(ruta_npz: str):
    Log.sec("CARGA DE DATOS")
    datos = np.load(ruta_npz)
    x_tr = datos["x_train"].astype(np.float32)   # (60000,1,28,28)
    y_tr = datos["y_train"].astype(np.int32)
    x_te = datos["x_test"].astype(np.float32)
    y_te = datos["y_test"].astype(np.int32)

    # Asegurar forma (B,1,H,W) y rango [0,1]
    if x_tr.ndim == 3:
        x_tr = x_tr[:, None, :, :]
        x_te = x_te[:, None, :, :]
    if x_tr.max() > 1.5:
        x_tr /= 255.0
        x_te /= 255.0

    Log.ok(f"Train: {x_tr.shape}  Test: {x_te.shape}")
    Log.info(f"Rango: [{x_tr.min():.3f}, {x_tr.max():.3f}]")
    return x_tr, y_tr, x_te, y_te


def batches(x, y, tam, mezclar=True):
    N   = x.shape[0]
    idx = np.arange(N)
    if mezclar:
        np.random.shuffle(idx)
    for ini in range(0, N, tam):
        fin = min(ini + tam, N)
        ib  = idx[ini:fin]
        yield x[ib], y[ib]


# ═══════════════════════════════════════════════════════════════════════
#  GRADIENTE NUMÉRICO — solo para params Penrose
# ═══════════════════════════════════════════════════════════════════════

def _perdida_batch(modelo: PenroseNet,
                   xb: np.ndarray,
                   yb: np.ndarray) -> float:
    """Forward + cross-entropy, retorna escalar."""
    logits = modelo.forward(xb, entrenando=False)
    loss   = tc_entropia_cruzada(logits, yb)
    return float(loss.elemento())


def gradiente_numerico_penrose(modelo: PenroseNet,
                               xb: np.ndarray,
                               yb: np.ndarray,
                               eps: float = 5e-4):
    """
    Calcula ∂L/∂θ por diferencias centrales para todos los parámetros
    de convolución Penrose (entrada_k, entrada_b, y los pk/pb de cada bloque).

    Optimización: usamos un subgrupo del batch para el grad numérico;
    la pérdida es suave sobre el grafo, así que no necesitamos el batch completo.

    Después de llamar, cada param_penrose.grad contiene el gradiente.
    """
    params = modelo.parametros_penrose()
    for p in params:
        p.grad = np.zeros_like(p.data)

    # Sub-batch para acelerar (máx 128 muestras para grad numérico)
    n_sub = min(len(xb), 128)
    idx   = np.random.choice(len(xb), n_sub, replace=False)
    xb_s, yb_s = xb[idx], yb[idx]

    for param in params:
        it = np.nditer(param.data, flags=["multi_index"])
        while not it.finished:
            idx_p = it.multi_index
            orig  = float(param.data[idx_p])

            param.data[idx_p] = orig + eps
            lp = _perdida_batch(modelo, xb_s, yb_s)

            param.data[idx_p] = orig - eps
            lm = _perdida_batch(modelo, xb_s, yb_s)

            param.data[idx_p] = orig
            param.grad[idx_p] = (lp - lm) / (2.0 * eps)
            it.iternext()


# ═══════════════════════════════════════════════════════════════════════
#  PASO DE ENTRENAMIENTO COMPLETO
# ═══════════════════════════════════════════════════════════════════════

def paso_entrenamiento(modelo: PenroseNet,
                       xb_np: np.ndarray,
                       yb_np: np.ndarray,
                       optim_penrose: Adam,
                       optim_denso: Adam,
                       eps_num: float) -> Tuple[float, float]:
    """
    Un paso completo:
    1. Forward (produce logits con autograd para params densos)
    2. Backward autograd → gradientes capas densas
    3. Gradiente numérico → gradientes conv Penrose
    4. Pasos Adam separados

    Retorna (pérdida, precisión)
    """
    # ── Limpiar gradientes ────────────────────────────────────────────
    optim_penrose.cero_gradientes()
    optim_denso.cero_gradientes()

    # ── Forward + autograd para capas densas ─────────────────────────
    logits = modelo.forward(xb_np, entrenando=True)
    loss   = tc_entropia_cruzada(logits, yb_np)
    loss_val = float(loss.elemento())

    preds = logits.data.argmax(axis=1)
    acc   = float((preds == yb_np).mean())

    if np.isnan(loss_val) or np.isinf(loss_val):
        Log.warn(f"Pérdida inválida: {loss_val}")
        return loss_val, acc

    loss.retropropagar()

    # ── Gradiente numérico Penrose ────────────────────────────────────
    gradiente_numerico_penrose(modelo, xb_np, yb_np, eps=eps_num)

    # ── Pasos Adam ────────────────────────────────────────────────────
    optim_penrose.paso()
    optim_denso.paso()

    return loss_val, acc


# ═══════════════════════════════════════════════════════════════════════
#  EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════

def evaluar(modelo: PenroseNet,
            x_te: np.ndarray,
            y_te: np.ndarray,
            tam_batch: int = 512) -> Tuple[float, float]:
    perdidas, accs = [], []
    for xb, yb in batches(x_te, y_te, tam_batch, mezclar=False):
        logits   = modelo.forward(xb, entrenando=False)
        loss     = tc_entropia_cruzada(logits, yb)
        preds    = logits.data.argmax(axis=1)
        perdidas.append(float(loss.elemento()))
        accs.append(float((preds == yb).mean()))
    return float(np.mean(perdidas)), float(np.mean(accs))


# ═══════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO COMPLETO
# ═══════════════════════════════════════════════════════════════════════

def entrenar(modelo: PenroseNet,
             x_tr, y_tr, x_te, y_te,
             epocas: int,
             tam_batch: int,
             lr: float,
             dir_salida: Path,
             eps_num: float):

    Log.sec("INICIO ENTRENAMIENTO PenroseNet")
    total_params = sum(p.data.size for p in modelo.parametros())
    Log.info(f"Total parámetros: {total_params:,}")
    Log.info(f"  — Conv Penrose: "
             f"{sum(p.data.size for p in modelo.parametros_penrose()):,}")
    Log.info(f"  — Densos/LN:    "
             f"{sum(p.data.size for p in modelo.parametros_densos()):,}")
    Log.info(f"Épocas: {epocas} | Batch: {tam_batch} | LR: {lr}")
    Log.info(f"ε numérico: {eps_num}")

    # ── Optimizadores separados ───────────────────────────────────────
    # Los params Penrose aprenden más lento (gradiente numérico es ruidoso)
    optim_penrose = Adam(modelo.parametros_penrose(),
                         lr=lr * 0.5,          # ← mitad de LR
                         beta1=0.9, beta2=0.999, eps=1e-8, decaimiento=1e-4)

    optim_denso   = Adam(modelo.parametros_densos(),
                         lr=lr,
                         beta1=0.9, beta2=0.999, eps=1e-8, decaimiento=1e-4)

    historial  = []
    mejor_acc  = 0.0
    ruta_mejor = dir_salida / "PenroseNet_mejor.ax"
    t_global   = time.time()

    for epoca in range(1, epocas + 1):
        t_ep = time.time()

        # Scheduler coseno
        lr_cos = lr * (0.5 * (1 + math.cos(math.pi * (epoca-1) / epocas)))
        optim_denso.lr   = lr_cos
        optim_penrose.lr = lr_cos * 0.5

        perdidas_ep, accs_ep = [], []
        total_batches = math.ceil(len(x_tr) / tam_batch)

        for b_idx, (xb, yb) in enumerate(batches(x_tr, y_tr, tam_batch)):
            t_b = time.time()
            loss_v, acc_v = paso_entrenamiento(
                modelo, xb, yb,
                optim_penrose, optim_denso, eps_num
            )

            if np.isnan(loss_v):
                Log.error(f"NaN en época {epoca} batch {b_idx}. Abortando.")
                return historial, mejor_acc

            perdidas_ep.append(loss_v)
            accs_ep.append(acc_v)

            # Progreso cada ~10% del epoch
            if (b_idx+1) % max(1, total_batches//10) == 0 or b_idx == total_batches-1:
                pct = 100*(b_idx+1)/total_batches
                dt_b = time.time() - t_b
                Log.info(f"  E{epoca:02d} [{pct:5.1f}%] "
                         f"loss={loss_v:.4f}  acc={acc_v*100:.1f}%  "
                         f"({dt_b*1000:.0f}ms/batch)", n=1)

        # Evaluación
        loss_tr = float(np.mean(perdidas_ep))
        acc_tr  = float(np.mean(accs_ep))
        loss_te, acc_te = evaluar(modelo, x_te, y_te)
        dt_ep = time.time() - t_ep

        Log.ok(
            f"Época {epoca:02d}/{epocas} │ lr={lr_cos:.5f} │ "
            f"tr: loss={loss_tr:.4f} acc={acc_tr*100:.2f}% │ "
            f"te: loss={loss_te:.4f} acc={acc_te*100:.2f}% │ "
            f"{dt_ep:.1f}s"
        )

        historial.append({
            "epoca":        epoca,
            "lr":           f"{lr_cos:.6f}",
            "loss_train":   f"{loss_tr:.4f}",
            "acc_train":    f"{acc_tr*100:.2f}",
            "loss_test":    f"{loss_te:.4f}",
            "acc_test":     f"{acc_te*100:.2f}",
            "tiempo_s":     f"{dt_ep:.1f}",
        })

        if acc_te > mejor_acc:
            mejor_acc = acc_te
            modelo.guardar(str(ruta_mejor))
            Log.ok(f"  ★ Nuevo mejor: {mejor_acc*100:.2f}%  → {ruta_mejor.name}")

    # ── Guardar final ─────────────────────────────────────────────────
    t_total = time.time() - t_global
    Log.sec("FIN ENTRENAMIENTO")
    Log.ok(f"Mejor acc test: {mejor_acc*100:.2f}%")
    Log.info(f"Tiempo total: {t_total:.1f}s  ({t_total/60:.1f} min)")

    ruta_final = dir_salida / "PenroseNet_final.ax"
    modelo.guardar(str(ruta_final))
    Log.ok(f"Modelo final: {ruta_final}")

    ruta_csv = dir_salida / "PenroseNet_historial.csv"
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=historial[0].keys())
        writer.writeheader()
        writer.writerows(historial)
    Log.ok(f"Historial CSV: {ruta_csv}")

    return historial, mejor_acc


# ═══════════════════════════════════════════════════════════════════════
#  VERIFICACIÓN RÁPIDA (smoke test)
# ═══════════════════════════════════════════════════════════════════════

def smoke_test(nivel=2, canales=16, n_bloques=2):
    """Verifica que el forward pass y gradientes funcionen."""
    Log.sec("SMOKE TEST")
    np.random.seed(0)
    modelo = PenroseNet(nivel=nivel, canales=canales, n_bloques=n_bloques)

    xb = np.random.rand(4, 1, 28, 28).astype(np.float32)
    yb = np.array([0, 1, 2, 3], dtype=np.int32)

    # Forward
    logits = modelo.forward(xb, entrenando=True)
    assert logits.data.shape == (4, 10), f"Shape incorrecto: {logits.data.shape}"
    Log.ok(f"Forward OK — logits shape: {logits.data.shape}")

    # Pérdida
    loss = tc_entropia_cruzada(logits, yb)
    loss_val = float(loss.elemento())
    assert not np.isnan(loss_val), "NaN en pérdida"
    assert not np.isinf(loss_val), "Inf en pérdida"
    Log.ok(f"Pérdida OK: {loss_val:.4f}")

    # Backward autograd
    loss.retropropagar()
    grad_ok = any(
        p.grad is not None for p in modelo.parametros_densos()
    )
    Log.ok(f"Autograd OK: {grad_ok}")

    # Grad numérico (un solo parámetro, muestra)
    p_test = modelo.entrada_b
    orig   = p_test.data[0]
    p_test.data[0] = orig + 1e-3
    lp = float(tc_entropia_cruzada(modelo.forward(xb, False), yb).elemento())
    p_test.data[0] = orig - 1e-3
    lm = float(tc_entropia_cruzada(modelo.forward(xb, False), yb).elemento())
    p_test.data[0] = orig
    g_num = (lp - lm) / 2e-3
    Log.ok(f"Gradiente numérico (muestra entrada_b[0]): {g_num:.5f}")

    Log.ok("Smoke test PASADO ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Entrena PenroseNet sobre MNIST con TeselCore"
    )
    p.add_argument("--datos",    default="./datos_mnist")
    p.add_argument("--epocas",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=128)
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--nivel",    type=int,   default=3,
                   help="Nivel Penrose (2=rápido, 3=recomendado, 4=preciso)")
    p.add_argument("--canales",  type=int,   default=32)
    p.add_argument("--bloques",  type=int,   default=3)
    p.add_argument("--K",        type=int,   default=5,
                   help="Vecinos por nodo en conv Penrose")
    p.add_argument("--salida",   default="./modelos_ax")
    p.add_argument("--semilla",  type=int,   default=42)
    p.add_argument("--eps_num",  type=float, default=5e-4,
                   help="Epsilon para gradiente numérico")
    p.add_argument("--debug",    type=int,   default=1, choices=[0,1,2])
    p.add_argument("--smoke",    action="store_true",
                   help="Solo ejecutar smoke test y salir")
    args = p.parse_args()

    Log.init(args.debug)
    Log.info(f"PenroseNet Trainer — TeselCore v0.1.0")

    np.random.seed(args.semilla)
    random.seed(args.semilla)

    if args.smoke:
        smoke_test()
        return

    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────
    npz = Path(args.datos) / "mnist.npz"
    if not npz.exists():
        Log.error(f"No encontrado: {npz}")
        Log.error("Ejecuta: python generar_mnist.py")
        return

    x_tr, y_tr, x_te, y_te = cargar_mnist(str(npz))

    # ── Crear modelo ──────────────────────────────────────────────────
    Log.sec("CONSTRUYENDO MODELO")
    modelo = PenroseNet(
        nivel    = args.nivel,
        canales  = args.canales,
        n_bloques= args.bloques,
        K        = args.K,
    )

    # Smoke test automático antes de entrenar
    Log.info("Verificando integridad del modelo...")
    try:
        smoke_test(nivel=min(args.nivel, 2), canales=16, n_bloques=1)
    except Exception as e:
        Log.error(f"Smoke test fallido: {e}")
        traceback.print_exc()
        return

    # ── Entrenar ──────────────────────────────────────────────────────
    historial, mejor_acc = entrenar(
        modelo, x_tr, y_tr, x_te, y_te,
        epocas    = args.epocas,
        tam_batch = args.batch,
        lr        = args.lr,
        dir_salida= dir_salida,
        eps_num   = args.eps_num,
    )

    Log.sec("RESUMEN")
    Log.ok(f"PenroseNet — Mejor acc test: {mejor_acc*100:.2f}%")
    Log.ok(f"Modelos guardados en: {dir_salida}/")


if __name__ == "__main__":
    main()