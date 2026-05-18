"""
entrenar.py — Entrenamiento de CNN Clásico sobre MNIST
=======================================================
Entrena el modelo, guarda los pesos en formato .ax y escribe
un historial CSV con métricas por época.

Uso:
    python entrenar.py [opciones]

    --datos         ./datos_mnist      directorio con mnist.npz
    --epocas        15
    --batch         64
    --lr            1e-3
    --salida        ./modelos_ax       directorio para guardar .ax
    --semilla       42
    --debug         1                  nivel de debug (0=mínimo, 1=normal, 2=verboso, 3=máximo)
"""

import argparse
import csv
import random
import time
import traceback
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "../../teselcore-python")

from teselcore import Adam, Tensor
from modelos import CNNClasico, calcular_perdida_y_precision


# ─────────────────────────────────────────────────────────────────────
#  Sistema de Debugging
# ─────────────────────────────────────────────────────────────────────

class Debug:
    """
    Niveles:
        0 — Solo errores críticos
        1 — Progreso normal (épocas, batches cada N pasos, métricas)
        2 — Verboso: stats de tensores, gradientes, pesos
        3 — Máximo: cada batch, valores individuales, histogramas
    """
    nivel: int = 1
    _batch_log_cada: int = 50
    _t_inicio_global: float = 0.0

    @classmethod
    def init(cls, nivel: int):
        cls.nivel = nivel
        cls._t_inicio_global = time.time()
        cls._log(0, f"\n{'▓'*70}")
        cls._log(0, f"  [DEBUG] Sistema inicializado — Nivel {nivel}")
        cls._log(0, f"  Marcas de tiempo relativas al inicio del proceso")
        cls._log(0, f"{'▓'*70}\n")

    @classmethod
    def _ts(cls) -> str:
        return f"+{time.time() - cls._t_inicio_global:7.2f}s"

    @classmethod
    def _log(cls, nivel_minimo: int, msg: str, prefijo: str = ""):
        if cls.nivel >= nivel_minimo:
            marca = cls._ts()
            print(f"  [{marca}]{prefijo} {msg}", flush=True)

    @classmethod
    def info(cls, msg: str):
        cls._log(1, msg, " ℹ")

    @classmethod
    def ok(cls, msg: str):
        cls._log(1, msg, " ✓")

    @classmethod
    def warn(cls, msg: str):
        cls._log(0, msg, " ⚠")

    @classmethod
    def error(cls, msg: str):
        cls._log(0, msg, " ✗")

    @classmethod
    def seccion(cls, titulo: str):
        cls._log(1, f"\n{'─'*60}\n  {titulo}\n{'─'*60}")

    @classmethod
    def dato(cls, nombre: str, valor, nivel: int = 2):
        cls._log(nivel, f"{nombre:30s} = {valor}")

    @classmethod
    def tensor_stats(cls, nombre: str, t: Tensor, nivel: int = 2):
        if cls.nivel < nivel:
            return
        d = t.data
        nan_count = int(np.isnan(d).sum())
        inf_count = int(np.isinf(d).sum())
        cls._log(nivel, (
            f"Tensor '{nombre}': forma={d.shape} "
            f"| min={d.min():.5f}  max={d.max():.5f} "
            f"| μ={d.mean():.5f}  σ={d.std():.5f} "
            f"| NaN={nan_count}  Inf={inf_count}"
        ))
        if nan_count > 0 or inf_count > 0:
            cls.error(f"  ¡VALORES INVÁLIDOS en tensor '{nombre}'!")

    @classmethod
    def grad_stats(cls, nombre: str, t: Tensor, nivel: int = 2):
        if cls.nivel < nivel or t.grad is None:
            return
        g = t.grad
        nan_count = int(np.isnan(g).sum())
        inf_count = int(np.isinf(g).sum())
        cls._log(nivel, (
            f"Grad  '{nombre}': norma={np.linalg.norm(g):.5f} "
            f"| min={g.min():.6f}  max={g.max():.6f} "
            f"| NaN={nan_count}  Inf={inf_count}"
        ))
        if nan_count > 0 or inf_count > 0:
            cls.error(f"  ¡GRADIENTES INVÁLIDOS en '{nombre}'!")

    @classmethod
    def histograma_simple(cls, nombre: str, arr: np.ndarray, bins: int = 8, nivel: int = 3):
        if cls.nivel < nivel:
            return
        counts, edges = np.histogram(arr.flatten(), bins=bins)
        max_c = counts.max() if counts.max() > 0 else 1
        cls._log(nivel, f"Histograma '{nombre}':")
        for i in range(bins):
            bar = "█" * int(counts[i] * 20 / max_c)
            cls._log(nivel, f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f}) {bar} {counts[i]}")

    @classmethod
    def batch_progress(cls, batch_idx: int, total_batches: int,
                       perdida: float, acc: float, lr: float, nivel: int = 1):
        if cls.nivel < nivel:
            return
        if batch_idx % cls._batch_log_cada == 0 or batch_idx == total_batches - 1:
            pct = 100 * (batch_idx + 1) / total_batches
            bar_len = 20
            filled = int(bar_len * (batch_idx + 1) / total_batches)
            bar = "█" * filled + "░" * (bar_len - filled)
            cls._log(nivel,
                f"Batch [{bar}] {pct:5.1f}% ({batch_idx+1}/{total_batches}) "
                f"| loss={perdida:.4f}  acc={acc*100:.1f}%  lr={lr:.6f}"
            )

    @classmethod
    def epoch_summary(cls, epoca: int, total: int, perdida_tr: float,
                      acc_tr: float, perdida_te: float, acc_te: float,
                      dt: float, lr: float):
        cls._log(1,
            f"Época {epoca:02d}/{total} │ "
            f"lr={lr:.5f} │ "
            f"loss_tr={perdida_tr:.4f}  acc_tr={acc_tr*100:.2f}% │ "
            f"loss_te={perdida_te:.4f}  acc_te={acc_te*100:.2f}% │ "
            f"{dt:.1f}s"
        )

    @classmethod
    def verificar_nan_explosion(cls, nombre: str, perdida: float) -> bool:
        if np.isnan(perdida):
            cls.error(f"NaN en pérdida de '{nombre}' — entrenamiento divergido!")
            return True
        if np.isinf(perdida):
            cls.error(f"Inf en pérdida de '{nombre}' — explosión de gradiente!")
            return True
        if abs(perdida) > 1e6:
            cls.warn(f"Pérdida muy alta ({perdida:.2e}) en '{nombre}' — posible inestabilidad")
        return False


# ─────────────────────────────────────────────────────────────────────
#  Carga de datos
# ─────────────────────────────────────────────────────────────────────

def cargar_mnist(ruta_npz: str):
    Debug.seccion("CARGA DE DATOS")
    Debug.info(f"Ruta NPZ: {ruta_npz}")

    t0 = time.time()
    datos = np.load(ruta_npz)
    Debug.ok(f"Archivo NPZ abierto en {time.time()-t0:.3f}s")

    x_tr = datos["x_train"]
    y_tr = datos["y_train"]
    x_te = datos["x_test"]
    y_te = datos["y_test"]

    Debug.info(f"Train: {x_tr.shape}  dtype={x_tr.dtype}")
    Debug.info(f"Test : {x_te.shape}  dtype={x_te.dtype}")
    Debug.dato("x_train — min / max", f"{x_tr.min():.4f} / {x_tr.max():.4f}", nivel=2)
    Debug.dato("x_train — μ / σ",     f"{x_tr.mean():.4f} / {x_tr.std():.4f}", nivel=2)

    nan_tr = int(np.isnan(x_tr).sum())
    nan_te = int(np.isnan(x_te).sum())
    if nan_tr > 0 or nan_te > 0:
        Debug.warn(f"NaN en datos: train={nan_tr}  test={nan_te}")
    else:
        Debug.ok("Sin NaN en los datos")

    Debug.ok("Datos cargados correctamente")
    return x_tr, y_tr, x_te, y_te


def generador_batches(x, y, tam_batch, mezclar=True):
    N = x.shape[0]
    idx = np.arange(N)
    if mezclar:
        np.random.shuffle(idx)
    for inicio in range(0, N, tam_batch):
        fin = min(inicio + tam_batch, N)
        ib  = idx[inicio:fin]
        yield x[ib], y[ib]


# ─────────────────────────────────────────────────────────────────────
#  Bucle de entrenamiento
# ─────────────────────────────────────────────────────────────────────

def _entrenar_epoca(modelo, x_tr, y_tr, tam_batch, optim,
                    epoca_num=1, total_epocas=1, nombre_modelo=""):
    perdidas, accs = [], []
    total_batches = int(np.ceil(len(x_tr) / tam_batch))

    Debug.info(f"Iniciando época {epoca_num}/{total_epocas} — {total_batches} batches")

    if Debug.nivel >= 2:
        _debug_parametros_modelo(modelo, nombre_modelo, prefijo="[inicio época]")

    nan_detectado = False

    for batch_idx, (xb_np, yb_np) in enumerate(
            generador_batches(x_tr, y_tr, tam_batch)):

        t_batch = time.time()

        if Debug.nivel >= 3:
            Debug.info(f"  Batch {batch_idx}: x={xb_np.shape} y={yb_np.shape}")

        optim.cero_gradientes()

        try:
            xb     = Tensor.desde_numpy(xb_np)
            logits = modelo.hacia_adelante(xb, entrenando=True)
        except Exception as e:
            Debug.error(f"Error en forward pass (batch {batch_idx}): {e}")
            if Debug.nivel >= 2:
                traceback.print_exc()
            raise

        if Debug.nivel >= 3:
            Debug.tensor_stats(f"  logits[batch={batch_idx}]", logits, nivel=3)

        try:
            perdida, acc = calcular_perdida_y_precision(logits, yb_np)
        except Exception as e:
            Debug.error(f"Error en cálculo de pérdida (batch {batch_idx}): {e}")
            raise

        perdida_val = perdida.elemento()

        if Debug.verificar_nan_explosion(nombre_modelo, perdida_val):
            nan_detectado = True
            break

        try:
            perdida.retropropagar()
        except Exception as e:
            Debug.error(f"Error en retropropagación (batch {batch_idx}): {e}")
            if Debug.nivel >= 2:
                traceback.print_exc()
            raise

        if Debug.nivel >= 3:
            _debug_gradientes_modelo(modelo, nombre_modelo, batch_idx)

        try:
            optim.paso()
        except Exception as e:
            Debug.error(f"Error en paso del optimizador (batch {batch_idx}): {e}")
            raise

        perdidas.append(perdida_val)
        accs.append(acc)

        Debug.batch_progress(batch_idx, total_batches, perdida_val, acc, optim.lr)

        if Debug.nivel >= 3:
            dt_batch = time.time() - t_batch
            Debug.info(f"   Batch {batch_idx} completado en {dt_batch*1000:.1f}ms "
                       f"| loss={perdida_val:.4f}  acc={acc*100:.1f}%")

    if nan_detectado:
        Debug.warn(f"Época {epoca_num} interrumpida por NaN/Inf")
        return float("nan"), 0.0

    perdida_media = float(np.mean(perdidas))
    acc_media     = float(np.mean(accs))

    if Debug.nivel >= 2:
        _debug_parametros_modelo(modelo, nombre_modelo, prefijo="[fin época]")

    return perdida_media, acc_media


def _evaluar(modelo, x_te, y_te, tam_batch=256, nombre_modelo=""):
    Debug.info(f"Evaluando sobre {len(x_te)} muestras de test...")
    perdidas, accs = [], []
    total_batches = int(np.ceil(len(x_te) / tam_batch))

    for b_idx, (xb_np, yb_np) in enumerate(
            generador_batches(x_te, y_te, tam_batch, mezclar=False)):

        xb     = Tensor.desde_numpy(xb_np)
        logits = modelo.hacia_adelante(xb, entrenando=False)
        perdida, acc = calcular_perdida_y_precision(logits, yb_np)
        perdida_val = perdida.elemento()

        if Debug.verificar_nan_explosion(nombre_modelo + "(test)", perdida_val):
            break

        perdidas.append(perdida_val)
        accs.append(acc)

        if Debug.nivel >= 3 and b_idx % 10 == 0:
            Debug.info(f"  Eval batch {b_idx}/{total_batches} | "
                       f"loss={perdida_val:.4f}  acc={acc*100:.1f}%")

    perdida_media = float(np.mean(perdidas))
    acc_media     = float(np.mean(accs))
    Debug.info(f"Evaluación completada: loss={perdida_media:.4f}  acc={acc_media*100:.2f}%")
    return perdida_media, acc_media


#  Helpers de debugging

def _debug_parametros_modelo(modelo, nombre: str, prefijo: str = ""):
    if Debug.nivel < 2:
        return
    params = modelo.parametros()
    Debug.info(f"{prefijo} Parámetros de '{nombre}' ({len(params)} tensores):")
    total_params = 0
    for i, p in enumerate(params):
        n_params = p.data.size
        total_params += n_params
        norma = float(np.linalg.norm(p.data))
        Debug.dato(
            f"  param[{i:02d}] forma={str(p.data.shape):20s}",
            f"norma={norma:.4f}  μ={p.data.mean():.4f}  σ={p.data.std():.4f}  "
            f"#={n_params:,}",
            nivel=2
        )
        if Debug.nivel >= 3:
            Debug.histograma_simple(f"param[{i}]", p.data, nivel=3)
    Debug.dato(f"  TOTAL parámetros", f"{total_params:,}", nivel=2)


def _debug_gradientes_modelo(modelo, nombre: str, batch_idx: int):
    if Debug.nivel < 3:
        return
    params = modelo.parametros()
    Debug.info(f"  Gradientes tras batch {batch_idx}:")
    for i, p in enumerate(params):
        if p.grad is not None:
            norma_g = float(np.linalg.norm(p.grad))
            norma_p = float(np.linalg.norm(p.data))
            ratio   = norma_g / (norma_p + 1e-8)
            Debug.dato(
                f"    grad[{i:02d}]",
                f"‖g‖={norma_g:.5f}  ‖p‖={norma_p:.5f}  ratio={ratio:.5f}",
                nivel=3
            )
            if ratio > 10:
                Debug.warn(f"    ratio grad/peso muy alto: {ratio:.2f}")
            if ratio < 1e-7 and norma_g > 0:
                Debug.warn(f"    ratio grad/peso muy bajo: {ratio:.2e}")
        else:
            Debug.dato(f"    grad[{i:02d}]", "None (sin gradiente)", nivel=3)


def _debug_optimizador(optim: Adam, nombre: str):
    if Debug.nivel < 3:
        return
    Debug.info(f"Estado Adam '{nombre}': t={optim.t}  lr={optim.lr:.6f}")
    for i, (m1, m2) in enumerate(zip(optim.m1, optim.m2)):
        Debug.dato(
            f"  Adam[{i:02d}] m1_norma / m2_norma",
            f"{np.linalg.norm(m1):.5f} / {np.linalg.norm(m2):.5f}",
            nivel=3
        )


def entrenar_modelo(nombre: str,
                    modelo,
                    x_tr, y_tr, x_te, y_te,
                    epocas: int,
                    tam_batch: int,
                    lr: float,
                    dir_salida: Path):

    Debug.seccion(f"ENTRENANDO: {nombre}")
    Debug.dato("Épocas", epocas)
    Debug.dato("Batch size", tam_batch)
    Debug.dato("LR inicial", lr)
    Debug.dato("Muestras train", len(x_tr))
    Debug.dato("Muestras test",  len(x_te))
    Debug.dato("Directorio salida", str(dir_salida))

    params = modelo.parametros()
    total_params = sum(p.data.size for p in params)
    Debug.info(f"Modelo: {len(params)} tensores de parámetros | {total_params:,} parámetros totales")
    _debug_parametros_modelo(modelo, nombre, prefijo="[inicio entrenamiento]")

    optim = Adam(params, lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, decaimiento=1e-4)
    Debug.info(f"Optimizador: Adam(lr={lr}, β1=0.9, β2=0.999, ε=1e-8, wd=1e-4)")

    historial = []
    mejor_acc  = 0.0
    ruta_mejor = dir_salida / f"{nombre}_mejor.ax"

    t_entrenamiento_total = time.time()

    for epoca in range(1, epocas + 1):
        t_epoca = time.time()

        # Programador de LR coseno
        lr_actual = lr * (0.5 * (1 + np.cos(np.pi * (epoca - 1) / epocas)))
        optim.lr  = lr_actual

        if Debug.nivel >= 2:
            Debug.info(f"\n{'·'*50}")
            Debug.info(f"ÉPOCA {epoca}/{epocas} — LR coseno: {lr_actual:.6f}")
        else:
            Debug.info(f"\nÉpoca {epoca}/{epocas} | LR={lr_actual:.6f}")

        try:
            perdida_tr, acc_tr = _entrenar_epoca(
                modelo, x_tr, y_tr, tam_batch, optim,
                epoca_num=epoca,
                total_epocas=epocas,
                nombre_modelo=nombre
            )
        except Exception as e:
            Debug.error(f"Error en entrenamiento época {epoca}: {e}")
            if Debug.nivel >= 2:
                traceback.print_exc()
            raise

        if np.isnan(perdida_tr):
            Debug.error(f"Entrenamiento abortado en época {epoca} por NaN")
            break

        try:
            perdida_te, acc_te = _evaluar(modelo, x_te, y_te, nombre_modelo=nombre)
        except Exception as e:
            Debug.error(f"Error en evaluación época {epoca}: {e}")
            raise

        dt_epoca = time.time() - t_epoca

        Debug.epoch_summary(epoca, epocas, perdida_tr, acc_tr,
                            perdida_te, acc_te, dt_epoca, lr_actual)

        if Debug.nivel >= 2:
            gap_loss = perdida_te - perdida_tr
            gap_acc  = acc_tr - acc_te
            Debug.dato("  Gap pérdida (test-train)", f"{gap_loss:+.4f}", nivel=2)
            Debug.dato("  Gap acc (train-test)",     f"{gap_acc*100:+.2f}%", nivel=2)
            if gap_acc > 0.10:
                Debug.warn(f"  Posible overfitting: gap acc = {gap_acc*100:.1f}%")

        if Debug.nivel >= 3:
            _debug_optimizador(optim, nombre)

        fila = {
            "modelo":       nombre,
            "epoca":        epoca,
            "lr":           f"{lr_actual:.6f}",
            "perdida_train":f"{perdida_tr:.4f}",
            "acc_train":    f"{acc_tr*100:.2f}",
            "perdida_test": f"{perdida_te:.4f}",
            "acc_test":     f"{acc_te*100:.2f}",
            "tiempo_s":     f"{dt_epoca:.1f}",
        }
        historial.append(fila)

        if acc_te > mejor_acc:
            mejor_acc = acc_te
            try:
                modelo.guardar(str(ruta_mejor))
                Debug.ok(f"Nuevo mejor modelo guardado: {mejor_acc*100:.2f}%  → {ruta_mejor.name}")
            except Exception as e:
                Debug.error(f"Error al guardar mejor modelo: {e}")

    t_total = time.time() - t_entrenamiento_total

    Debug.seccion(f"FIN ENTRENAMIENTO: {nombre}")
    Debug.info(f"Tiempo total: {t_total:.1f}s  ({t_total/60:.1f} min)")
    Debug.dato("Mejor acc test",         f"{mejor_acc*100:.2f}%")
    Debug.dato("Épocas completadas",     len(historial))
    Debug.dato("Tiempo por época (prom)",f"{t_total/max(len(historial),1):.1f}s")

    ruta_final = dir_salida / f"{nombre}_final.ax"
    try:
        modelo.guardar(str(ruta_final))
        Debug.ok(f"Modelo final guardado: {ruta_final.name}")
    except Exception as e:
        Debug.error(f"Error al guardar modelo final: {e}")

    ruta_csv = dir_salida / f"{nombre}_historial.csv"
    try:
        with open(ruta_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=historial[0].keys())
            writer.writeheader()
            writer.writerows(historial)
        Debug.ok(f"Historial CSV guardado: {ruta_csv.name}")
    except Exception as e:
        Debug.error(f"Error al guardar CSV: {e}")

    _debug_parametros_modelo(modelo, nombre, prefijo="[fin entrenamiento]")

    return historial, mejor_acc


#  Punto de entrada
def main():
    parser = argparse.ArgumentParser(
        description="Entrena CNN Clásico sobre MNIST con TeselCore"
    )
    parser.add_argument("--datos",   default="./datos_mnist",
                        help="Directorio con mnist.npz")
    parser.add_argument("--epocas",  type=int,   default=15)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--salida",  default="./modelos_ax")
    parser.add_argument("--semilla", type=int,   default=42)
    parser.add_argument("--debug",   type=int,   default=1,
                        choices=[0, 1, 2, 3],
                        help=(
                            "Nivel de debugging: "
                            "0=solo errores, "
                            "1=progreso normal, "
                            "2=stats de tensores/gradientes, "
                            "3=máximo (cada batch, histogramas)"
                        ))
    args = parser.parse_args()

    Debug.init(args.debug)
    Debug.info("Argumentos recibidos:")
    for k, v in vars(args).items():
        Debug.dato(f"  --{k}", v, nivel=1)

    np.random.seed(args.semilla)
    random.seed(args.semilla)
    Debug.ok(f"Semilla aleatoria fijada: {args.semilla}")

    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    Debug.ok(f"Directorio de salida: {dir_salida.resolve()}")

    Debug.seccion("VERIFICACIÓN DE ENTORNO")
    Debug.dato("Python", sys.version.split()[0])
    Debug.dato("NumPy",  np.__version__)
    try:
        import teselcore
        Debug.ok("TeselCore importado correctamente")
    except ImportError as e:
        Debug.warn(f"TeselCore: {e}")

    npz_ruta = Path(args.datos) / "mnist.npz"
    if not npz_ruta.exists():
        Debug.error(f"No se encontró {npz_ruta}")
        Debug.error("Ejecuta primero: python generar_mnist.py")
        return

    x_tr, y_tr, x_te, y_te = cargar_mnist(str(npz_ruta))

    Debug.seccion("INICIALIZANDO CNN CLÁSICO")
    np.random.seed(args.semilla)
    try:
        modelo_cnn = CNNClasico()
        Debug.ok("CNNClasico instanciado")
    except Exception as e:
        Debug.error(f"Error al crear CNNClasico: {e}")
        raise

    historial, mejor_acc = entrenar_modelo(
        "CNNClasico", modelo_cnn,
        x_tr, y_tr, x_te, y_te,
        epocas=args.epocas,
        tam_batch=args.batch,
        lr=args.lr,
        dir_salida=dir_salida,
    )

    Debug.seccion("RESUMEN FINAL")
    Debug.info(f"  CNNClasico: {mejor_acc*100:.2f}% precisión test")
    Debug.ok(f"\n¡Entrenamiento completo!")
    Debug.ok(f"Pesos guardados en: {dir_salida}/")
    Debug.info(f"Tiempo total del proceso: {time.time() - Debug._t_inicio_global:.1f}s")


if __name__ == "__main__":
    main()