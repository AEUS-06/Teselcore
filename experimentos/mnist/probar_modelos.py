import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, "../../teselcore-python")

from teselcore import Tensor
from modelos import CNNClasico, calcular_perdida_y_precision
from penrose_net import PenroseNet


# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────

def cargar_mnist_test(ruta_npz: str):
    datos = np.load(ruta_npz)
    return datos["x_test"], datos["y_test"]


def generador_batches(x, y, tam_batch):
    N = x.shape[0]
    for inicio in range(0, N, tam_batch):
        fin = min(inicio + tam_batch, N)
        yield x[inicio:fin], y[inicio:fin]


# ─────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────

def evaluar_modelo(nombre, modelo, x_te, y_te, tam_batch):
    print(f"\nEvaluando {nombre} ...")

    N = x_te.shape[0]
    predicciones = np.zeros(N, dtype=np.int32)
    perdidas = []
    t_total = 0.0

    for i, (xb_np, yb_np) in enumerate(generador_batches(x_te, y_te, tam_batch)):

        t0 = time.perf_counter()

        if nombre == "PenroseNet":
            logits = modelo.forward(xb_np, entrenando=False)
        else:
            xb = Tensor.desde_numpy(xb_np)
            logits = modelo.hacia_adelante(xb, entrenando=False)

        t_total += time.perf_counter() - t0

        perdida, _ = calcular_perdida_y_precision(logits, yb_np)

        if hasattr(perdida, "elemento"):
            perdidas.append(float(perdida.elemento()))
        else:
            perdidas.append(float(perdida))

        if hasattr(logits, "data"):
            pred = logits.data.argmax(axis=1)
        else:
            pred = logits.argmax(axis=1)

        ini = i * tam_batch
        fin = min(ini + tam_batch, N)
        predicciones[ini:fin] = pred

    # ── Métricas
    correctas = (predicciones == y_te).sum()
    acc = correctas / N

    # Matriz de confusión
    cm = np.zeros((10, 10), dtype=int)
    for p, r in zip(predicciones, y_te):
        cm[r, p] += 1

    tiempo_total_ms = t_total * 1000
    tiempo_por_muestra_us = (t_total / N) * 1e6

    return {
        "nombre": nombre,
        "num_muestras": int(N),
        "correctas": int(correctas),
        "precision_global": float(acc),
        "perdida": float(np.mean(perdidas)),
        "matriz_confusion": cm.tolist(),
        "tiempo_total_ms": float(tiempo_total_ms),
        "tiempo_por_muestra_us": float(tiempo_por_muestra_us),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default="./datos_mnist")
    parser.add_argument("--modelos_ax", default="./modelos_ax")
    parser.add_argument("--batch", type=int, default=256)

    args = parser.parse_args()

    # ── Cargar datos
    x_te, y_te = cargar_mnist_test(f"{args.datos}/mnist.npz")
    print(f"[datos] {x_te.shape}")

    dir_modelos = Path(args.modelos_ax)

    if not dir_modelos.exists():
        print("[ERROR] Carpeta de modelos no existe")
        return

    archivos = list(dir_modelos.glob("*.ax"))

    if not archivos:
        print("[ERROR] No hay modelos .ax")
        return

    print("\nModelos detectados:")
    for f in archivos:
        print(" -", f.name)

    resultados = {}

    mejor_cnn = None
    mejor_penrose = None

    # ─────────────────────────────────────────
    # CARGA Y EVALUACIÓN
    # ─────────────────────────────────────────

    for archivo in archivos:
        nombre_archivo = archivo.name.lower()

        try:
            if "cnnclasico" in nombre_archivo:
                print(f"\n[] Cargando CNNClasico: {archivo.name}")
                modelo = CNNClasico.cargar(str(archivo))
                res = evaluar_modelo("CNNClasico", modelo, x_te, y_te, args.batch)

                if mejor_cnn is None or res["precision_global"] > mejor_cnn["precision_global"]:
                    mejor_cnn = res

            elif "penrose" in nombre_archivo:
                print(f"\n[] Cargando PenroseNet: {archivo.name}")
                modelo = PenroseNet.cargar(str(archivo))
                res = evaluar_modelo("PenroseNet", modelo, x_te, y_te, args.batch)

                if mejor_penrose is None or res["precision_global"] > mejor_penrose["precision_global"]:
                    mejor_penrose = res

        except Exception as e:
            print(f"[ERROR] {archivo.name}: {e}")

    # ─────────────────────────────────────────
    # RESULTADOS FINALES (compatibles con gráficas)
    # ─────────────────────────────────────────

    if mejor_cnn:
        resultados["cnn"] = mejor_cnn
    if mejor_penrose:
        resultados["penrose"] = mejor_penrose

    print("\nRESULTADOS FINALES:")
    for k, v in resultados.items():
        print(
            f"{k}: acc={v['precision_global']:.4f}, "
            f"loss={v['perdida']:.4f}, "
            f"tiempo={v['tiempo_por_muestra_us']:.2f} us"
        )

    # Guardar JSON compatible con tu script de gráficas
    with open("comparativa_resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print("\n[] Guardado en comparativa_resultados.json")


if __name__ == "__main__":
    main()