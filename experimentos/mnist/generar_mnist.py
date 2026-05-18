"""
generar_mnist.py — Descarga y prepara el dataset MNIST
=======================================================
Descarga MNIST desde los servidores oficiales de Yann LeCun
(o usa torchvision/keras si están disponibles) y lo guarda
en un formato .npz portable que usan los demás scripts.

Uso:
    python generar_mnist.py [--destino ./datos_mnist]
"""

import argparse
import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
#  URLs espejo (LeCun + alternativas)
# ──────────────────────────────────────────────────────────────
ESPEJOS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

ARCHIVOS = {
    "train_imagenes": "train-images-idx3-ubyte.gz",
    "train_etiquetas": "train-labels-idx1-ubyte.gz",
    "test_imagenes":  "t10k-images-idx3-ubyte.gz",
    "test_etiquetas": "t10k-labels-idx1-ubyte.gz",
}

# SHA256 para verificar integridad
SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz":  "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz":  "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def _sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _descargar(nombre_archivo: str, destino: Path) -> Path:
    ruta = destino / nombre_archivo
    if ruta.exists():
        print(f"  [✓] Ya existe: {nombre_archivo}")
        return ruta

    ultimo_error = None
    for espejo in ESPEJOS:
        url = espejo + nombre_archivo
        try:
            print(f"  Descargando {url} ...", end="", flush=True)
            urllib.request.urlretrieve(url, ruta)
            print(" OK")
            return ruta
        except Exception as e:
            ultimo_error = e
            print(f" FALLO ({e})")

    raise RuntimeError(f"No se pudo descargar {nombre_archivo}: {ultimo_error}")


def _verificar(nombre_archivo: str, ruta: Path) -> bool:
    esperado = SHA256.get(nombre_archivo)
    if esperado is None:
        return True
    actual = _sha256_archivo(ruta)
    if actual != esperado:
        print(f"  [!] SHA256 mismatch para {nombre_archivo}: {actual[:16]}... != {esperado[:16]}...")
        return False
    return True


def _leer_imagenes(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803, f"Magic incorrecto: {magic}"
        datos = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)
    return datos


def _leer_etiquetas(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, f"Magic incorrecto: {magic}"
        datos = np.frombuffer(f.read(), dtype=np.uint8)
    return datos


def _intentar_con_keras(destino: Path) -> bool:
    """Descarga rápida si Keras está instalado."""
    try:
        import tensorflow.keras as keras  # type: ignore
        print("[INFO] Usando tensorflow.keras para descargar MNIST.")
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
        _guardar_npz(destino, x_tr, y_tr, x_te, y_te)
        return True
    except ImportError:
        return False


def _intentar_con_torch(destino: Path) -> bool:
    """Descarga rápida si PyTorch está instalado."""
    try:
        import torchvision  # type: ignore
        import torchvision.transforms as T  # type: ignore
        print("[INFO] Usando torchvision para descargar MNIST.")
        ds_tr = torchvision.datasets.MNIST(str(destino), train=True,  download=True)
        ds_te = torchvision.datasets.MNIST(str(destino), train=False, download=True)
        x_tr = ds_tr.data.numpy(); y_tr = ds_tr.targets.numpy()
        x_te = ds_te.data.numpy(); y_te = ds_te.targets.numpy()
        _guardar_npz(destino, x_tr, y_tr, x_te, y_te)
        return True
    except ImportError:
        return False


def _guardar_npz(destino: Path, x_tr, y_tr, x_te, y_te):
    npz_ruta = destino / "mnist.npz"
    # Normalizar a float32 [0, 1] y añadir dimensión de canal
    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0

    # normalización estándar MNIST: μ=0.1307, σ=0.3081
    mu  = 0.1307
    std = 0.3081
    x_tr = (x_tr - mu) / std
    x_te = (x_te - mu) / std

    # (N, H, W) → (N, 1, H, W)
    x_tr = x_tr[:, None, :, :]
    x_te = x_te[:, None, :, :]

    np.savez_compressed(npz_ruta,
                        x_train=x_tr, y_train=y_tr.astype(np.int32),
                        x_test=x_te,  y_test=y_te.astype(np.int32))

    print(f"\n[✓] Dataset guardado en: {npz_ruta}")
    print(f"    x_train: {x_tr.shape}   y_train: {y_tr.shape}")
    print(f"    x_test:  {x_te.shape}   y_test:  {y_te.shape}")
    print(f"    Rango x_train — min:{x_tr.min():.3f}  max:{x_tr.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Generador de dataset MNIST para TeselCore")
    parser.add_argument("--destino", default="./datos_mnist", help="Directorio de salida")
    parser.add_argument("--sin_cache", action="store_true",
                        help="Forzar nueva descarga aunque ya existan los archivos")
    args = parser.parse_args()

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)

    npz_ruta = destino / "mnist.npz"
    if npz_ruta.exists() and not args.sin_cache:
        print(f"[✓] mnist.npz ya existe en {npz_ruta}. Usa --sin_cache para regenerar.")
        _mostrar_estadisticas(npz_ruta)
        return

    print("=" * 60)
    print("  Generador de Dataset MNIST para TeselCore")
    print("=" * 60)

    # Intentar librerías de alto nivel primero
    if _intentar_con_torch(destino):
        _mostrar_estadisticas(npz_ruta)
        return
    if _intentar_con_keras(destino):
        _mostrar_estadisticas(npz_ruta)
        return

    # Descarga manual desde IDX
    print("\n[INFO] Descargando archivos IDX directamente...")
    rutas = {}
    for clave, nombre in ARCHIVOS.items():
        ruta = _descargar(nombre, destino)
        if not _verificar(nombre, ruta):
            ruta.unlink()
            ruta = _descargar(nombre, destino)
        rutas[clave] = ruta

    print("\n[INFO] Leyendo archivos IDX...")
    x_tr = _leer_imagenes(rutas["train_imagenes"])
    y_tr = _leer_etiquetas(rutas["train_etiquetas"])
    x_te = _leer_imagenes(rutas["test_imagenes"])
    y_te = _leer_etiquetas(rutas["test_etiquetas"])

    _guardar_npz(destino, x_tr, y_tr, x_te, y_te)
    _mostrar_estadisticas(npz_ruta)


def _mostrar_estadisticas(npz_ruta: Path):
    datos = np.load(npz_ruta)
    print("\n─── Estadísticas del dataset ───────────────────────────")
    for k in datos.files:
        a = datos[k]
        if a.dtype in (np.float32, np.float64):
            print(f"  {k:12s}: {a.shape}  μ={a.mean():.4f}  σ={a.std():.4f}")
        else:
            print(f"  {k:12s}: {a.shape}  clases={sorted(set(a.flat))}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()