"""
modelos.py — CNN clásico para MNIST con TeselCore
==================================================
Define la arquitectura:

  CNNClasico — Conv2D estándar (espejo del tc_conv2d del núcleo C)

Implementa:
    .hacia_adelante(x, entrenando) → logits Tensor (B, 10)
    .parametros()                  → List[Tensor]
    .guardar(ruta)                 → escribe .ax
    .cargar(ruta)                  → carga .ax (class-method)
"""

import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, "../../teselcore-python")

from teselcore import (
    Tensor,
    tc_conv2d,
    tc_agrupacion_max,
    tc_normalizacion_lote,
    tc_abandono,
    tc_entropia_cruzada,
    tc_guardar,
    tc_cargar,
)

#  Inicializadores de Kaiming / He

def _kaiming(forma: tuple, fan_in: int) -> Tensor:
    std = math.sqrt(2.0 / fan_in)
    return Tensor(np.random.randn(*forma).astype(np.float32) * std, requiere_grad=True)

def _ceros(forma: tuple) -> Tensor:
    return Tensor(np.zeros(forma, dtype=np.float32), requiere_grad=True)

def _unos(forma: tuple) -> Tensor:
    return Tensor(np.ones(forma, dtype=np.float32), requiere_grad=True)


class CNNClasico:
    """
    Arquitectura:
        Conv(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2,2)   [14×14]
        Conv(32→64,3×3, pad=1) → BN → ReLU → MaxPool(2,2)   [ 7×7]
        Flatten → FC(3136→256) → Dropout(0.4) → ReLU
        FC(256→10)
    """

    def __init__(self):
        # Bloque conv 1
        self.w1  = _kaiming((32,  1, 3, 3),  1*3*3)
        self.b1  = _ceros((32,))
        self.gn1 = _unos((32,))
        self.bn1 = _ceros((32,))

        # Bloque conv 2
        self.w2  = _kaiming((64, 32, 3, 3), 32*3*3)
        self.b2  = _ceros((64,))
        self.gn2 = _unos((64,))
        self.bn2 = _ceros((64,))

        # BN state (running stats — no req_grad)
        self._bn1_mu  = np.zeros(32, np.float32)
        self._bn1_var = np.ones(32,  np.float32)
        self._bn2_mu  = np.zeros(64, np.float32)
        self._bn2_var = np.ones(64,  np.float32)

        # Capas FC
        self.w3 = _kaiming((256, 64*7*7), 64*7*7)
        self.b3 = _ceros((256,))
        self.w4 = _kaiming((10,  256),    256)
        self.b4 = _ceros((10,))

    def parametros(self) -> List[Tensor]:
        return [self.w1, self.b1, self.gn1, self.bn1,
                self.w2, self.b2, self.gn2, self.bn2,
                self.w3, self.b3,
                self.w4, self.b4]

    def hacia_adelante(self, x: Tensor, entrenando: bool = True) -> Tensor:
        # ── Bloque 1
        x = tc_conv2d(x, self.w1, self.b1, paso=1, relleno=1)
        x = tc_normalizacion_lote(x, self.gn1, self.bn1, entrenando=entrenando)
        x = x.relu()
        x = tc_agrupacion_max(x, kernel=2, paso=2)

        # ── Bloque 2
        x = tc_conv2d(x, self.w2, self.b2, paso=1, relleno=1)
        x = tc_normalizacion_lote(x, self.gn2, self.bn2, entrenando=entrenando)
        x = x.relu()
        x = tc_agrupacion_max(x, kernel=2, paso=2)

        # ── Flatten + FC
        B = x.data.shape[0]
        x = x.reformar(B, -1)

        # FC3
        x = x.matmul(self.w3.transponer())
        x = x + Tensor(np.tile(self.b3.data, (B, 1)))
        x = tc_abandono(x, p=0.4, entrenando=entrenando)
        x = x.relu()

        # FC4 (logits)
        x = x.matmul(self.w4.transponer())
        x = x + Tensor(np.tile(self.b4.data, (B, 1)))
        return x

    #Serializacion 

    def guardar(self, ruta: str):
        tc_guardar({
            "w1": self.w1, "b1": self.b1, "gn1": self.gn1, "bn1": self.bn1,
            "w2": self.w2, "b2": self.b2, "gn2": self.gn2, "bn2": self.bn2,
            "w3": self.w3, "b3": self.b3,
            "w4": self.w4, "b4": self.b4,
        }, ruta, metadatos='{"modelo":"CNNClasico","version":"1.0"}')

    @classmethod
    def cargar(cls, ruta: str) -> "CNNClasico":
        modelo = cls()
        pesos  = tc_cargar(ruta)
        for nombre, tensor in pesos.items():
            attr = getattr(modelo, nombre)
            attr.data[:] = tensor.data
        print(f"[CNNClasico] Pesos cargados desde {ruta}")
        return modelo

#  FUNCION DE PÉRDIDA COMPARTIDA

def calcular_perdida_y_precision(logits: Tensor,
                                  etiquetas: np.ndarray) -> Tuple[Tensor, float]:
    perdida = tc_entropia_cruzada(logits, etiquetas)
    preds   = logits.data.argmax(axis=1)
    acc     = (preds == etiquetas).mean()
    return perdida, float(acc)