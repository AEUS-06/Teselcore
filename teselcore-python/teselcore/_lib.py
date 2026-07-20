"""
teselcore._lib — Carga de la librería C y definiciones ctypes
================================================================
Este módulo NO reimplementa ninguna operación matemática: solo declara
la forma exacta (ctypes.Structure) de los structs de teselcore.h y las
firmas (argtypes/restype) de cada función exportada por libteselcore.so.
Toda la aritmética real ocurre en C.
"""
import ctypes
import os
from pathlib import Path

TC_MAX_DIMS = 8

# ── Constantes que reflejan los enums de teselcore.h ──────────────────
TC_FLOAT32, TC_FLOAT64, TC_INT32, TC_INT8, TC_FLOAT16 = 0, 1, 2, 3, 4
TC_CPU, TC_CUDA, TC_METAL, TC_WASM = 0, 1, 2, 3
TC_PENROSE_KITE, TC_PENROSE_DART = 0, 1


# ── Structs (deben coincidir EXACTAMENTE con teselcore.h) ─────────────
class TcTensor(ctypes.Structure):
    _fields_ = [
        ("datos", ctypes.c_void_p),
        ("gradiente", ctypes.POINTER(ctypes.c_float)),
        ("forma", ctypes.c_int * TC_MAX_DIMS),
        ("pasos", ctypes.c_int * TC_MAX_DIMS),
        ("ndim", ctypes.c_int),
        ("total", ctypes.c_size_t),
        ("tipo", ctypes.c_int),
        ("dispositivo", ctypes.c_int),
        ("requiere_grad", ctypes.c_int),
        ("_refs", ctypes.c_int),
        ("_nodo_grad", ctypes.c_void_p),
    ]


TcTensorP = ctypes.POINTER(TcTensor)


class TcTejaPenrose(ctypes.Structure):
    _fields_ = [
        ("centro_x", ctypes.c_float),
        ("centro_y", ctypes.c_float),
        ("angulo", ctypes.c_float),
        ("tipo", ctypes.c_int),
        ("vecinos", ctypes.c_int * 7),
        ("num_vecinos", ctypes.c_int),
        ("puntos", (ctypes.c_float * 2) * 4),
    ]


class TcTeselacionPenrose(ctypes.Structure):
    _fields_ = [
        ("tejas", ctypes.POINTER(TcTejaPenrose)),
        ("num_tejas", ctypes.c_int),
        ("nivel", ctypes.c_int),
        ("escala", ctypes.c_float),
    ]


TcTeselacionPenroseP = ctypes.POINTER(TcTeselacionPenrose)


class TcKernelPenrose(ctypes.Structure):
    _fields_ = [
        ("teselacion", TcTeselacionPenroseP),
        ("pesos", TcTensorP),
        ("sesgo", TcTensorP),
        ("canales_entrada", ctypes.c_int),
        ("canales_salida", ctypes.c_int),
    ]


TcKernelPenroseP = ctypes.POINTER(TcKernelPenrose)


class TcTensorNombrado(ctypes.Structure):
    _fields_ = [("nombre", ctypes.c_char * 256), ("tensor", TcTensorP)]


class TcModelo(ctypes.Structure):
    _fields_ = [
        ("tensores", ctypes.POINTER(TcTensorNombrado)),
        ("num_tensores", ctypes.c_int),
        ("metadatos", ctypes.c_char_p),
        ("version_mayor", ctypes.c_uint16),
        ("version_menor", ctypes.c_uint16),
    ]


TcModeloP = ctypes.POINTER(TcModelo)
TcOptimizadorP = ctypes.c_void_p  # struct opaco en el header público


# ── Localizar y cargar libteselcore.so ────────────────────────────────
def _buscar_libreria() -> ctypes.CDLL:
    candidatos = [
        Path(__file__).parent / "libteselcore.so",
        Path(__file__).parent.parent / "libteselcore.so",
        Path(__file__).parent.parent.parent / "teselcore-nucleo" / "build" / "libteselcore.so",
        Path("/usr/local/lib/libteselcore.so"),
    ]
    override = os.environ.get("TESELCORE_LIB")
    if override:
        candidatos.insert(0, Path(override))

    for ruta in candidatos:
        if ruta.exists():
            return ctypes.CDLL(str(ruta))

    raise OSError(
        "No se encontró libteselcore.so. Compílala con 'make so' en "
        "teselcore-nucleo/ o exporta TESELCORE_LIB=/ruta/a/libteselcore.so"
    )


LIB = _buscar_libreria()

Pf   = ctypes.POINTER(ctypes.c_float)
Pi   = ctypes.POINTER(ctypes.c_int)
c_i  = ctypes.c_int
c_f  = ctypes.c_float
c_sz = ctypes.c_size_t


def _firma(nombre, restype, argtypes):
    """Registra argtypes/restype de una función de la librería C."""
    fn = getattr(LIB, nombre)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# ── Creación de tensores ──────────────────────────────────────────────
tc_vacio              = _firma("tc_vacio", TcTensorP, [Pi, c_i, c_i, c_i])
tc_ceros              = _firma("tc_ceros", TcTensorP, [Pi, c_i])
tc_unos                = _firma("tc_unos", TcTensorP, [Pi, c_i])
tc_aleatorio_uniforme  = _firma("tc_aleatorio_uniforme", TcTensorP, [Pi, c_i])
tc_aleatorio_normal    = _firma("tc_aleatorio_normal", TcTensorP, [Pi, c_i])
tc_desde_datos         = _firma("tc_desde_datos", TcTensorP, [ctypes.c_void_p, Pi, c_i, c_i])
tc_escalar             = _firma("tc_escalar", TcTensorP, [c_f])
tc_clonar              = _firma("tc_clonar", TcTensorP, [TcTensorP])
tc_liberar             = _firma("tc_liberar", None, [TcTensorP])
tc_elemento            = _firma("tc_elemento", c_f, [TcTensorP])
tc_cero_gradiente      = _firma("tc_cero_gradiente", None, [TcTensorP])
tc_imprimir            = _firma("tc_imprimir", None, [TcTensorP])
tc_imprimir_info       = _firma("tc_imprimir_info", None, [])
tc_semilla_aleatoria   = _firma("tc_semilla_aleatoria", None, [ctypes.c_uint64])


# ── Operaciones aritméticas ───────────────────────────────────────────
tc_sumar          = _firma("tc_sumar", TcTensorP, [TcTensorP, TcTensorP])
tc_restar         = _firma("tc_restar", TcTensorP, [TcTensorP, TcTensorP])
tc_multiplicar    = _firma("tc_multiplicar", TcTensorP, [TcTensorP, TcTensorP])
tc_dividir        = _firma("tc_dividir", TcTensorP, [TcTensorP, TcTensorP])
tc_negar          = _firma("tc_negar", TcTensorP, [TcTensorP])
tc_potencia       = _firma("tc_potencia", TcTensorP, [TcTensorP, c_f])
tc_logaritmo      = _firma("tc_logaritmo", TcTensorP, [TcTensorP])
tc_exponencial    = _firma("tc_exponencial", TcTensorP, [TcTensorP])
tc_raiz           = _firma("tc_raiz", TcTensorP, [TcTensorP])
tc_valor_absoluto = _firma("tc_valor_absoluto", TcTensorP, [TcTensorP])

# ── Reducción ──────────────────────────────────────────────────────────
tc_suma_dimension   = _firma("tc_suma_dimension", TcTensorP, [TcTensorP, c_i, c_i])
tc_media_dimension  = _firma("tc_media_dimension", TcTensorP, [TcTensorP, c_i, c_i])
tc_maximo_dimension = _firma("tc_maximo_dimension", TcTensorP, [TcTensorP, c_i, c_i])
tc_minimo_dimension = _firma("tc_minimo_dimension", TcTensorP, [TcTensorP, c_i, c_i])

# ── Álgebra lineal ─────────────────────────────────────────────────────
tc_multiplicacion_matricial = _firma("tc_multiplicacion_matricial", TcTensorP, [TcTensorP, TcTensorP])
tc_transponer               = _firma("tc_transponer", TcTensorP, [TcTensorP, c_i, c_i])
tc_reformar                 = _firma("tc_reformar", TcTensorP, [TcTensorP, Pi, c_i])
tc_aplanar                  = _firma("tc_aplanar", TcTensorP, [TcTensorP, c_i])

# ── Activaciones ───────────────────────────────────────────────────────
tc_relu                 = _firma("tc_relu", TcTensorP, [TcTensorP])
tc_relu_con_fuga        = _firma("tc_relu_con_fuga", TcTensorP, [TcTensorP, c_f])
tc_sigmoide             = _firma("tc_sigmoide", TcTensorP, [TcTensorP])
tc_tangente_hiperbolica = _firma("tc_tangente_hiperbolica", TcTensorP, [TcTensorP])
tc_softmax              = _firma("tc_softmax", TcTensorP, [TcTensorP, c_i])
tc_gelu                 = _firma("tc_gelu", TcTensorP, [TcTensorP])
tc_silu                 = _firma("tc_silu", TcTensorP, [TcTensorP])

# ── Capas ──────────────────────────────────────────────────────────────
tc_lineal              = _firma("tc_lineal", TcTensorP, [TcTensorP, TcTensorP, TcTensorP])
tc_conv2d              = _firma("tc_conv2d", TcTensorP, [TcTensorP, TcTensorP, TcTensorP, c_i, c_i])
tc_agrupacion_max      = _firma("tc_agrupacion_max", TcTensorP, [TcTensorP, c_i, c_i])
tc_agrupacion_promedio = _firma("tc_agrupacion_promedio", TcTensorP, [TcTensorP, c_i, c_i])
tc_normalizacion_lote  = _firma("tc_normalizacion_lote", TcTensorP,
                                [TcTensorP, TcTensorP, TcTensorP, TcTensorP, TcTensorP, c_f, c_f, c_i])
tc_abandono            = _firma("tc_abandono", TcTensorP, [TcTensorP, c_f, c_i])
tc_embedding           = _firma("tc_embedding", TcTensorP, [TcTensorP, TcTensorP])

# ── Pérdidas ───────────────────────────────────────────────────────────
tc_entropia_cruzada         = _firma("tc_entropia_cruzada", TcTensorP, [TcTensorP, TcTensorP])
tc_error_cuadratico_medio   = _firma("tc_error_cuadratico_medio", TcTensorP, [TcTensorP, TcTensorP])
tc_entropia_cruzada_binaria = _firma("tc_entropia_cruzada_binaria", TcTensorP, [TcTensorP, TcTensorP])

# ── Autograd ───────────────────────────────────────────────────────────
tc_retropropagar = _firma("tc_retropropagar", None, [TcTensorP])


# ── Penrose ────────────────────────────────────────────────────────────
tc_crear_teselacion         = _firma("tc_crear_teselacion", TcTeselacionPenroseP, [c_i, c_f])
tc_liberar_teselacion       = _firma("tc_liberar_teselacion", None, [TcTeselacionPenroseP])
tc_crear_kernel_penrose     = _firma("tc_crear_kernel_penrose", TcKernelPenroseP, [c_i, c_i, c_i, c_f])
tc_liberar_kernel_penrose   = _firma("tc_liberar_kernel_penrose", None, [TcKernelPenroseP])
tc_conv_penrose             = _firma("tc_conv_penrose", TcTensorP, [TcTensorP, TcKernelPenroseP, c_i, c_i])
tc_conv_penrose_grafo       = _firma("tc_conv_penrose_grafo", TcTensorP,
                                    [TcTensorP, TcTensorP, TcTensorP, TcTeselacionPenroseP])
tc_imagen_a_teselacion      = _firma("tc_imagen_a_teselacion", TcTensorP, [TcTensorP, TcTeselacionPenroseP])
tc_teselacion_a_imagen      = _firma("tc_teselacion_a_imagen", TcTensorP,
                                    [TcTensorP, TcTeselacionPenroseP, c_i, c_i])
tc_exportar_svg_teselacion  = _firma("tc_exportar_svg_teselacion", c_i,
                                    [TcTeselacionPenroseP, ctypes.c_char_p])

# ── Modelo (.ax) ───────────────────────────────────────────────────────
tc_guardar               = _firma("tc_guardar", c_i, [TcModeloP, ctypes.c_char_p])
tc_cargar                = _firma("tc_cargar", TcModeloP, [ctypes.c_char_p])
tc_liberar_modelo        = _firma("tc_liberar_modelo", None, [TcModeloP])
tc_liberar_modelo_estructura = _firma("tc_liberar_modelo_estructura", None, [TcModeloP])
tc_modelo_nuevo          = _firma("tc_modelo_nuevo", TcModeloP, [ctypes.c_char_p])
tc_modelo_agregar_tensor = _firma("tc_modelo_agregar_tensor", c_i, [TcModeloP, ctypes.c_char_p, TcTensorP])
tc_modelo_obtener_tensor = _firma("tc_modelo_obtener_tensor", TcTensorP, [TcModeloP, ctypes.c_char_p])

# ── Optimizadores ──────────────────────────────────────────────────────
tc_sgd    = _firma("tc_sgd", TcOptimizadorP, [ctypes.POINTER(TcTensorP), c_i, c_f, c_f])
tc_adam   = _firma("tc_adam", TcOptimizadorP, [ctypes.POINTER(TcTensorP), c_i, c_f, c_f, c_f, c_f])
tc_adamw  = _firma("tc_adamw", TcOptimizadorP, [ctypes.POINTER(TcTensorP), c_i, c_f, c_f, c_f, c_f, c_f])
tc_paso_optimizador           = _firma("tc_paso_optimizador", None, [TcOptimizadorP])
tc_cero_gradientes_optimizador = _firma("tc_cero_gradientes_optimizador", None, [TcOptimizadorP])
tc_liberar_optimizador        = _firma("tc_liberar_optimizador", None, [TcOptimizadorP])
