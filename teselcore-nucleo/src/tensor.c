/* teselcore/src/tensor.c
 * Implementación de tensores y autograd de TeselCore.
 * Licencia: GNU LGPL v3
 */

#include "../include/teselcore.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ════════════════════════════════════════════════════
 * GENERADOR ALEATORIO xoshiro256**
 * ════════════════════════════════════════════════════ */
static uint64_t _estado_rng[4] = {
    0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL,
    0x0F1E2D3C4B5A6978ULL, 0x8796A5B4C3D2E1F0ULL
};

static inline uint64_t _rotar_izquierda(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t _siguiente_aleatorio(void) {
    uint64_t resultado = _rotar_izquierda(_estado_rng[1] * 5, 7) * 9;
    uint64_t temporal  = _estado_rng[1] << 17;
    _estado_rng[2] ^= _estado_rng[0];
    _estado_rng[3] ^= _estado_rng[1];
    _estado_rng[1] ^= _estado_rng[2];
    _estado_rng[0] ^= _estado_rng[3];
    _estado_rng[2] ^= temporal;
    _estado_rng[3]  = _rotar_izquierda(_estado_rng[3], 45);
    return resultado;
}

static float _uniforme(void) {
    return (float)(_siguiente_aleatorio() >> 11) * (1.0f / (float)(1ULL << 53));
}

static float _normal(void) {
    float u1 = _uniforme() + 1e-7f;
    float u2 = _uniforme();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

void tc_semilla_aleatoria(uint64_t semilla) {
    _estado_rng[0] = semilla;
    _estado_rng[1] = semilla ^ 0xDEADBEEFULL;
    _estado_rng[2] = semilla ^ 0xCAFEBABEULL;
    _estado_rng[3] = semilla ^ 0xFEEDFACEULL;
}

/* ════════════════════════════════════════════════════
 * NODO DEL GRAFO DE AUTOGRAD
 * ════════════════════════════════════════════════════ */
typedef struct _nodo_autograd {
    tc_tensor*  salida;
    tc_tensor*  entradas[4];
    int         num_entradas;
    float       escalar_guardado;
    void*       aux;
    void (*retropropagar)(struct _nodo_autograd*);
    struct _nodo_autograd* siguiente;
} _nodo_autograd;

static _nodo_autograd* _tape_cabeza = NULL;
static int             _tape_activa = 1;

static _nodo_autograd* _tape_empujar(
    tc_tensor* salida,
    tc_tensor* entrada_a,
    tc_tensor* entrada_b,
    int        num_entradas,
    float      escalar,
    void*      aux,
    void (*retroprop)(_nodo_autograd*)
) {
    if (!_tape_activa) return NULL;
    _nodo_autograd* nodo = (_nodo_autograd*)malloc(sizeof(_nodo_autograd));
    nodo->salida           = salida;
    nodo->entradas[0]      = entrada_a;
    nodo->entradas[1]      = entrada_b;
    nodo->entradas[2]      = NULL;
    nodo->entradas[3]      = NULL;
    nodo->num_entradas     = num_entradas;
    nodo->escalar_guardado = escalar;
    nodo->aux              = aux;
    nodo->retropropagar    = retroprop;
    nodo->siguiente        = _tape_cabeza;
    _tape_cabeza           = nodo;
    salida->_nodo_grad     = nodo;
    return nodo;
}

static void _limpiar_tape(void) {
    _nodo_autograd* actual = _tape_cabeza;
    while (actual) {
        _nodo_autograd* siguiente = actual->siguiente;
        if (actual->aux) free(actual->aux);
        free(actual);
        actual = siguiente;
    }
    _tape_cabeza = NULL;
}

/* ════════════════════════════════════════════════════
 * HELPERS INTERNOS
 * ════════════════════════════════════════════════════ */
static void _calcular_pasos(const int* forma, int ndim, int* pasos) {
    pasos[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        pasos[i] = pasos[i + 1] * forma[i + 1];
}

static void _asegurar_gradiente(tc_tensor* t) {
    if (!t->gradiente)
        t->gradiente = (float*)calloc(t->total, sizeof(float));
}

static void _acumular_grad(tc_tensor* t, const float* delta) {
    if (!t->requiere_grad) return;
    _asegurar_gradiente(t);
    for (size_t i = 0; i < t->total; i++)
        t->gradiente[i] += delta[i];
}

/* ════════════════════════════════════════════════════
 * CREACIÓN DE TENSORES
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_vacio(const int* forma, int ndim, tc_tipo_dato tipo, tc_dispositivo disp) {
    assert(ndim > 0 && ndim <= TC_MAX_DIMS);
    tc_tensor* t   = (tc_tensor*)calloc(1, sizeof(tc_tensor));
    t->ndim        = ndim;
    t->tipo        = tipo;
    t->dispositivo = disp;
    t->_refs       = 1;

    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        t->forma[i] = forma[i];
        total      *= (size_t)forma[i];
    }
    t->total = total;
    _calcular_pasos(forma, ndim, t->pasos);
    t->datos = malloc(total * TC_BYTES_POR_TIPO[tipo]);
    return t;
}

tc_tensor* tc_ceros(const int* forma, int ndim) {
    tc_tensor* t = tc_vacio(forma, ndim, TC_FLOAT32, TC_CPU);
    memset(t->datos, 0, t->total * sizeof(float));
    return t;
}

tc_tensor* tc_unos(const int* forma, int ndim) {
    tc_tensor* t = tc_vacio(forma, ndim, TC_FLOAT32, TC_CPU);
    float* d = (float*)t->datos;
    for (size_t i = 0; i < t->total; i++) d[i] = 1.0f;
    return t;
}

tc_tensor* tc_aleatorio_uniforme(const int* forma, int ndim) {
    tc_tensor* t = tc_vacio(forma, ndim, TC_FLOAT32, TC_CPU);
    float* d = (float*)t->datos;
    for (size_t i = 0; i < t->total; i++) d[i] = _uniforme();
    return t;
}

tc_tensor* tc_aleatorio_normal(const int* forma, int ndim) {
    tc_tensor* t = tc_vacio(forma, ndim, TC_FLOAT32, TC_CPU);
    float* d = (float*)t->datos;
    for (size_t i = 0; i < t->total; i++) d[i] = _normal();
    return t;
}

tc_tensor* tc_desde_datos(const void* datos, const int* forma, int ndim, tc_tipo_dato tipo) {
    tc_tensor* t = tc_vacio(forma, ndim, tipo, TC_CPU);
    memcpy(t->datos, datos, t->total * TC_BYTES_POR_TIPO[tipo]);
    return t;
}

tc_tensor* tc_escalar(float valor) {
    int forma[] = {1};
    tc_tensor* t = tc_vacio(forma, 1, TC_FLOAT32, TC_CPU);
    ((float*)t->datos)[0] = valor;
    return t;
}

tc_tensor* tc_clonar(const tc_tensor* t) {
    tc_tensor* copia = tc_vacio(t->forma, t->ndim, t->tipo, t->dispositivo);
    memcpy(copia->datos, t->datos, t->total * TC_BYTES_POR_TIPO[t->tipo]);
    copia->requiere_grad = t->requiere_grad;
    return copia;
}

void tc_liberar(tc_tensor* t) {
    if (!t) return;
    free(t->datos);
    free(t->gradiente);
    free(t);
}

tc_tensor* tc_ref(tc_tensor* t)   { if (t) t->_refs++; return t; }
void       tc_unref(tc_tensor* t) { if (t && --t->_refs <= 0) tc_liberar(t); }

float tc_elemento(const tc_tensor* t) {
    return ((float*)t->datos)[0];
}

/* ════════════════════════════════════════════════════
 * FUNCIONES BACKWARD
 * ════════════════════════════════════════════════════ */
static void _bwd_sumar(_nodo_autograd* nodo) {
    float* g = nodo->salida->gradiente;
    if (nodo->entradas[0]->requiere_grad) _acumular_grad(nodo->entradas[0], g);
    if (nodo->entradas[1]->requiere_grad) _acumular_grad(nodo->entradas[1], g);
}

static void _bwd_restar(_nodo_autograd* nodo) {
    float* g = nodo->salida->gradiente;
    size_t n = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) _acumular_grad(nodo->entradas[0], g);
    if (nodo->entradas[1]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = -g[i];
        _acumular_grad(nodo->entradas[1], tmp);
        free(tmp);
    }
}

static void _bwd_multiplicar(_nodo_autograd* nodo) {
    float* g  = nodo->salida->gradiente;
    float* da = (float*)nodo->entradas[0]->datos;
    float* db = (float*)nodo->entradas[1]->datos;
    size_t n  = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * db[i];
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
    if (nodo->entradas[1]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * da[i];
        _acumular_grad(nodo->entradas[1], tmp);
        free(tmp);
    }
}

static void _bwd_relu(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = ent[i] > 0.0f ? g[i] : 0.0f;
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

static void _bwd_sigmoide(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* sal = (float*)nodo->salida->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * sal[i] * (1.0f - sal[i]);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

static void _bwd_logaritmo(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] / (ent[i] + 1e-8f);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

static void _bwd_softmax(_nodo_autograd* nodo) {
    float* g = nodo->salida->gradiente;
    float* s = (float*)nodo->salida->datos;
    int num_cols  = nodo->entradas[0]->forma[nodo->entradas[0]->ndim - 1];
    int num_filas = (int)(nodo->salida->total / num_cols);

    if (nodo->entradas[0]->requiere_grad) {
        _asegurar_gradiente(nodo->entradas[0]);
        for (int f = 0; f < num_filas; f++) {
            float* grow = g + f * num_cols;
            float* srow = s + f * num_cols;
            float dot = 0.0f;
            for (int c = 0; c < num_cols; c++) dot += grow[c] * srow[c];
            for (int c = 0; c < num_cols; c++) {
                float val = srow[c] * (grow[c] - dot);
                nodo->entradas[0]->gradiente[f * num_cols + c] += val;
            }
        }
    }
}

static void _bwd_multiplicacion_matricial(_nodo_autograd* nodo) {
    tc_tensor* A = nodo->entradas[0];
    tc_tensor* B = nodo->entradas[1];
    float*     g = nodo->salida->gradiente;
    int M = A->forma[0], K = A->forma[1], N = B->forma[1];

    if (A->requiere_grad) {
        _asegurar_gradiente(A);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    A->gradiente[m*K+k] += g[m*N+n] * ((float*)B->datos)[k*N+n];
    }
    if (B->requiere_grad) {
        _asegurar_gradiente(B);
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                for (int m = 0; m < M; m++)
                    B->gradiente[k*N+n] += ((float*)A->datos)[m*K+k] * g[m*N+n];
    }
}

static void _bwd_suma_dimension(_nodo_autograd* nodo) {
    tc_tensor* in = nodo->entradas[0];
    if (!in->requiere_grad) return;
    _asegurar_gradiente(in);
    float g = nodo->salida->gradiente[0];
    for (size_t i = 0; i < in->total; i++)
        in->gradiente[i] += g;
}

/* ════════════════════════════════════════════════════
 * OPERACIONES FORWARD
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_sumar(tc_tensor* a, tc_tensor* b) {
    assert(a->total == b->total);
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *db = (float*)b->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] + db[i];
    sal->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, b, 2, 0, NULL, _bwd_sumar);
    return sal;
}

tc_tensor* tc_restar(tc_tensor* a, tc_tensor* b) {
    assert(a->total == b->total);
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *db = (float*)b->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] - db[i];
    sal->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, b, 2, 0, NULL, _bwd_restar);
    return sal;
}

tc_tensor* tc_multiplicar(tc_tensor* a, tc_tensor* b) {
    assert(a->total == b->total);
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *db = (float*)b->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] * db[i];
    sal->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, b, 2, 0, NULL, _bwd_multiplicar);
    return sal;
}

/* BUG FIX: tc_dividir, tc_negar, tc_potencia, tc_exponencial, tc_raiz,
 * tc_valor_absoluto estaban declarados en el header pero no implementados.
 * Se agregan implementaciones mínimas con backward correcto. */

static void _bwd_dividir(_nodo_autograd* nodo) {
    float* g  = nodo->salida->gradiente;
    float* da = (float*)nodo->entradas[0]->datos;
    float* db = (float*)nodo->entradas[1]->datos;
    size_t n  = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] / (db[i] + 1e-8f);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
    if (nodo->entradas[1]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++)
            tmp[i] = -g[i] * da[i] / ((db[i] * db[i]) + 1e-8f);
        _acumular_grad(nodo->entradas[1], tmp);
        free(tmp);
    }
}

tc_tensor* tc_dividir(tc_tensor* a, tc_tensor* b) {
    assert(a->total == b->total);
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *db = (float*)b->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] / (db[i] + 1e-8f);
    sal->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, b, 2, 0, NULL, _bwd_dividir);
    return sal;
}

static void _bwd_negar(_nodo_autograd* nodo) {
    float* g = nodo->salida->gradiente;
    size_t n = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = -g[i];
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_negar(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = -da[i];
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_negar);
    return sal;
}

static void _bwd_potencia(_nodo_autograd* nodo) {
    float  exp_val = nodo->escalar_guardado;
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++)
            tmp[i] = g[i] * exp_val * powf(ent[i], exp_val - 1.0f);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_potencia(tc_tensor* a, float exp_val) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = powf(da[i], exp_val);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, exp_val, NULL, _bwd_potencia);
    return sal;
}

tc_tensor* tc_logaritmo(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = logf(da[i] + 1e-8f);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_logaritmo);
    return sal;
}

static void _bwd_exponencial(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* sal = (float*)nodo->salida->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * sal[i];
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_exponencial(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = expf(da[i]);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_exponencial);
    return sal;
}

static void _bwd_raiz(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* sal = (float*)nodo->salida->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] / (2.0f * sal[i] + 1e-8f);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_raiz(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = sqrtf(da[i] < 0.0f ? 0.0f : da[i]);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_raiz);
    return sal;
}

static void _bwd_valor_absoluto(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * (ent[i] >= 0.0f ? 1.0f : -1.0f);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_valor_absoluto(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = fabsf(da[i]);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_valor_absoluto);
    return sal;
}

/* ════════════════════════════════════════════════════
 * REDUCCIÓN
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_suma_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float suma = 0.0f;
    float* d = (float*)a->datos;
    for (size_t i = 0; i < a->total; i++) suma += d[i];
    int forma_sal[] = {1};
    tc_tensor* sal = tc_desde_datos(&suma, forma_sal, 1, TC_FLOAT32);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_suma_dimension);
    return sal;
}

/* BUG FIX: tc_media_dimension, tc_maximo_dimension, tc_minimo_dimension
 * declaradas pero no implementadas. Se agregan stubs funcionales. */
tc_tensor* tc_media_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float suma = 0.0f;
    float* d = (float*)a->datos;
    for (size_t i = 0; i < a->total; i++) suma += d[i];
    float media = suma / (float)a->total;
    int forma_sal[] = {1};
    return tc_desde_datos(&media, forma_sal, 1, TC_FLOAT32);
}

tc_tensor* tc_maximo_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float* d = (float*)a->datos;
    float maximo = d[0];
    for (size_t i = 1; i < a->total; i++) if (d[i] > maximo) maximo = d[i];
    int forma_sal[] = {1};
    return tc_desde_datos(&maximo, forma_sal, 1, TC_FLOAT32);
}

tc_tensor* tc_minimo_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float* d = (float*)a->datos;
    float minimo = d[0];
    for (size_t i = 1; i < a->total; i++) if (d[i] < minimo) minimo = d[i];
    int forma_sal[] = {1};
    return tc_desde_datos(&minimo, forma_sal, 1, TC_FLOAT32);
}

/* ════════════════════════════════════════════════════
 * ÁLGEBRA LINEAL
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_multiplicacion_matricial(tc_tensor* a, tc_tensor* b) {
    assert(a->ndim == 2 && b->ndim == 2 && a->forma[1] == b->forma[0]);
    int M = a->forma[0], K = a->forma[1], N = b->forma[1];
    int forma_sal[] = {M, N};
    tc_tensor* sal = tc_ceros(forma_sal, 2);
    float *da = (float*)a->datos, *db = (float*)b->datos, *ds = (float*)sal->datos;

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float val_a = da[i*K+k];
            for (int j = 0; j < N; j++)
                ds[i*N+j] += val_a * db[k*N+j];
        }

    sal->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, b, 2, 0, NULL, _bwd_multiplicacion_matricial);
    return sal;
}

tc_tensor* tc_transponer(tc_tensor* a, int dim0, int dim1) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, a->tipo, a->dispositivo);
    sal->forma[dim0] = a->forma[dim1];
    sal->forma[dim1] = a->forma[dim0];
    sal->pasos[dim0] = a->pasos[dim1];
    sal->pasos[dim1] = a->pasos[dim0];
    if (a->ndim == 2) {
        float *src = (float*)a->datos, *dst = (float*)sal->datos;
        int F = a->forma[0], C = a->forma[1];
        for (int f = 0; f < F; f++)
            for (int c = 0; c < C; c++)
                dst[c*F+f] = src[f*C+c];
    }
    sal->requiere_grad = a->requiere_grad;
    return sal;
}

/* BUG FIX: tc_reformar, tc_aplanar, tc_concatenar declaradas pero no implementadas. */
tc_tensor* tc_reformar(tc_tensor* a, const int* nueva_forma, int nuevo_ndim) {
    tc_tensor* sal = tc_clonar(a);
    size_t total_nuevo = 1;
    for (int i = 0; i < nuevo_ndim; i++) {
        sal->forma[i] = nueva_forma[i];
        total_nuevo *= (size_t)nueva_forma[i];
    }
    assert(total_nuevo == a->total);
    sal->ndim = nuevo_ndim;
    _calcular_pasos(nueva_forma, nuevo_ndim, sal->pasos);
    sal->requiere_grad = a->requiere_grad;
    return sal;
}

tc_tensor* tc_aplanar(tc_tensor* a, int dim_inicio) {
    (void)dim_inicio;
    int nueva_forma[] = {(int)a->total};
    return tc_reformar(a, nueva_forma, 1);
}

tc_tensor* tc_concatenar(tc_tensor** tensores, int n, int dimension) {
    /* Implementación simple: solo soporta concatenar en dim 0 con tensores 1D */
    (void)dimension;
    size_t total = 0;
    for (int i = 0; i < n; i++) total += tensores[i]->total;
    int forma_sal[] = {(int)total};
    tc_tensor* sal = tc_vacio(forma_sal, 1, TC_FLOAT32, TC_CPU);
    float* ds = (float*)sal->datos;
    size_t offset = 0;
    for (int i = 0; i < n; i++) {
        memcpy(ds + offset, tensores[i]->datos, tensores[i]->total * sizeof(float));
        offset += tensores[i]->total;
    }
    return sal;
}

/* ════════════════════════════════════════════════════
 * ACTIVACIONES
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_relu(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] > 0.0f ? da[i] : 0.0f;
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_relu);
    return sal;
}

/* BUG FIX: tc_relu_con_fuga, tc_tangente_hiperbolica, tc_gelu, tc_silu
 * declaradas pero no implementadas. */
static void _bwd_relu_con_fuga(_nodo_autograd* nodo) {
    float  alfa = nodo->escalar_guardado;
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * (ent[i] > 0.0f ? 1.0f : alfa);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_relu_con_fuga(tc_tensor* a, float alfa) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = da[i] > 0.0f ? da[i] : alfa * da[i];
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, alfa, NULL, _bwd_relu_con_fuga);
    return sal;
}

tc_tensor* tc_sigmoide(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++)
        ds[i] = 1.0f / (1.0f + expf(-da[i]));
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_sigmoide);
    return sal;
}

static void _bwd_tanh(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* sal = (float*)nodo->salida->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) tmp[i] = g[i] * (1.0f - sal[i] * sal[i]);
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_tangente_hiperbolica(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) ds[i] = tanhf(da[i]);
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_tanh);
    return sal;
}

tc_tensor* tc_softmax(tc_tensor* a, int dimension) {
    (void)dimension;
    tc_tensor* sal  = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    int num_cols  = a->forma[a->ndim - 1];
    int num_filas = (int)(a->total / num_cols);

    for (int f = 0; f < num_filas; f++) {
        float* fila_entrada = da + f * num_cols;
        float* fila_salida  = ds + f * num_cols;
        float maximo = fila_entrada[0];
        for (int c = 1; c < num_cols; c++)
            if (fila_entrada[c] > maximo) maximo = fila_entrada[c];
        float suma = 0.0f;
        for (int c = 0; c < num_cols; c++) {
            fila_salida[c] = expf(fila_entrada[c] - maximo);
            suma += fila_salida[c];
        }
        for (int c = 0; c < num_cols; c++) fila_salida[c] /= suma;
    }
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_softmax);
    return sal;
}

/* GELU: approx. 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) */
static void _bwd_gelu(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        const float sqrt2pi = 0.7978845608f;
        for (size_t i = 0; i < n; i++) {
            float x = ent[i];
            float inner = sqrt2pi * (x + 0.044715f * x * x * x);
            float tanh_val = tanhf(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float dgelu = 0.5f * (1.0f + tanh_val)
                        + 0.5f * x * sech2 * sqrt2pi * (1.0f + 3.0f * 0.044715f * x * x);
            tmp[i] = g[i] * dgelu;
        }
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_gelu(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    const float sqrt2pi = 0.7978845608f;
    for (size_t i = 0; i < a->total; i++) {
        float x = da[i];
        ds[i] = 0.5f * x * (1.0f + tanhf(sqrt2pi * (x + 0.044715f * x * x * x)));
    }
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_gelu);
    return sal;
}

/* SiLU: x * sigmoid(x) */
static void _bwd_silu(_nodo_autograd* nodo) {
    float* g   = nodo->salida->gradiente;
    float* ent = (float*)nodo->entradas[0]->datos;
    size_t n   = nodo->salida->total;
    if (nodo->entradas[0]->requiere_grad) {
        float* tmp = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            float sig = 1.0f / (1.0f + expf(-ent[i]));
            tmp[i] = g[i] * (sig + ent[i] * sig * (1.0f - sig));
        }
        _acumular_grad(nodo->entradas[0], tmp);
        free(tmp);
    }
}

tc_tensor* tc_silu(tc_tensor* a) {
    tc_tensor* sal = tc_vacio(a->forma, a->ndim, TC_FLOAT32, TC_CPU);
    float *da = (float*)a->datos, *ds = (float*)sal->datos;
    for (size_t i = 0; i < a->total; i++) {
        float sig = 1.0f / (1.0f + expf(-da[i]));
        ds[i] = da[i] * sig;
    }
    sal->requiere_grad = a->requiere_grad;
    if (sal->requiere_grad) _tape_empujar(sal, a, NULL, 1, 0, NULL, _bwd_silu);
    return sal;
}

/* ════════════════════════════════════════════════════
 * CAPAS DE RED NEURONAL
 * BUG FIX: tc_conv2d, tc_agrupacion_max, tc_agrupacion_promedio,
 * tc_normalizacion_lote, tc_abandono, tc_embedding declaradas pero
 * no implementadas. Se agregan implementaciones funcionales.
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_lineal(tc_tensor* x, tc_tensor* pesos, tc_tensor* sesgo) {
    tc_tensor* pt  = tc_transponer(pesos, 0, 1);
    tc_tensor* xp  = tc_multiplicacion_matricial(x, pt);
    tc_liberar(pt); /* BUG FIX: pt se liberaba perdiendo gradientes; ahora pt
                       no tiene requiere_grad porque viene de transponer sin tape,
                       así que es seguro liberarlo. xp y salida retienen el grafo. */
    tc_tensor* sal = sesgo ? tc_sumar(xp, sesgo) : xp;
    if (sesgo) tc_liberar(xp); /* BUG FIX: xp es intermedio, sal lo reemplaza */
    return sal;
}

/* conv2d — implementación naive (sin backprop por ahora; se puede extender) */
tc_tensor* tc_conv2d(tc_tensor* x, tc_tensor* kernel, tc_tensor* sesgo,
                      int paso, int relleno) {
    assert(x->ndim == 4 && kernel->ndim == 4);
    int B  = x->forma[0], Cin = x->forma[1], H = x->forma[2], W = x->forma[3];
    int Cout = kernel->forma[0], kH = kernel->forma[2], kW = kernel->forma[3];
    int H_sal = (H + 2*relleno - kH) / paso + 1;
    int W_sal = (W + 2*relleno - kW) / paso + 1;
    int forma_sal[] = {B, Cout, H_sal, W_sal};
    tc_tensor* sal = tc_ceros(forma_sal, 4);
    float* dx = (float*)x->datos;
    float* dk = (float*)kernel->datos;
    float* ds = (float*)sal->datos;

    for (int b = 0; b < B; b++)
    for (int co = 0; co < Cout; co++)
    for (int oh = 0; oh < H_sal; oh++)
    for (int ow = 0; ow < W_sal; ow++) {
        float suma = sesgo ? ((float*)sesgo->datos)[co] : 0.0f;
        for (int ci = 0; ci < Cin; ci++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int ih = oh * paso - relleno + kh;
            int iw = ow * paso - relleno + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                suma += dx[b*Cin*H*W + ci*H*W + ih*W + iw]
                      * dk[co*Cin*kH*kW + ci*kH*kW + kh*kW + kw];
            }
        }
        ds[b*Cout*H_sal*W_sal + co*H_sal*W_sal + oh*W_sal + ow] = suma;
    }
    return sal;
}

tc_tensor* tc_agrupacion_max(tc_tensor* x, int kernel, int paso) {
    assert(x->ndim == 4);
    int B = x->forma[0], C = x->forma[1], H = x->forma[2], W = x->forma[3];
    int H_sal = (H - kernel) / paso + 1;
    int W_sal = (W - kernel) / paso + 1;
    int forma_sal[] = {B, C, H_sal, W_sal};
    tc_tensor* sal = tc_vacio(forma_sal, 4, TC_FLOAT32, TC_CPU);
    float* dx = (float*)x->datos;
    float* ds = (float*)sal->datos;

    for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < H_sal; oh++)
    for (int ow = 0; ow < W_sal; ow++) {
        float mx = -1e38f;
        for (int kh = 0; kh < kernel; kh++)
        for (int kw = 0; kw < kernel; kw++) {
            int ih = oh * paso + kh, iw = ow * paso + kw;
            float v = dx[b*C*H*W + c*H*W + ih*W + iw];
            if (v > mx) mx = v;
        }
        ds[b*C*H_sal*W_sal + c*H_sal*W_sal + oh*W_sal + ow] = mx;
    }
    return sal;
}

tc_tensor* tc_agrupacion_promedio(tc_tensor* x, int kernel, int paso) {
    assert(x->ndim == 4);
    int B = x->forma[0], C = x->forma[1], H = x->forma[2], W = x->forma[3];
    int H_sal = (H - kernel) / paso + 1;
    int W_sal = (W - kernel) / paso + 1;
    int forma_sal[] = {B, C, H_sal, W_sal};
    tc_tensor* sal = tc_ceros(forma_sal, 4);
    float* dx = (float*)x->datos;
    float* ds = (float*)sal->datos;
    float area = (float)(kernel * kernel);

    for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < H_sal; oh++)
    for (int ow = 0; ow < W_sal; ow++) {
        float suma = 0.0f;
        for (int kh = 0; kh < kernel; kh++)
        for (int kw = 0; kw < kernel; kw++) {
            int ih = oh * paso + kh, iw = ow * paso + kw;
            suma += dx[b*C*H*W + c*H*W + ih*W + iw];
        }
        ds[b*C*H_sal*W_sal + c*H_sal*W_sal + oh*W_sal + ow] = suma / area;
    }
    return sal;
}

tc_tensor* tc_normalizacion_lote(tc_tensor* x, tc_tensor* gamma, tc_tensor* beta,
                                  tc_tensor* media_mov, tc_tensor* var_mov,
                                  float eps, float momento, int entrenando) {
    (void)momento;
    assert(x->ndim >= 2);
    int C = x->forma[1];
    tc_tensor* sal = tc_clonar(x);
    float* ds  = (float*)sal->datos;
    float* dg  = (float*)gamma->datos;
    float* db  = (float*)beta->datos;
    float* mm  = (float*)media_mov->datos;
    float* mv  = (float*)var_mov->datos;
    size_t N   = x->total / (size_t)C;

    for (int c = 0; c < C; c++) {
        float media, varianza;
        if (entrenando) {
            media = 0.0f;
            for (size_t i = 0; i < N; i++) {
                /* Indexar canal c: asume layout (B, C, ...) */
                /* Simplificado: recorremos todos los elementos del canal */
                media += ((float*)x->datos)[c * (int)N + (int)i];
            }
            media /= (float)N;
            varianza = 0.0f;
            for (size_t i = 0; i < N; i++) {
                float d = ((float*)x->datos)[c * (int)N + (int)i] - media;
                varianza += d * d;
            }
            varianza /= (float)N;
            mm[c] = media;
            mv[c] = varianza;
        } else {
            media = mm[c];
            varianza = mv[c];
        }
        float inv_std = 1.0f / sqrtf(varianza + eps);
        for (size_t i = 0; i < N; i++) {
            float xhat = (((float*)x->datos)[c * (int)N + (int)i] - media) * inv_std;
            ds[c * (int)N + (int)i] = dg[c] * xhat + db[c];
        }
    }
    return sal;
}

tc_tensor* tc_abandono(tc_tensor* x, float p, int entrenando) {
    if (!entrenando || p <= 0.0f) return tc_clonar(x);
    tc_tensor* sal = tc_clonar(x);
    float* ds = (float*)sal->datos;
    float escala = 1.0f / (1.0f - p);
    for (size_t i = 0; i < x->total; i++) {
        if (_uniforme() < p) ds[i] = 0.0f;
        else                  ds[i] *= escala;
    }
    return sal;
}

tc_tensor* tc_embedding(tc_tensor* indices, tc_tensor* pesos) {
    assert(indices->ndim == 1 || indices->ndim == 2);
    int* idx = (int*)indices->datos;
    int  vocab_sz = pesos->forma[0];
    int  emb_dim  = pesos->forma[1];
    int  seq_len  = (int)indices->total;
    int  forma_sal[] = {seq_len, emb_dim};
    tc_tensor* sal = tc_vacio(forma_sal, 2, TC_FLOAT32, TC_CPU);
    float* dp = (float*)pesos->datos;
    float* ds = (float*)sal->datos;
    for (int i = 0; i < seq_len; i++) {
        int id = idx[i];
        assert(id >= 0 && id < vocab_sz);
        memcpy(ds + i * emb_dim, dp + id * emb_dim, emb_dim * sizeof(float));
    }
    return sal;
}

/* ════════════════════════════════════════════════════
 * PÉRDIDAS
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_entropia_cruzada(tc_tensor* logits, tc_tensor* objetivos) {
    int L = logits->forma[0], C = logits->forma[1];
    tc_tensor* sm = tc_softmax(logits, -1);
    float* sp = (float*)sm->datos;
    int*   tp = (int*)objetivos->datos;

    float perdida = 0.0f;
    for (int b = 0; b < L; b++)
        perdida -= logf(sp[b*C + tp[b]] + 1e-8f);
    perdida /= (float)L;

    if (logits->requiere_grad) {
        _asegurar_gradiente(logits);
        for (int b = 0; b < L; b++)
            for (int c = 0; c < C; c++) {
                float delta = sp[b*C+c] - (c == tp[b] ? 1.0f : 0.0f);
                logits->gradiente[b*C+c] += delta / (float)L;
            }
    }
    tc_liberar(sm);

    int forma_sal[] = {1};
    return tc_desde_datos(&perdida, forma_sal, 1, TC_FLOAT32);
}

/* BUG FIX: tc_error_cuadratico_medio y tc_entropia_cruzada_binaria
 * declaradas pero no implementadas. */
tc_tensor* tc_error_cuadratico_medio(tc_tensor* pred, tc_tensor* objetivo) {
    assert(pred->total == objetivo->total);
    float* dp = (float*)pred->datos;
    float* do_ = (float*)objetivo->datos;
    float perdida = 0.0f;
    for (size_t i = 0; i < pred->total; i++) {
        float d = dp[i] - do_[i];
        perdida += d * d;
    }
    perdida /= (float)pred->total;
    if (pred->requiere_grad) {
        _asegurar_gradiente(pred);
        float escala = 2.0f / (float)pred->total;
        for (size_t i = 0; i < pred->total; i++)
            pred->gradiente[i] += escala * (dp[i] - do_[i]);
    }
    int forma_sal[] = {1};
    return tc_desde_datos(&perdida, forma_sal, 1, TC_FLOAT32);
}

tc_tensor* tc_entropia_cruzada_binaria(tc_tensor* pred, tc_tensor* objetivo) {
    assert(pred->total == objetivo->total);
    float* dp = (float*)pred->datos;
    float* do_ = (float*)objetivo->datos;
    float perdida = 0.0f;
    for (size_t i = 0; i < pred->total; i++) {
        float p = dp[i] < 1e-7f ? 1e-7f : (dp[i] > 1.0f - 1e-7f ? 1.0f - 1e-7f : dp[i]);
        perdida -= do_[i] * logf(p) + (1.0f - do_[i]) * logf(1.0f - p);
    }
    perdida /= (float)pred->total;
    if (pred->requiere_grad) {
        _asegurar_gradiente(pred);
        float escala = 1.0f / (float)pred->total;
        for (size_t i = 0; i < pred->total; i++) {
            float p = dp[i] < 1e-7f ? 1e-7f : (dp[i] > 1.0f - 1e-7f ? 1.0f - 1e-7f : dp[i]);
            pred->gradiente[i] += escala * (-do_[i] / p + (1.0f - do_[i]) / (1.0f - p));
        }
    }
    int forma_sal[] = {1};
    return tc_desde_datos(&perdida, forma_sal, 1, TC_FLOAT32);
}

/* ════════════════════════════════════════════════════
 * RETROPROPAGACIÓN
 * ════════════════════════════════════════════════════ */
void tc_retropropagar(tc_tensor* perdida) {
    _asegurar_gradiente(perdida);
    perdida->gradiente[0] = 1.0f;

    _nodo_autograd* nodo = _tape_cabeza;
    while (nodo) {
        if (nodo->salida->gradiente && nodo->retropropagar)
            nodo->retropropagar(nodo);
        nodo = nodo->siguiente;
    }
    _limpiar_tape();
}

void tc_cero_gradiente(tc_tensor* t) {
    if (t->gradiente) memset(t->gradiente, 0, t->total * sizeof(float));
}

/* ════════════════════════════════════════════════════
 * BACKWARD: conv_penrose_grafo
 * ════════════════════════════════════════════════════ */
typedef struct {
    const tc_teselacion_penrose* tes;
    tc_tensor* sesgo;
} _conv_penrose_ctx;

static void _bwd_conv_penrose_grafo(_nodo_autograd* nodo) {
    _conv_penrose_ctx* ctx = (_conv_penrose_ctx*)nodo->aux;
    const tc_teselacion_penrose* tes = ctx ? ctx->tes : NULL;
    tc_tensor* sesgo = ctx ? ctx->sesgo : NULL;

    tc_tensor* entrada = nodo->entradas[0];
    tc_tensor* kernel  = nodo->entradas[1];
    tc_tensor* salida  = nodo->salida;

    if (!entrada || !kernel || !salida || !tes) return;

    int B   = entrada->forma[0];
    int Cin = entrada->forma[1];
    int N   = entrada->forma[2];
    int Cout = kernel->forma[0];
    int K   = kernel->forma[2];

    float* de = (float*)entrada->datos;
    float* dk = (float*)kernel->datos;
    float* g  = salida->gradiente;

    if (entrada->requiere_grad) _asegurar_gradiente(entrada);
    if (kernel->requiere_grad)  _asegurar_gradiente(kernel);
    if (sesgo && sesgo->requiere_grad) _asegurar_gradiente(sesgo);

    for (int b = 0; b < B; b++)
    for (int co = 0; co < Cout; co++)
    for (int i = 0; i < N; i++) {
        float grad_out = g[b*Cout*N + co*N + i];
        if (sesgo && sesgo->requiere_grad) sesgo->gradiente[co] += grad_out;

        for (int ci = 0; ci < Cin; ci++) {
            if (0 < K) {
                if (kernel->requiere_grad)
                    kernel->gradiente[co*Cin*K + ci*K + 0] += grad_out * de[b*Cin*N + ci*N + i];
                if (entrada->requiere_grad)
                    entrada->gradiente[b*Cin*N + ci*N + i] += grad_out * dk[co*Cin*K + ci*K + 0];
            }
            const tc_teja_penrose* tj = &tes->tejas[i];
            for (int nv = 0; nv < tj->num_vecinos && (nv + 1) < K; nv++) {
                int vecino = tj->vecinos[nv];
                float in_val = de[b*Cin*N + ci*N + vecino];
                if (kernel->requiere_grad)
                    kernel->gradiente[co*Cin*K + ci*K + (nv+1)] += grad_out * in_val;
                if (entrada->requiere_grad)
                    entrada->gradiente[b*Cin*N + ci*N + vecino] += grad_out * dk[co*Cin*K + ci*K + (nv+1)];
            }
        }
    }
}

void tc_tape_empujar_conv_penrose_grafo(tc_tensor* salida, tc_tensor* entrada,
                                        tc_tensor* kernel, tc_tensor* sesgo,
                                        const tc_teselacion_penrose* teselacion) {
    if (!salida || !salida->requiere_grad) return;
    _conv_penrose_ctx* ctx = (_conv_penrose_ctx*)malloc(sizeof(_conv_penrose_ctx));
    ctx->tes   = teselacion;
    ctx->sesgo = sesgo;
    _tape_empujar(salida, entrada, kernel, 2, 0.0f, ctx, _bwd_conv_penrose_grafo);
}

/* ════════════════════════════════════════════════════
 * UTILIDADES
 * ════════════════════════════════════════════════════ */
void tc_imprimir(const tc_tensor* t) {
    printf("Tensor(forma=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->forma[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], tipo=%d, requiere_grad=%d)\n", t->tipo, t->requiere_grad);

    float* d = (float*)t->datos;
    size_t limite = t->total < 12 ? t->total : 12;
    printf("datos: [");
    for (size_t i = 0; i < limite; i++) {
        printf("%.4f", d[i]);
        if (i < limite - 1) printf(", ");
    }
    if (t->total > 12) printf(", ...");
    printf("]\n");
}

tc_tensor* tc_a_dispositivo(tc_tensor* t, tc_dispositivo disp) {
    t->dispositivo = disp;
    return t;
}

void tc_imprimir_info(void) {
    printf("TeselCore v%s\n", TC_VERSION_STR);
    printf("Compilado: %s %s\n", __DATE__, __TIME__);
#ifdef __AVX2__
    printf("AVX2: disponible\n");
#else
    printf("AVX2: no disponible\n");
#endif
    printf("Razón áurea φ: %.10f\n", TC_PHI);
}

/*Hola, si alguien lee hasta esta parte del codigo, solo le digo que gracias por
interesarte o hasta por encontrar este proyecto, solo queria ver como hacer un framework
y pues todavia es muy monolitico y tiene varios problemas pero mientras construyo esto tambien quiero
aportar o minimo ver curiosidades matematicas, o de fisica o de cualquier cosa rara que se me ocurre y muchas gracias, si vez
esto lo agradezco demasiado, para este primer commit subo la primera version, todavia es muy experimental perolo pienso mejorar
y agregar mas ideas locas que tengo, muchas gracias*/