/* teselcore/include/teselcore.h */
#ifndef TESELCORE_H
#define TESELCORE_H

#include <stdint.h>
#include <stddef.h>

#define TC_VERSION_MAYOR  0
#define TC_VERSION_MENOR  1
#define TC_VERSION_PARCHE 0
#define TC_VERSION_STR    "0.1.0"

#define TC_PHI 1.6180339887498948482f

typedef enum {
    TC_FLOAT32 = 0,
    TC_FLOAT64 = 1,
    TC_INT32   = 2,
    TC_INT8    = 3,
    TC_FLOAT16 = 4,
} tc_tipo_dato;

static const int TC_BYTES_POR_TIPO[] = {4, 8, 4, 1, 2};

typedef enum {
    TC_CPU   = 0,
    TC_CUDA  = 1,
    TC_METAL = 2,
    TC_WASM  = 3,
} tc_dispositivo;

#define TC_MAX_DIMS 8

typedef struct tc_tensor {
    void*          datos;
    float*         gradiente;
    int            forma[TC_MAX_DIMS];
    int            pasos[TC_MAX_DIMS];
    int            ndim;
    size_t         total;
    tc_tipo_dato   tipo;
    tc_dispositivo dispositivo;
    int            requiere_grad;
    int            _refs;
    void*          _nodo_grad;
} tc_tensor;

tc_tensor* tc_vacio               (const int* forma, int ndim, tc_tipo_dato tipo, tc_dispositivo disp);
tc_tensor* tc_ceros               (const int* forma, int ndim);
tc_tensor* tc_unos                (const int* forma, int ndim);
tc_tensor* tc_aleatorio_uniforme  (const int* forma, int ndim);
tc_tensor* tc_aleatorio_normal    (const int* forma, int ndim);
tc_tensor* tc_desde_datos         (const void* datos, const int* forma, int ndim, tc_tipo_dato tipo);
tc_tensor* tc_escalar             (float valor);
tc_tensor* tc_clonar              (const tc_tensor* t);

void       tc_liberar  (tc_tensor* t);
tc_tensor* tc_ref      (tc_tensor* t);
void       tc_unref    (tc_tensor* t);

tc_tensor* tc_sumar               (tc_tensor* a, tc_tensor* b);
tc_tensor* tc_restar              (tc_tensor* a, tc_tensor* b);
tc_tensor* tc_multiplicar         (tc_tensor* a, tc_tensor* b);
tc_tensor* tc_dividir             (tc_tensor* a, tc_tensor* b);
tc_tensor* tc_negar               (tc_tensor* a);
tc_tensor* tc_potencia            (tc_tensor* a, float exp);
tc_tensor* tc_logaritmo           (tc_tensor* a);
tc_tensor* tc_exponencial         (tc_tensor* a);
tc_tensor* tc_raiz                (tc_tensor* a);
tc_tensor* tc_valor_absoluto      (tc_tensor* a);

tc_tensor* tc_suma_dimension      (tc_tensor* a, int dimension, int mantener_dim);
tc_tensor* tc_media_dimension     (tc_tensor* a, int dimension, int mantener_dim);
tc_tensor* tc_maximo_dimension    (tc_tensor* a, int dimension, int mantener_dim);
tc_tensor* tc_minimo_dimension    (tc_tensor* a, int dimension, int mantener_dim);

tc_tensor* tc_multiplicacion_matricial (tc_tensor* a, tc_tensor* b);
tc_tensor* tc_transponer               (tc_tensor* a, int dim0, int dim1);
tc_tensor* tc_reformar                 (tc_tensor* a, const int* nueva_forma, int nuevo_ndim);
tc_tensor* tc_aplanar                  (tc_tensor* a, int dim_inicio);
tc_tensor* tc_concatenar               (tc_tensor** tensores, int n, int dimension);

tc_tensor* tc_relu                (tc_tensor* a);
tc_tensor* tc_relu_con_fuga       (tc_tensor* a, float alfa);
tc_tensor* tc_sigmoide            (tc_tensor* a);
tc_tensor* tc_tangente_hiperbolica(tc_tensor* a);
tc_tensor* tc_softmax             (tc_tensor* a, int dimension);
tc_tensor* tc_gelu                (tc_tensor* a);
tc_tensor* tc_silu                (tc_tensor* a);

tc_tensor* tc_lineal              (tc_tensor* x, tc_tensor* pesos, tc_tensor* sesgo);
tc_tensor* tc_conv2d              (tc_tensor* x, tc_tensor* kernel, tc_tensor* sesgo,
                                   int paso, int relleno);
tc_tensor* tc_agrupacion_max      (tc_tensor* x, int kernel, int paso);
tc_tensor* tc_agrupacion_promedio (tc_tensor* x, int kernel, int paso);
tc_tensor* tc_normalizacion_lote  (tc_tensor* x, tc_tensor* gamma, tc_tensor* beta,
                                   tc_tensor* media_mov, tc_tensor* var_mov,
                                   float eps, float momento, int entrenando);
tc_tensor* tc_abandono            (tc_tensor* x, float p, int entrenando);
tc_tensor* tc_embedding           (tc_tensor* indices, tc_tensor* pesos);

typedef enum {
    TC_PENROSE_KITE = 0,
    TC_PENROSE_DART = 1,
} tc_tipo_teja_penrose;

typedef struct {
    float              centro_x, centro_y;
    float              angulo;
    tc_tipo_teja_penrose tipo;
    int                vecinos[7];
    int                num_vecinos;
    float              puntos[4][2];
} tc_teja_penrose;

typedef struct {
    tc_teja_penrose* tejas;
    int              num_tejas;
    int              nivel;
    float            escala;
} tc_teselacion_penrose;

tc_teselacion_penrose* tc_crear_teselacion  (int nivel, float escala);
void                   tc_liberar_teselacion(tc_teselacion_penrose* t);

typedef struct {
    tc_teselacion_penrose* teselacion;
    tc_tensor*             pesos;
    tc_tensor*             sesgo;
    int                    canales_entrada;
    int                    canales_salida;
} tc_kernel_penrose;

tc_kernel_penrose* tc_crear_kernel_penrose  (int canales_ent, int canales_sal,
                                             int nivel_teselacion, float escala);
void               tc_liberar_kernel_penrose(tc_kernel_penrose* k);

tc_tensor* tc_conv_penrose(tc_tensor* entrada, tc_kernel_penrose* kernel,
                            int paso, int modo_relleno);
tc_tensor* tc_conv_penrose_grafo(tc_tensor* entrada,
                                  tc_tensor* kernel,
                                  tc_tensor* sesgo,
                                  const tc_teselacion_penrose* teselacion);
tc_tensor* tc_imagen_a_teselacion (tc_tensor* imagen,
                                    const tc_teselacion_penrose* teselacion);
tc_tensor* tc_teselacion_a_imagen (tc_tensor* caracteristicas,
                                    const tc_teselacion_penrose* teselacion,
                                    int alto, int ancho);
int tc_exportar_svg_teselacion(const tc_teselacion_penrose* t, const char* ruta);

tc_tensor* tc_entropia_cruzada         (tc_tensor* logits, tc_tensor* objetivos);
tc_tensor* tc_error_cuadratico_medio   (tc_tensor* pred,   tc_tensor* objetivo);
tc_tensor* tc_entropia_cruzada_binaria (tc_tensor* pred,   tc_tensor* objetivo);

void tc_retropropagar  (tc_tensor* perdida);
void tc_cero_gradiente (tc_tensor* t);
void tc_tape_empujar_conv_penrose_grafo(tc_tensor* salida, tc_tensor* entrada,
                                       tc_tensor* kernel, tc_tensor* sesgo,
                                       const tc_teselacion_penrose* teselacion);

typedef struct {
    char       nombre[256];
    tc_tensor* tensor;
} tc_tensor_nombrado;

typedef struct {
    tc_tensor_nombrado* tensores;
    int                 num_tensores;
    char*               metadatos;
    uint16_t            version_mayor;
    uint16_t            version_menor;
} tc_modelo;

int        tc_guardar               (const tc_modelo* modelo, const char* ruta);
tc_modelo* tc_cargar                (const char* ruta);
void       tc_liberar_modelo        (tc_modelo* modelo);
tc_modelo* tc_modelo_nuevo          (const char* metadatos_json);
int        tc_modelo_agregar_tensor (tc_modelo* modelo, const char* nombre, tc_tensor* t);
tc_tensor* tc_modelo_obtener_tensor (const tc_modelo* modelo, const char* nombre);

typedef struct tc_optimizador tc_optimizador;

tc_optimizador* tc_sgd   (tc_tensor** parametros, int n, float lr, float momento);
tc_optimizador* tc_adam  (tc_tensor** parametros, int n, float lr,
                           float beta1, float beta2, float eps);
tc_optimizador* tc_adamw (tc_tensor** parametros, int n, float lr,
                           float beta1, float beta2, float eps, float decaimiento_pesos);
void tc_paso_optimizador          (tc_optimizador* opt);
void tc_cero_gradientes_optimizador(tc_optimizador* opt);
void tc_liberar_optimizador       (tc_optimizador* opt);

void       tc_imprimir        (const tc_tensor* t);
float      tc_elemento        (const tc_tensor* t);
tc_tensor* tc_a_dispositivo   (tc_tensor* t, tc_dispositivo disp);
void       tc_imprimir_info   (void);
void       tc_semilla_aleatoria(uint64_t semilla);

#endif /* TESELCORE_H */