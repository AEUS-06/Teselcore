/* teselcore/src/formato_ax.c
 * Formato binario .ax y optimizadores SGD / Adam / AdamW.
 * Licencia: GNU LGPL v3
 */

#include "../include/teselcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ════════════════════════════════════════════════════
 * CRC32
 * ════════════════════════════════════════════════════ */
static uint32_t _tabla_crc32[256];
static int      _tabla_inicializada = 0;

static void _construir_tabla_crc32(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
        _tabla_crc32[i] = c;
    }
    _tabla_inicializada = 1;
}

static uint32_t _calcular_crc32(const uint8_t* datos, size_t longitud) {
    if (!_tabla_inicializada) _construir_tabla_crc32();
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < longitud; i++)
        crc = _tabla_crc32[(crc ^ datos[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFU;
}

/* ════════════════════════════════════════════════════
 * MACROS I/O
 * ════════════════════════════════════════════════════ */
#define ESCRIBIR_U8(f,v)  { uint8_t  _v=(uint8_t)(v);  fwrite(&_v,1,1,f); }
#define ESCRIBIR_U16(f,v) { uint16_t _v=(uint16_t)(v); fwrite(&_v,2,1,f); }
#define ESCRIBIR_U32(f,v) { uint32_t _v=(uint32_t)(v); fwrite(&_v,4,1,f); }
#define ESCRIBIR_U64(f,v) { uint64_t _v=(uint64_t)(v); fwrite(&_v,8,1,f); }

static uint8_t  LEER_U8 (FILE* f) { uint8_t  v=0; fread(&v,1,1,f); return v; }
static uint16_t LEER_U16(FILE* f) { uint16_t v=0; fread(&v,2,1,f); return v; }
static uint32_t LEER_U32(FILE* f) { uint32_t v=0; fread(&v,4,1,f); return v; }
static uint64_t LEER_U64(FILE* f) { uint64_t v=0; fread(&v,8,1,f); return v; }

/* ════════════════════════════════════════════════════
 * GUARDAR MODELO
 * ════════════════════════════════════════════════════ */
int tc_guardar(const tc_modelo* modelo, const char* ruta) {
    FILE* archivo = fopen(ruta, "wb");
    if (!archivo) {
        perror("tc_guardar: no se pudo abrir el archivo");
        return -1;
    }

    fwrite("AXON", 1, 4, archivo);
    ESCRIBIR_U16(archivo, modelo->version_mayor);
    ESCRIBIR_U16(archivo, modelo->version_menor);
    ESCRIBIR_U32(archivo, 0);

    uint32_t longitud_meta = modelo->metadatos
                             ? (uint32_t)strlen(modelo->metadatos) : 0;
    ESCRIBIR_U32(archivo, longitud_meta);
    if (longitud_meta > 0)
        fwrite(modelo->metadatos, 1, longitud_meta, archivo);

    ESCRIBIR_U32(archivo, (uint32_t)modelo->num_tensores);

    for (int i = 0; i < modelo->num_tensores; i++) {
        const tc_tensor_nombrado* tn = &modelo->tensores[i];
        tc_tensor* t = tn->tensor;

        uint16_t longitud_nombre = (uint16_t)strlen(tn->nombre);
        ESCRIBIR_U16(archivo, longitud_nombre);
        fwrite(tn->nombre, 1, longitud_nombre, archivo);

        ESCRIBIR_U8(archivo, (uint8_t)t->tipo);
        ESCRIBIR_U8(archivo, (uint8_t)t->ndim);
        for (int d = 0; d < t->ndim; d++)
            ESCRIBIR_U32(archivo, (uint32_t)t->forma[d]);

        uint64_t bytes_datos = (uint64_t)(t->total * TC_BYTES_POR_TIPO[t->tipo]);
        ESCRIBIR_U64(archivo, bytes_datos);
        fwrite(t->datos, 1, (size_t)bytes_datos, archivo);
    }

    long posicion = ftell(archivo);
    fclose(archivo);

    /* CRC: releer todo el archivo y appender el CRC */
    FILE* archivo_r = fopen(ruta, "rb");
    if (!archivo_r) {
        perror("tc_guardar: no se pudo reabrir para CRC");
        return -1;
    }
    fseek(archivo_r, 0, SEEK_END);
    long tam = ftell(archivo_r);
    fseek(archivo_r, 0, SEEK_SET);
    uint8_t* buffer_crc = (uint8_t*)malloc((size_t)tam);
    if (!buffer_crc) { fclose(archivo_r); return -1; }
    fread(buffer_crc, 1, (size_t)tam, archivo_r);
    uint32_t crc = _calcular_crc32(buffer_crc, (size_t)tam);
    free(buffer_crc);
    fclose(archivo_r);

    FILE* archivo_a = fopen(ruta, "ab");
    if (!archivo_a) {
        perror("tc_guardar: no se pudo abrir para append");
        return -1;
    }
    ESCRIBIR_U32(archivo_a, crc);
    fclose(archivo_a);
    printf("tc_guardar: '%s' guardado (%.2f KB)\n", ruta, (posicion + 4) / 1024.0);
    return 0;
}

/* ════════════════════════════════════════════════════
 * CARGAR MODELO
 * ════════════════════════════════════════════════════ */
tc_modelo* tc_cargar(const char* ruta) {
    FILE* archivo = fopen(ruta, "rb");
    if (!archivo) {
        perror("tc_cargar: no se pudo abrir el archivo");
        return NULL;
    }

    fseek(archivo, 0, SEEK_END);
    long tam_archivo = ftell(archivo);
    fseek(archivo, 0, SEEK_SET);

    /* BUG FIX: verificar que el archivo tiene al menos 4 bytes para el CRC */
    if (tam_archivo < 4) {
        fprintf(stderr, "tc_cargar: archivo demasiado pequeño\n");
        fclose(archivo);
        return NULL;
    }

    uint8_t* contenido = (uint8_t*)malloc((size_t)tam_archivo);
    if (!contenido) { fclose(archivo); return NULL; }
    fread(contenido, 1, (size_t)tam_archivo, archivo);
    fseek(archivo, 0, SEEK_SET);

    uint32_t crc_guardado, crc_calculado;
    memcpy(&crc_guardado, contenido + tam_archivo - 4, 4);
    crc_calculado = _calcular_crc32(contenido, (size_t)(tam_archivo - 4));
    free(contenido);

    if (crc_guardado != crc_calculado) {
        fprintf(stderr, "tc_cargar: CRC inválido — archivo corrupto\n");
        fclose(archivo);
        return NULL;
    }

    char magic[4];
    fread(magic, 1, 4, archivo);
    if (memcmp(magic, "AXON", 4) != 0) {
        fprintf(stderr, "tc_cargar: no es un archivo .ax válido\n");
        fclose(archivo);
        return NULL;
    }

    tc_modelo* modelo = (tc_modelo*)calloc(1, sizeof(tc_modelo));
    modelo->version_mayor = LEER_U16(archivo);
    modelo->version_menor = LEER_U16(archivo);
    LEER_U32(archivo); /* banderas */

    uint32_t longitud_meta = LEER_U32(archivo);
    if (longitud_meta > 0) {
        modelo->metadatos = (char*)malloc(longitud_meta + 1);
        fread(modelo->metadatos, 1, longitud_meta, archivo);
        modelo->metadatos[longitud_meta] = '\0';
    }

    uint32_t num = LEER_U32(archivo);
    modelo->num_tensores = (int)num;
    modelo->tensores     = (tc_tensor_nombrado*)calloc(num, sizeof(tc_tensor_nombrado));

    for (uint32_t i = 0; i < num; i++) {
        tc_tensor_nombrado* tn = &modelo->tensores[i];

        uint16_t longitud_nombre = LEER_U16(archivo);
        if (longitud_nombre >= (uint16_t)sizeof(tn->nombre)) {
            size_t to_read = sizeof(tn->nombre) - 1;
            fread(tn->nombre, 1, to_read, archivo);
            tn->nombre[to_read] = '\0';
            long skip = (long)longitud_nombre - (long)to_read;
            if (skip > 0) fseek(archivo, skip, SEEK_CUR);
        } else {
            fread(tn->nombre, 1, longitud_nombre, archivo);
            tn->nombre[longitud_nombre] = '\0';
        }

        tc_tipo_dato tipo = (tc_tipo_dato)LEER_U8(archivo);
        int          ndim = (int)LEER_U8(archivo);
        int          forma[TC_MAX_DIMS];
        for (int d = 0; d < ndim; d++)
            forma[d] = (int)LEER_U32(archivo);

        uint64_t bytes_datos = LEER_U64(archivo);
        void* datos = malloc((size_t)bytes_datos);
        fread(datos, 1, (size_t)bytes_datos, archivo);
        tn->tensor = tc_desde_datos(datos, forma, ndim, tipo);
        free(datos);
    }

    fclose(archivo);
    printf("tc_cargar: '%s' cargado (%d tensores)\n", ruta, modelo->num_tensores);
    return modelo;
}

/* ════════════════════════════════════════════════════
 * API DEL MODELO
 * ════════════════════════════════════════════════════ */
tc_modelo* tc_modelo_nuevo(const char* metadatos_json) {
    tc_modelo* m = (tc_modelo*)calloc(1, sizeof(tc_modelo));
    m->version_mayor = TC_VERSION_MAYOR;
    m->version_menor = TC_VERSION_MENOR;
    if (metadatos_json)
        m->metadatos = strdup(metadatos_json);
    return m;
}

int tc_modelo_agregar_tensor(tc_modelo* modelo, const char* nombre, tc_tensor* t) {
    modelo->tensores = (tc_tensor_nombrado*)realloc(
        modelo->tensores,
        (size_t)(modelo->num_tensores + 1) * sizeof(tc_tensor_nombrado)
    );
    tc_tensor_nombrado* tn = &modelo->tensores[modelo->num_tensores++];
    strncpy(tn->nombre, nombre, 255);
    tn->nombre[255] = '\0';
    tn->tensor = t;
    return 0;
}

tc_tensor* tc_modelo_obtener_tensor(const tc_modelo* modelo, const char* nombre) {
    for (int i = 0; i < modelo->num_tensores; i++)
        if (strcmp(modelo->tensores[i].nombre, nombre) == 0)
            return modelo->tensores[i].tensor;
    return NULL;
}

void tc_liberar_modelo(tc_modelo* modelo) {
    if (!modelo) return;
    for (int i = 0; i < modelo->num_tensores; i++)
        tc_liberar(modelo->tensores[i].tensor);
    free(modelo->tensores);
    free(modelo->metadatos);
    free(modelo);
}

/* ════════════════════════════════════════════════════
 * OPTIMIZADORES
 * ════════════════════════════════════════════════════ */
struct tc_optimizador {
    tc_tensor** parametros;
    int         num_params;
    float       lr;

    enum { OPT_SGD=0, OPT_ADAM=1, OPT_ADAMW=2 } tipo;

    float   momento;
    float** velocidades;

    float   beta1, beta2, eps, decaimiento_pesos;
    float** momento1;
    float** momento2;
    int     paso;
};

tc_optimizador* tc_sgd(tc_tensor** params, int n, float lr, float momento) {
    tc_optimizador* opt = (tc_optimizador*)calloc(1, sizeof(tc_optimizador));
    opt->parametros  = params;
    opt->num_params  = n;
    opt->lr          = lr;
    opt->momento     = momento;
    opt->tipo        = OPT_SGD;
    opt->velocidades = (float**)calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++)
        opt->velocidades[i] = (float*)calloc(params[i]->total, sizeof(float));
    return opt;
}

tc_optimizador* tc_adam(tc_tensor** params, int n, float lr,
                         float beta1, float beta2, float eps) {
    tc_optimizador* opt = (tc_optimizador*)calloc(1, sizeof(tc_optimizador));
    opt->parametros = params;
    opt->num_params = n;
    opt->lr   = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps   = eps;
    opt->tipo  = OPT_ADAM;
    opt->momento1 = (float**)calloc(n, sizeof(float*));
    opt->momento2 = (float**)calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++) {
        opt->momento1[i] = (float*)calloc(params[i]->total, sizeof(float));
        opt->momento2[i] = (float*)calloc(params[i]->total, sizeof(float));
    }
    return opt;
}

tc_optimizador* tc_adamw(tc_tensor** params, int n, float lr,
                          float beta1, float beta2, float eps, float decaimiento) {
    tc_optimizador* opt    = tc_adam(params, n, lr, beta1, beta2, eps);
    opt->tipo              = OPT_ADAMW;
    opt->decaimiento_pesos = decaimiento;
    return opt;
}

void tc_paso_optimizador(tc_optimizador* opt) {
    opt->paso++;
    for (int i = 0; i < opt->num_params; i++) {
        tc_tensor* p = opt->parametros[i];
        if (!p->requiere_grad || !p->gradiente) continue;
        float* d = (float*)p->datos;
        float* g = p->gradiente;

        if (opt->tipo == OPT_SGD) {
            float* v = opt->velocidades[i];
            for (size_t j = 0; j < p->total; j++) {
                v[j] = opt->momento * v[j] + g[j];
                d[j] -= opt->lr * v[j];
            }
        } else {
            float correc1 = 1.0f - powf(opt->beta1, (float)opt->paso);
            float correc2 = 1.0f - powf(opt->beta2, (float)opt->paso);
            float lr_corr = opt->lr * sqrtf(correc2) / correc1;
            float* m1 = opt->momento1[i];
            float* m2 = opt->momento2[i];

            for (size_t j = 0; j < p->total; j++) {
                float gv = g[j];
                m1[j] = opt->beta1 * m1[j] + (1.0f - opt->beta1) * gv;
                m2[j] = opt->beta2 * m2[j] + (1.0f - opt->beta2) * gv * gv;
                float actualizacion = lr_corr * m1[j] / (sqrtf(m2[j]) + opt->eps);

                if (opt->tipo == OPT_ADAMW)
                    actualizacion += opt->lr * opt->decaimiento_pesos * d[j];

                d[j] -= actualizacion;
            }
        }
    }
}

void tc_cero_gradientes_optimizador(tc_optimizador* opt) {
    for (int i = 0; i < opt->num_params; i++)
        tc_cero_gradiente(opt->parametros[i]);
}

void tc_liberar_optimizador(tc_optimizador* opt) {
    if (!opt) return;
    for (int i = 0; i < opt->num_params; i++) {
        if (opt->velocidades) free(opt->velocidades[i]);
        if (opt->momento1)    free(opt->momento1[i]);
        if (opt->momento2)    free(opt->momento2[i]);
    }
    free(opt->velocidades);
    free(opt->momento1);
    free(opt->momento2);
    free(opt);
}