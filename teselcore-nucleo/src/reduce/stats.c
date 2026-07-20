/* src/reduce/stats.c — Media, máximo y mínimo */
#include "../../include/teselcore.h"

tc_tensor* tc_media_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float suma = 0.0f;
    float* d = (float*)a->datos;
    for (size_t i = 0; i < a->total; i++) suma += d[i];
    float media = suma / (float)a->total;
    int forma[] = {1};
    return tc_desde_datos(&media, forma, 1, TC_FLOAT32);
}

tc_tensor* tc_maximo_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float* d = (float*)a->datos, mx = d[0];
    for (size_t i = 1; i < a->total; i++) if (d[i] > mx) mx = d[i];
    int forma[] = {1};
    return tc_desde_datos(&mx, forma, 1, TC_FLOAT32);
}

tc_tensor* tc_minimo_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float* d = (float*)a->datos, mn = d[0];
    for (size_t i = 1; i < a->total; i++) if (d[i] < mn) mn = d[i];
    int forma[] = {1};
    return tc_desde_datos(&mn, forma, 1, TC_FLOAT32);
}
