/* src/tensor/create_random.c — Constructores aleatorios */
#include "../../include/teselcore.h"
#include "../../include/internal/rng.h"

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
