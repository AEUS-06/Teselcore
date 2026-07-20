/* src/tensor/create_basic.c — Constructores básicos: vacío, ceros, unos */
#include "../../include/teselcore.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

tc_tensor* tc_vacio(const int* forma, int ndim, tc_tipo_dato tipo, tc_dispositivo disp) {
    assert(ndim > 0 && ndim <= TC_MAX_DIMS);
    tc_tensor* t = (tc_tensor*)calloc(1, sizeof(tc_tensor));
    t->ndim = ndim; t->tipo = tipo; t->dispositivo = disp; t->_refs = 1;
    size_t total = 1;
    for (int i = 0; i < ndim; i++) { t->forma[i] = forma[i]; total *= (size_t)forma[i]; }
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
