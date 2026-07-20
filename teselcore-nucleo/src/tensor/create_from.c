/* src/tensor/create_from.c — Constructores desde datos o valor escalar */
#include "../../include/teselcore.h"
#include <string.h>

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
