/* src/tensor/lifecycle.c — Ciclo de vida: clonar, liberar, referencias */
#include "../../include/teselcore.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <string.h>

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

void tc_cero_gradiente(tc_tensor* t) {
    if (t->gradiente) memset(t->gradiente, 0, t->total * sizeof(float));
}
