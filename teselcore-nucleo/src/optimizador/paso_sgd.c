/* src/optimizador/paso_sgd.c — Un paso de actualización SGD con momento */
#include "../../include/internal/optimizador_interno.h"

void _paso_sgd(tc_optimizador* opt, tc_tensor* p, float* v) {
    float* d = (float*)p->datos;
    float* g = p->gradiente;
    for (size_t j = 0; j < p->total; j++) {
        v[j] = opt->momento * v[j] + g[j];
        d[j] -= opt->lr * v[j];
    }
}
