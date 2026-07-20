/* src/linalg/concat.c — Aplanamiento y concatenación */
#include "../../include/teselcore.h"
#include <string.h>

tc_tensor* tc_aplanar(tc_tensor* a, int dim_inicio) {
    (void)dim_inicio;
    int nueva_forma[] = {(int)a->total};
    return tc_reformar(a, nueva_forma, 1);
}

/* Nota: sólo concatena tensores 1D en dim 0. Extender a N-dim es un paso
   natural sin romper la interfaz pública. */
tc_tensor* tc_concatenar(tc_tensor** tensores, int n, int dimension) {
    (void)dimension;
    size_t total = 0;
    for (int i=0;i<n;i++) total += tensores[i]->total;
    int forma[] = {(int)total};
    tc_tensor* s = tc_vacio(forma, 1, TC_FLOAT32, TC_CPU);
    float* ds = (float*)s->datos;
    size_t offset = 0;
    for (int i=0;i<n;i++) {
        memcpy(ds+offset, tensores[i]->datos, tensores[i]->total*sizeof(float));
        offset += tensores[i]->total;
    }
    return s;
}
