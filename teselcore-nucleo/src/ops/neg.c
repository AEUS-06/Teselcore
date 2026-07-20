/* src/ops/neg.c — Negación con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>

static void _bwd_negar(_nodo_autograd* n) {
    float* g=n->salida->gradiente; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=-g[i];
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_negar(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=-da[i];
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_negar);
    return s;
}
