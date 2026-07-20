/* src/ops/sub.c — Resta con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <assert.h>

static void _bwd_restar(_nodo_autograd* n) {
    float* g=n->salida->gradiente; size_t sz=n->salida->total;
    if (n->entradas[0]->requiere_grad) _acumular_grad(n->entradas[0],g);
    if (n->entradas[1]->requiere_grad) {
        float* tmp=(float*)malloc(sz*sizeof(float));
        for (size_t i=0;i<sz;i++) tmp[i]=-g[i];
        _acumular_grad(n->entradas[1],tmp); free(tmp);
    }
}

tc_tensor* tc_restar(tc_tensor* a, tc_tensor* b) {
    assert(a->total==b->total);
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos,*db=(float*)b->datos,*ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=da[i]-db[i];
    s->requiere_grad=a->requiere_grad||b->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,b,2,0,NULL,_bwd_restar);
    return s;
}
