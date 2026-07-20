/* src/ops/exp_log.c — Exponencial y logaritmo con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

static void _bwd_exp(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *sal=(float*)n->salida->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*sal[i];
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

static void _bwd_log(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]/(ent[i]+1e-8f);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_exponencial(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=expf(da[i]);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_exp);
    return s;
}

tc_tensor* tc_logaritmo(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=logf(da[i]+1e-8f);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_log);
    return s;
}
