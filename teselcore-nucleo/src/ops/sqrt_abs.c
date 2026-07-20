/* src/ops/sqrt_abs.c — Raíz cuadrada y valor absoluto con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

static void _bwd_raiz(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *sal=(float*)n->salida->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]/(2.0f*sal[i]+1e-8f);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

static void _bwd_abs(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*(ent[i]>=0.0f?1.0f:-1.0f);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_raiz(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=sqrtf(da[i]<0.0f?0.0f:da[i]);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_raiz);
    return s;
}

tc_tensor* tc_valor_absoluto(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=fabsf(da[i]);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_abs);
    return s;
}
