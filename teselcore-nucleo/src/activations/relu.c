/* src/activations/relu.c — ReLU y Leaky-ReLU con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>

static void _bwd_relu(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=ent[i]>0.0f?g[i]:0.0f;
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

static void _bwd_relu_fuga(_nodo_autograd* n) {
    float alfa=n->escalar_guardado, *g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos;
    size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*(ent[i]>0.0f?1.0f:alfa);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_relu(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=da[i]>0.0f?da[i]:0.0f;
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_relu);
    return s;
}

tc_tensor* tc_relu_con_fuga(tc_tensor* a, float alfa) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=da[i]>0.0f?da[i]:alfa*da[i];
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,alfa,NULL,_bwd_relu_fuga);
    return s;
}
