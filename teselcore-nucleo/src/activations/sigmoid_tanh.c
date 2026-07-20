/* src/activations/sigmoid_tanh.c — Sigmoide y tangente hiperbólica */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

static void _bwd_sigmoide(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *sal=(float*)n->salida->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*sal[i]*(1.0f-sal[i]);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

static void _bwd_tanh(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *sal=(float*)n->salida->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*(1.0f-sal[i]*sal[i]);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_sigmoide(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=1.0f/(1.0f+expf(-da[i]));
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_sigmoide);
    return s;
}

tc_tensor* tc_tangente_hiperbolica(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=tanhf(da[i]);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_tanh);
    return s;
}
