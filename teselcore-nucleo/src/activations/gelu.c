/* src/activations/gelu.c — GELU: approx. 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

#define SQRT2PI 0.7978845608f
#define GELU_C  0.044715f

static void _bwd_gelu(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) {
        float x=ent[i], th=tanhf(SQRT2PI*(x+GELU_C*x*x*x)), s2=1.0f-th*th;
        tmp[i]=g[i]*(0.5f*(1.0f+th)+0.5f*x*s2*SQRT2PI*(1.0f+3.0f*GELU_C*x*x));
    }
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_gelu(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) {
        float x=da[i]; ds[i]=0.5f*x*(1.0f+tanhf(SQRT2PI*(x+GELU_C*x*x*x)));
    }
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_gelu);
    return s;
}
