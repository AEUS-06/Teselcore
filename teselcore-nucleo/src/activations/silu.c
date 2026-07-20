/* src/activations/silu.c — SiLU: x · σ(x) */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

static void _bwd_silu(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos; size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) {
        float sig=1.0f/(1.0f+expf(-ent[i]));
        tmp[i]=g[i]*(sig+ent[i]*sig*(1.0f-sig));
    }
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_silu(tc_tensor* a) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) {
        float sig=1.0f/(1.0f+expf(-da[i]));
        ds[i]=da[i]*sig;
    }
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_silu);
    return s;
}
