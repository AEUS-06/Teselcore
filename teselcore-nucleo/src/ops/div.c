/* src/ops/div.c — División elemento a elemento con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <assert.h>

static void _bwd_dividir(_nodo_autograd* n) {
    float* g=n->salida->gradiente;
    float* da=(float*)n->entradas[0]->datos, *db=(float*)n->entradas[1]->datos;
    size_t sz=n->salida->total;
    if (n->entradas[0]->requiere_grad) {
        float* tmp=(float*)malloc(sz*sizeof(float));
        for (size_t i=0;i<sz;i++) tmp[i]=g[i]/(db[i]+1e-8f);
        _acumular_grad(n->entradas[0],tmp);
        free(tmp);
    }
    if (n->entradas[1]->requiere_grad) {
        float* tmp=(float*)malloc(sz*sizeof(float));
        for (size_t i=0;i<sz;i++) tmp[i]=-g[i]*da[i]/(db[i]*db[i]+1e-8f);
        _acumular_grad(n->entradas[1],tmp);
        free(tmp);
    }
}

tc_tensor* tc_dividir(tc_tensor* a, tc_tensor* b) {
    assert(a->total==b->total);
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *db=(float*)b->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=da[i]/(db[i]+1e-8f);
    s->requiere_grad=a->requiere_grad||b->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,b,2,0,NULL,_bwd_dividir);
    return s;
}
