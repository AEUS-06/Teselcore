/* src/ops/pow.c — Potencia con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <math.h>

static void _bwd_potencia(_nodo_autograd* n) {
    float ev=n->escalar_guardado, *g=n->salida->gradiente, *ent=(float*)n->entradas[0]->datos;
    size_t sz=n->salida->total;
    if (!n->entradas[0]->requiere_grad) return;
    float* tmp=(float*)malloc(sz*sizeof(float));
    for (size_t i=0;i<sz;i++) tmp[i]=g[i]*ev*powf(ent[i],ev-1.0f);
    _acumular_grad(n->entradas[0],tmp); free(tmp);
}

tc_tensor* tc_potencia(tc_tensor* a, float ev) {
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    for (size_t i=0;i<a->total;i++) ds[i]=powf(da[i],ev);
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,ev,NULL,_bwd_potencia);
    return s;
}
