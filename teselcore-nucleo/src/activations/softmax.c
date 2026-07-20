/* src/activations/softmax.c — Softmax con retropropagación estable */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <math.h>

static void _bwd_softmax(_nodo_autograd* n) {
    float* g=n->salida->gradiente, *s=(float*)n->salida->datos;
    int cols=n->entradas[0]->forma[n->entradas[0]->ndim-1];
    int filas=(int)(n->salida->total/cols);
    if (!n->entradas[0]->requiere_grad) return;
    _asegurar_gradiente(n->entradas[0]);
    for (int f=0;f<filas;f++) {
        float *gr=g+f*cols, *sr=s+f*cols, dot=0.0f;
        for (int c=0;c<cols;c++) dot+=gr[c]*sr[c];
        for (int c=0;c<cols;c++) n->entradas[0]->gradiente[f*cols+c]+=sr[c]*(gr[c]-dot);
    }
}

tc_tensor* tc_softmax(tc_tensor* a, int dimension) {
    (void)dimension;
    tc_tensor* s=tc_vacio(a->forma,a->ndim,TC_FLOAT32,TC_CPU);
    float *da=(float*)a->datos, *ds=(float*)s->datos;
    int cols=a->forma[a->ndim-1], filas=(int)(a->total/cols);
    for (int f=0;f<filas;f++) {
        float *fe=da+f*cols, *fs=ds+f*cols, mx=fe[0];
        for (int c=1;c<cols;c++) if (fe[c]>mx) mx=fe[c];
        float suma=0.0f;
        for (int c=0;c<cols;c++) { fs[c]=expf(fe[c]-mx); suma+=fs[c]; }
        for (int c=0;c<cols;c++) fs[c]/=suma;
    }
    s->requiere_grad=a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s,a,NULL,1,0,NULL,_bwd_softmax);
    return s;
}
