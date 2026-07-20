/* src/linalg/transform.c — Transposición y reforma de forma */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <assert.h>

static void _bwd_transponer(_nodo_autograd* n) {
    tc_tensor* in = n->entradas[0];
    if (!in->requiere_grad || in->ndim != 2) return;
    _asegurar_gradiente(in);
    float* g = n->salida->gradiente;
    int F = in->forma[0], C = in->forma[1];
    for (int f = 0; f < F; f++)
        for (int c = 0; c < C; c++)
            in->gradiente[f*C+c] += g[c*F+f];
}

tc_tensor* tc_transponer(tc_tensor* a, int dim0, int dim1) {
    tc_tensor* s = tc_vacio(a->forma, a->ndim, a->tipo, a->dispositivo);
    s->forma[dim0]=a->forma[dim1]; s->forma[dim1]=a->forma[dim0];
    s->pasos[dim0]=a->pasos[dim1]; s->pasos[dim1]=a->pasos[dim0];
    if (a->ndim == 2) {
        float *src=(float*)a->datos, *dst=(float*)s->datos;
        int F=a->forma[0], C=a->forma[1];
        for (int f=0;f<F;f++) for (int c=0;c<C;c++) dst[c*F+f]=src[f*C+c];
    }
    s->requiere_grad = a->requiere_grad;
    if (s->requiere_grad && a->ndim == 2)
        _tape_empujar(s, a, NULL, 1, 0, NULL, _bwd_transponer);
    return s;
}

tc_tensor* tc_reformar(tc_tensor* a, const int* nueva_forma, int nuevo_ndim) {
    tc_tensor* s = tc_clonar(a);
    size_t total = 1;
    for (int i=0;i<nuevo_ndim;i++) { s->forma[i]=nueva_forma[i]; total*=(size_t)nueva_forma[i]; }
    assert(total == a->total);
    s->ndim = nuevo_ndim;
    _calcular_pasos(nueva_forma, nuevo_ndim, s->pasos);
    s->requiere_grad = a->requiere_grad;
    return s;
}
