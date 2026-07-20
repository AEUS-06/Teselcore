/* src/linalg/matmul.c — Multiplicación matricial con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <assert.h>

static void _bwd_matmul(_nodo_autograd* n) {
    tc_tensor* A=n->entradas[0], *B=n->entradas[1];
    float* g=n->salida->gradiente;
    int M=A->forma[0], K=A->forma[1], N=B->forma[1];
    if (A->requiere_grad) {
        _asegurar_gradiente(A);
        for (int m=0;m<M;m++) for (int k=0;k<K;k++) for (int nn=0;nn<N;nn++)
            A->gradiente[m*K+k] += g[m*N+nn] * ((float*)B->datos)[k*N+nn];
    }
    if (B->requiere_grad) {
        _asegurar_gradiente(B);
        for (int k=0;k<K;k++) for (int nn=0;nn<N;nn++) for (int m=0;m<M;m++)
            B->gradiente[k*N+nn] += ((float*)A->datos)[m*K+k] * g[m*N+nn];
    }
}

tc_tensor* tc_multiplicacion_matricial(tc_tensor* a, tc_tensor* b) {
    assert(a->ndim==2 && b->ndim==2 && a->forma[1]==b->forma[0]);
    int M=a->forma[0], K=a->forma[1], N=b->forma[1];
    int forma[]={M,N};
    tc_tensor* s=tc_ceros(forma, 2);
    float *da=(float*)a->datos, *db=(float*)b->datos, *ds=(float*)s->datos;
    for (int i=0;i<M;i++) for (int k=0;k<K;k++) {
        float va=da[i*K+k];
        for (int j=0;j<N;j++) ds[i*N+j]+=va*db[k*N+j];
    }
    s->requiere_grad = a->requiere_grad || b->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s, a, b, 2, 0, NULL, _bwd_matmul);
    return s;
}
