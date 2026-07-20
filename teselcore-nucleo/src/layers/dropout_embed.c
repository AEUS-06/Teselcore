/* src/layers/dropout_embed.c — Abandono (dropout) y tabla de embeddings */
#include "../../include/teselcore.h"
#include "../../include/internal/rng.h"
#include <string.h>
#include <assert.h>

tc_tensor* tc_abandono(tc_tensor* x, float p, int entrenando) {
    if (!entrenando || p <= 0.0f) return tc_clonar(x);
    tc_tensor* sal=tc_clonar(x);
    float* ds=(float*)sal->datos, escala=1.0f/(1.0f-p);
    for (size_t i=0;i<x->total;i++) {
        if (_uniforme()<p) ds[i]=0.0f;
        else               ds[i]*=escala;
    }
    return sal;
}

tc_tensor* tc_embedding(tc_tensor* indices, tc_tensor* pesos) {
    assert(indices->ndim==1||indices->ndim==2);
    int* idx=(int*)indices->datos;
    int vocab=pesos->forma[0], dim=pesos->forma[1], seq=(int)indices->total;
    int forma[]={seq,dim};
    tc_tensor* sal=tc_vacio(forma,2,TC_FLOAT32,TC_CPU);
    float *dp=(float*)pesos->datos, *ds=(float*)sal->datos;
    for (int i=0;i<seq;i++) {
        assert(idx[i]>=0&&idx[i]<vocab);
        memcpy(ds+i*dim, dp+idx[i]*dim, dim*sizeof(float));
    }
    return sal;
}
