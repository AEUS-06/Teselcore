/* src/layers/batchnorm.c — Normalización por lotes */
#include "../../include/teselcore.h"
#include <math.h>
#include <assert.h>

tc_tensor* tc_normalizacion_lote(tc_tensor* x, tc_tensor* gamma, tc_tensor* beta,
                                  tc_tensor* media_mov, tc_tensor* var_mov,
                                  float eps, float momento, int entrenando) {
    (void)momento;
    assert(x->ndim >= 2);
    int C=(int)x->forma[1];
    tc_tensor* sal=tc_clonar(x);
    float *ds=(float*)sal->datos, *dg=(float*)gamma->datos, *db=(float*)beta->datos;
    float *mm=(float*)media_mov->datos, *mv=(float*)var_mov->datos;
    size_t N=x->total/(size_t)C;

    for (int c=0;c<C;c++) {
        float media=0.0f, varianza=0.0f;
        if (entrenando) {
            for (size_t i=0;i<N;i++) media+=((float*)x->datos)[c*(int)N+(int)i];
            media/=(float)N;
            for (size_t i=0;i<N;i++) {
                float d=((float*)x->datos)[c*(int)N+(int)i]-media;
                varianza+=d*d;
            }
            varianza/=(float)N; mm[c]=media; mv[c]=varianza;
        } else { media=mm[c]; varianza=mv[c]; }
        float inv=1.0f/sqrtf(varianza+eps);
        for (size_t i=0;i<N;i++) {
            float xh=(((float*)x->datos)[c*(int)N+(int)i]-media)*inv;
            ds[c*(int)N+(int)i]=dg[c]*xh+db[c];
        }
    }
    return sal;
}
