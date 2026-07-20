/* src/layers/pooling.c — Agrupación máxima y promedio */
#include "../../include/teselcore.h"
#include <assert.h>

tc_tensor* tc_agrupacion_max(tc_tensor* x, int kernel, int paso) {
    assert(x->ndim==4);
    int B=x->forma[0],C=x->forma[1],H=x->forma[2],W=x->forma[3];
    int Hs=(H-kernel)/paso+1, Ws=(W-kernel)/paso+1;
    int forma[]={B,C,Hs,Ws};
    tc_tensor* sal=tc_vacio(forma,4,TC_FLOAT32,TC_CPU);
    float *dx=(float*)x->datos, *ds=(float*)sal->datos;
    for (int b=0;b<B;b++) for (int c=0;c<C;c++)
    for (int oh=0;oh<Hs;oh++) for (int ow=0;ow<Ws;ow++) {
        float mx=-1e38f;
        for (int kh=0;kh<kernel;kh++) for (int kw=0;kw<kernel;kw++) {
            float v=dx[b*C*H*W+c*H*W+(oh*paso+kh)*W+(ow*paso+kw)];
            if (v>mx) mx=v;
        }
        ds[b*C*Hs*Ws+c*Hs*Ws+oh*Ws+ow]=mx;
    }
    return sal;
}

tc_tensor* tc_agrupacion_promedio(tc_tensor* x, int kernel, int paso) {
    assert(x->ndim==4);
    int B=x->forma[0],C=x->forma[1],H=x->forma[2],W=x->forma[3];
    int Hs=(H-kernel)/paso+1, Ws=(W-kernel)/paso+1;
    int forma[]={B,C,Hs,Ws};
    tc_tensor* sal=tc_ceros(forma,4);
    float *dx=(float*)x->datos, *ds=(float*)sal->datos, area=(float)(kernel*kernel);
    for (int b=0;b<B;b++) for (int c=0;c<C;c++)
    for (int oh=0;oh<Hs;oh++) for (int ow=0;ow<Ws;ow++) {
        float suma=0.0f;
        for (int kh=0;kh<kernel;kh++) for (int kw=0;kw<kernel;kw++)
            suma+=dx[b*C*H*W+c*H*W+(oh*paso+kh)*W+(ow*paso+kw)];
        ds[b*C*Hs*Ws+c*Hs*Ws+oh*Ws+ow]=suma/area;
    }
    return sal;
}
