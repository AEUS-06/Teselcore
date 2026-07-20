/* src/layers/conv2d.c — Convolución 2D (implementación naive, extensible) */
#include "../../include/teselcore.h"
#include <assert.h>

tc_tensor* tc_conv2d(tc_tensor* x, tc_tensor* kernel, tc_tensor* sesgo,
                     int paso, int relleno) {
    assert(x->ndim==4 && kernel->ndim==4);
    int B=x->forma[0], Cin=x->forma[1], H=x->forma[2], W=x->forma[3];
    int Cout=kernel->forma[0], kH=kernel->forma[2], kW=kernel->forma[3];
    int Hs=(H+2*relleno-kH)/paso+1, Ws=(W+2*relleno-kW)/paso+1;
    int forma[]={B,Cout,Hs,Ws};
    tc_tensor* sal=tc_ceros(forma, 4);
    float *dx=(float*)x->datos, *dk=(float*)kernel->datos, *ds=(float*)sal->datos;

    for (int b=0;b<B;b++) for (int co=0;co<Cout;co++)
    for (int oh=0;oh<Hs;oh++) for (int ow=0;ow<Ws;ow++) {
        float suma = sesgo ? ((float*)sesgo->datos)[co] : 0.0f;
        for (int ci=0;ci<Cin;ci++) for (int kh=0;kh<kH;kh++) for (int kw=0;kw<kW;kw++) {
            int ih=oh*paso-relleno+kh, iw=ow*paso-relleno+kw;
            if (ih>=0&&ih<H&&iw>=0&&iw<W)
                suma+=dx[b*Cin*H*W+ci*H*W+ih*W+iw]*dk[co*Cin*kH*kW+ci*kH*kW+kh*kW+kw];
        }
        ds[b*Cout*Hs*Ws+co*Hs*Ws+oh*Ws+ow]=suma;
    }
    return sal;
}
