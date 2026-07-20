/* src/penrose/conv_imagen.c — Convolución de Penrose en modo imagen (bilineal) */
#include "../../include/teselcore.h"
#include "../../include/internal/penrose_interno.h"

tc_tensor* tc_conv_penrose(tc_tensor* entrada, tc_kernel_penrose* kernel,
                           int paso, int modo_relleno) {
    (void)modo_relleno;
    int L=entrada->forma[0], Ce=entrada->forma[1], H=entrada->forma[2], W=entrada->forma[3];
    int Cs=kernel->canales_salida, T=kernel->teselacion->num_tejas;
    int Hs=H/paso, Ws=W/paso;
    int forma[]={L,Cs,Hs,Ws};
    tc_tensor* sal=tc_ceros(forma, 4);
    float *de=(float*)entrada->datos, *ds=(float*)sal->datos;
    float *dp=(float*)kernel->pesos->datos, *db=(float*)kernel->sesgo->datos;

    float xmin,xmax,ymin,ymax;
    _calcular_bbox_teselacion(kernel->teselacion, &xmin,&xmax,&ymin,&ymax);
    float xrng=xmax-xmin+1e-8f, yrng=ymax-ymin+1e-8f;

    for (int b=0;b<L;b++) for (int cs=0;cs<Cs;cs++)
    for (int ih=0;ih<Hs;ih++) for (int iw=0;iw<Ws;iw++) {
        float ch=ih*paso+paso/2.0f, cw=iw*paso+paso/2.0f, acc=0.0f;
        for (int t=0;t<T;t++) {
            float nx=(kernel->teselacion->tejas[t].centro_x-xmin)/xrng;
            float ny=(kernel->teselacion->tejas[t].centro_y-ymin)/yrng;
            float px=cw+(nx-0.5f)*paso, py=ch+(ny-0.5f)*paso;
            for (int ce=0;ce<Ce;ce++) {
                const float* canal=de+b*Ce*H*W+ce*H*W;
                float v=_interpolar_bilineal(canal,H,W,px,py);
                acc += v * dp[cs*Ce*T+ce*T+t];
            }
        }
        ds[b*Cs*Hs*Ws+cs*Hs*Ws+ih*Ws+iw]=acc+db[cs];
    }
    return sal;
}
