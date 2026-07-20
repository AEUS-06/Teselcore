/* src/penrose/teselacion_imagen.c — Proyección teselación → imagen (splatting) */
#include "../../include/teselcore.h"
#include "../../include/internal/penrose_interno.h"
#include <assert.h>

tc_tensor* tc_teselacion_a_imagen(tc_tensor* car, const tc_teselacion_penrose* tes,
                                  int alto, int ancho) {
    assert(car->ndim == 3);
    int B=car->forma[0], C=car->forma[1], N=car->forma[2];
    assert(N == tes->num_tejas);
    int forma[]={B,C,alto,ancho};
    tc_tensor* sal=tc_ceros(forma, 4), *pesos=tc_ceros(forma, 4);
    float *dc=(float*)car->datos, *ds=(float*)sal->datos, *dp=(float*)pesos->datos;

    float xmin,xmax,ymin,ymax;
    _calcular_bbox_teselacion(tes, &xmin,&xmax,&ymin,&ymax);
    float xrng=xmax-xmin+1e-8f, yrng=ymax-ymin+1e-8f;

    for (int b=0;b<B;b++) for (int i=0;i<N;i++) {
        float nx=(tes->tejas[i].centro_x-xmin)/xrng, ny=(tes->tejas[i].centro_y-ymin)/yrng;
        float fx=nx*(ancho-1), fy=ny*(alto-1);
        int x0=(int)fx, y0=(int)fy, x1=x0+1<ancho?x0+1:x0, y1=y0+1<alto?y0+1:y0;
        float wx=fx-x0, wy=fy-y0;
        float ws[4]={(1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy};
        int xs[4]={x0,x1,x0,x1}, ys[4]={y0,y0,y1,y1};
        for (int c=0;c<C;c++) {
            float val=dc[b*C*N+c*N+i];
            for (int p=0;p<4;p++) {
                int idx=b*C*alto*ancho+c*alto*ancho+ys[p]*ancho+xs[p];
                ds[idx]+=val*ws[p]; dp[idx]+=ws[p];
            }
        }
    }
    size_t total=(size_t)B*C*alto*ancho;
    for (size_t i=0;i<total;i++) if (dp[i]>1e-8f) ds[i]/=dp[i];
    tc_liberar(pesos);
    return sal;
}
