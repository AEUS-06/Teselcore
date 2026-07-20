/* src/penrose/imagen_teselacion.c — Proyección imagen → teselación (bilineal) */
#include "../../include/teselcore.h"
#include "../../include/internal/penrose_interno.h"
#include <assert.h>

tc_tensor* tc_imagen_a_teselacion(tc_tensor* imagen, const tc_teselacion_penrose* tes) {
    assert(imagen->ndim == 4);
    int B=imagen->forma[0], C=imagen->forma[1], H=imagen->forma[2], W=imagen->forma[3];
    int N=tes->num_tejas;
    int forma[]={B,C,N};
    tc_tensor* sal=tc_ceros(forma, 3);
    float *di=(float*)imagen->datos, *ds=(float*)sal->datos;

    float xmin,xmax,ymin,ymax;
    _calcular_bbox_teselacion(tes, &xmin,&xmax,&ymin,&ymax);
    float xrng=xmax-xmin+1e-8f, yrng=ymax-ymin+1e-8f;

    for (int b=0;b<B;b++) for (int i=0;i<N;i++) {
        float nx=(tes->tejas[i].centro_x-xmin)/xrng, ny=(tes->tejas[i].centro_y-ymin)/yrng;
        float fx=nx*(W-1), fy=ny*(H-1);
        int x0=(int)fx, y0=(int)fy, x1=x0+1<W?x0+1:x0, y1=y0+1<H?y0+1:y0;
        float wx=fx-x0, wy=fy-y0;
        for (int c=0;c<C;c++) {
            float v00=di[b*C*H*W+c*H*W+y0*W+x0], v10=di[b*C*H*W+c*H*W+y0*W+x1];
            float v01=di[b*C*H*W+c*H*W+y1*W+x0], v11=di[b*C*H*W+c*H*W+y1*W+x1];
            ds[b*C*N+c*N+i] = v00*(1-wx)*(1-wy)+v10*wx*(1-wy)+v01*(1-wx)*wy+v11*wx*wy;
        }
    }
    return sal;
}
