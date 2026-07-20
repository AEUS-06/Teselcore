/* src/penrose/semilla.c — Generación de las 10 tejas semilla ("configuración sol") */
#include "../../include/internal/penrose_interno.h"
#include <math.h>

/* BUG FIX: la semilla original ignoraba iL/iR con (void) y usaba índices
   incorrectos (iO, iC, iP, iP), generando geometría degenerada desde nivel 0.
   Convención correcta: v[0]=centro, v[1]=izq, v[2]=punta, v[3]=der. */
void _generar_semilla(_BufVert* verts, _BufBald* actual, float escala) {
    for (int i = 0; i < 10; i++) {
        float ang = i * 2.0f * PI_TC / 10.0f;
        float ox=0.0f, oy=0.0f;
        float lx=cosf(ang+ANG36)*escala, ly=sinf(ang+ANG36)*escala;
        float rx=cosf(ang-ANG36)*escala, ry=sinf(ang-ANG36)*escala;
        float px=cosf(ang+ANG36)*escala/TC_PHI, py=sinf(ang+ANG36)*escala/TC_PHI;

        int iO=_bv_obtener(verts,ox,oy);
        int iL=_bv_obtener(verts,lx,ly);
        int iR=_bv_obtener(verts,rx,ry);
        int iP=_bv_obtener(verts,px,py);

        int tipo = (i % 2 == 0) ? TC_PENROSE_KITE : TC_PENROSE_DART;
        _bb_agregar(actual, tipo, iO, iL, iP, iR, ang);
    }
}
