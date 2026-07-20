/* src/penrose/subdivide.c — Subdivisión deflacionaria de tejas kite/dart */
#include "../../include/internal/penrose_interno.h"

void _subdividir(_BufVert* verts, _BufBald* entrada, _BufBald* salida) {
    for (int i = 0; i < entrada->n; i++) {
        _Baldosa* b = &entrada->buf[i];
        _Vertice* v = verts->buf;
        float cx=v[b->v[0]].x, cy=v[b->v[0]].y, lx=v[b->v[1]].x, ly=v[b->v[1]].y;
        float px=v[b->v[2]].x, py=v[b->v[2]].y, rx=v[b->v[3]].x, ry=v[b->v[3]].y;

        if (b->tipo == TC_PENROSE_KITE) {
            float qx=cx+(px-cx)/TC_PHI,  qy=cy+(py-cy)/TC_PHI;
            float r2x=lx+(cx-lx)/TC_PHI, r2y=ly+(cy-ly)/TC_PHI;
            float sx=rx+(cx-rx)/TC_PHI,  sy=ry+(cy-ry)/TC_PHI;
            int iC=b->v[0], iL=b->v[1], iP=b->v[2], iR=b->v[3];
            int iQ=_bv_obtener(verts,qx,qy), iR2=_bv_obtener(verts,r2x,r2y), iS=_bv_obtener(verts,sx,sy);
            /* Convención: v[0]=centro, v[1]=izq, v[2]=punta, v[3]=der */
            _bb_agregar(salida, TC_PENROSE_KITE, iQ, iL, iP, iR2, b->angulo);
            _bb_agregar(salida, TC_PENROSE_DART, iC, iR2, iQ, iL, b->angulo + ANG36);
            _bb_agregar(salida, TC_PENROSE_DART, iC, iS, iQ, iR, b->angulo - ANG36);
        } else {
            float qx=lx+(px-lx)/TC_PHI, qy=ly+(py-ly)/TC_PHI;
            int iC=b->v[0], iL=b->v[1], iP=b->v[2], iR=b->v[3];
            int iQ=_bv_obtener(verts,qx,qy);
            _bb_agregar(salida, TC_PENROSE_KITE, iQ, iC, iP, iR, b->angulo);
            _bb_agregar(salida, TC_PENROSE_DART, iL, iQ, iC, iP, b->angulo + ANG36*2.0f);
        }
    }
}
