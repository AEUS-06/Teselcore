/* src/penrose/interp.c — Interpolación bilineal (helper interno compartido) */
#include "../../include/internal/penrose_interno.h"

float _interpolar_bilineal(const float* canal, int alto, int ancho, float x, float y) {
    if (x < 0 || x >= ancho - 1 || y < 0 || y >= alto - 1) return 0.0f;
    int x0=(int)x, y0=(int)y;
    float dx=x-x0, dy=y-y0;
    return canal[y0*ancho+x0]      *(1-dx)*(1-dy)
         + canal[y0*ancho+x0+1]    *dx    *(1-dy)
         + canal[(y0+1)*ancho+x0]  *(1-dx)*dy
         + canal[(y0+1)*ancho+x0+1]*dx    *dy;
}
