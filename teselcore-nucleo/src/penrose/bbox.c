/* src/penrose/bbox.c — Caja delimitadora de los centros de las tejas */
#include "../../include/internal/penrose_interno.h"

void _calcular_bbox_teselacion(const tc_teselacion_penrose* t,
                                float* xmin, float* xmax, float* ymin, float* ymax) {
    *xmin=1e9f; *xmax=-1e9f; *ymin=1e9f; *ymax=-1e9f;
    for (int i = 0; i < t->num_tejas; i++) {
        float x = t->tejas[i].centro_x, y = t->tejas[i].centro_y;
        if (x < *xmin) *xmin = x;
        if (x > *xmax) *xmax = x;
        if (y < *ymin) *ymin = y;
        if (y > *ymax) *ymax = y;
    }
}
