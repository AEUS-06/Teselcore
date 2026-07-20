#pragma once
#include "../teselcore.h"

#define PI_TC 3.14159265358979323846f
#define ANG36 (PI_TC / 5.0f)
#define ANG72 (2.0f * PI_TC / 5.0f)

typedef struct { float x, y; } _Vertice;
typedef struct { int tipo; int v[4]; float angulo; } _Baldosa;
typedef struct { _Vertice* buf; int n, cap; } _BufVert;
typedef struct { _Baldosa* buf; int n, cap; } _BufBald;

/* buffers.c */
void _bv_agregar(_BufVert* bv, float x, float y);
int  _bv_obtener(_BufVert* bv, float x, float y);
void _bb_agregar(_BufBald* bb, int tipo, int v0, int v1, int v2, int v3, float ang);

/* subdivide.c */
void _subdividir(_BufVert* verts, _BufBald* entrada, _BufBald* salida);

/* semilla.c */
void _generar_semilla(_BufVert* verts, _BufBald* actual, float escala);

/* vecindad.c */
void _construir_vecindad(_BufBald* tejas, int T, int* neigh, int* n_neigh);

/* interp.c */
float _interpolar_bilineal(const float* canal, int alto, int ancho, float x, float y);

/* bbox.c */
void _calcular_bbox_teselacion(const tc_teselacion_penrose* t,
                                float* xmin, float* xmax, float* ymin, float* ymax);
