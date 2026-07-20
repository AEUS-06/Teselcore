/* src/penrose/buffers.c — Buffers dinámicos de vértices y baldosas */
#include "../../include/internal/penrose_interno.h"
#include <stdlib.h>

void _bv_agregar(_BufVert* bv, float x, float y) {
    if (bv->n >= bv->cap) {
        bv->cap = bv->cap ? bv->cap * 2 : 64;
        bv->buf = (_Vertice*)realloc(bv->buf, bv->cap * sizeof(_Vertice));
    }
    bv->buf[bv->n++] = (_Vertice){x, y};
}

int _bv_obtener(_BufVert* bv, float x, float y) {
    const float EPS = 1e-5f;
    for (int i = 0; i < bv->n; i++) {
        float dx = bv->buf[i].x - x, dy = bv->buf[i].y - y;
        if (dx*dx + dy*dy < EPS*EPS) return i;
    }
    _bv_agregar(bv, x, y);
    return bv->n - 1;
}

void _bb_agregar(_BufBald* bb, int tipo, int v0, int v1, int v2, int v3, float ang) {
    if (bb->n >= bb->cap) {
        bb->cap = bb->cap ? bb->cap * 2 : 64;
        bb->buf = (_Baldosa*)realloc(bb->buf, bb->cap * sizeof(_Baldosa));
    }
    bb->buf[bb->n++] = (_Baldosa){tipo, {v0, v1, v2, v3}, ang};
}
