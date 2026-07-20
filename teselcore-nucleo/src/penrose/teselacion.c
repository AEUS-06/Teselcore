/* src/penrose/teselacion.c — Orquesta la creación/liberación de la teselación */
#include "../../include/internal/penrose_interno.h"
#include <stdlib.h>
#include <stdio.h>

static void _poblar_tejas(tc_teselacion_penrose* ts, _BufVert* verts, _BufBald* actual,
                          int* neigh, int* n_neigh) {
    for (int i = 0; i < ts->num_tejas; i++) {
        tc_teja_penrose* tj = &ts->tejas[i];
        int* vs = actual->buf[i].v;
        float cx=0.0f, cy=0.0f;
        for (int p=0;p<4;p++) { cx+=verts->buf[vs[p]].x; cy+=verts->buf[vs[p]].y; }
        tj->centro_x=cx*0.25f; tj->centro_y=cy*0.25f;
        tj->num_vecinos=n_neigh[i];
        for (int k=0;k<n_neigh[i];k++) tj->vecinos[k]=neigh[i*7+k];
        tj->tipo = (tc_tipo_teja_penrose)actual->buf[i].tipo;
        tj->angulo = actual->buf[i].angulo;
        for (int p=0;p<4;p++) {
            tj->puntos[p][0]=verts->buf[vs[p]].x;
            tj->puntos[p][1]=verts->buf[vs[p]].y;
        }
    }
}

tc_teselacion_penrose* tc_crear_teselacion(int nivel, float escala) {
    _BufVert verts={0};
    _BufBald actual={0}, siguiente={0};
    _generar_semilla(&verts, &actual, escala);

    for (int n=0;n<nivel;n++) {
        siguiente.n=0;
        _subdividir(&verts, &actual, &siguiente);
        _BufBald tmp=actual; actual=siguiente; siguiente=tmp;
    }

    int T = actual.n;
    int* neigh   = (int*)calloc(T*7, sizeof(int));
    int* n_neigh = (int*)calloc(T, sizeof(int));
    _construir_vecindad(&actual, T, neigh, n_neigh);

    tc_teselacion_penrose* ts = (tc_teselacion_penrose*)malloc(sizeof(tc_teselacion_penrose));
    ts->num_tejas=T; ts->nivel=nivel; ts->escala=escala;
    ts->tejas=(tc_teja_penrose*)calloc(T, sizeof(tc_teja_penrose));
    _poblar_tejas(ts, &verts, &actual, neigh, n_neigh);

    free(neigh); free(n_neigh); free(verts.buf); free(actual.buf); free(siguiente.buf);
    printf("Teselación de Penrose nivel %d: %d nodos generados\n", nivel, T);
    return ts;
}

void tc_liberar_teselacion(tc_teselacion_penrose* t) {
    if (!t) return;
    free(t->tejas);
    free(t);
}
