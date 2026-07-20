/* src/penrose/svg_export.c — Exportación de la teselación a SVG */
#include "../../include/teselcore.h"
#include <stdio.h>

static void _bbox_puntos(const tc_teselacion_penrose* t, float* xmin, float* xmax,
                         float* ymin, float* ymax) {
    *xmin=1e9f; *xmax=-1e9f; *ymin=1e9f; *ymax=-1e9f;
    for (int i=0;i<t->num_tejas;i++) for (int v=0;v<4;v++) {
        float px=t->tejas[i].puntos[v][0], py=t->tejas[i].puntos[v][1];
        if (px<*xmin) *xmin=px;
        if (px>*xmax) *xmax=px;
        if (py<*ymin) *ymin=py;
        if (py>*ymax) *ymax=py;
    }
}

int tc_exportar_svg_teselacion(const tc_teselacion_penrose* t, const char* ruta) {
    FILE* f = fopen(ruta, "w");
    if (!f) return -1;

    float xmin,xmax,ymin,ymax;
    _bbox_puntos(t, &xmin,&xmax,&ymin,&ymax);
    float margen=(xmax-xmin)*0.05f;
    float vx=xmin-margen, vy=ymin-margen, vw=(xmax-xmin)+2*margen, vh=(ymax-ymin)+2*margen;

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"%.2f %.2f %.2f %.2f\" width=\"800\" height=\"800\">\n", vx,vy,vw,vh);
    fprintf(f, "  <rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#0f0f1a\"/>\n", vx,vy,vw,vh);

    for (int i=0;i<t->num_tejas;i++) {
        const tc_teja_penrose* tj=&t->tejas[i];
        const char* cf = tj->tipo==TC_PENROSE_KITE ? "#1a3a5c" : "#3a1a0f";
        const char* cs = tj->tipo==TC_PENROSE_KITE ? "#00e5ff" : "#ff6b35";
        fprintf(f, "  <polygon points=\"");
        for (int v=0;v<4;v++) fprintf(f, "%.3f,%.3f%s", tj->puntos[v][0], tj->puntos[v][1], v<3?" ":"");
        fprintf(f, "\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.3\" opacity=\"0.85\"/>\n", cf, cs);
        fprintf(f, "  <circle cx=\"%.3f\" cy=\"%.3f\" r=\"0.15\" fill=\"%s\" opacity=\"0.6\"/>\n", tj->centro_x, tj->centro_y, cs);
    }
    fprintf(f, "</svg>\n");
    fclose(f);
    printf("SVG exportado: %s (%d nodos)\n", ruta, t->num_tejas);
    return 0;
}
