/* src/cli/cmd_gen.c — Comando: gen <nivel> <svg> */
#include "../../include/teselcore.h"
#include <stdio.h>
#include <stdlib.h>

int _cmd_gen(int argc, char** argv, const char* prog) {
    if (argc < 4) { fprintf(stderr, "Uso: %s gen <nivel> <svg>\n", prog); return 1; }
    int nivel = atoi(argv[2]);
    const char* svg = argv[3];
    tc_teselacion_penrose* t = tc_crear_teselacion(nivel, 100.0f);
    if (!t) { fprintf(stderr, "Error creando teselacion\n"); return 1; }
    int r = tc_exportar_svg_teselacion(t, svg);
    tc_liberar_teselacion(t);
    if (r == 0) { printf("SVG guardado: %s\n", svg); return 0; }
    fprintf(stderr, "Error exportando SVG\n");
    return 1;
}
