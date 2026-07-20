/* src/cli/usage.c — Texto de ayuda del CLI */
#include <stdio.h>

void _usage(const char* prog) {
    printf("TeselCore CLI - comandos cortos y portables\n");
    printf("Uso: %s <comando> [args]\n\n", prog);
    printf("Comandos:\n");
    printf("  help                 Muestra esta ayuda\n");
    printf("  info                 Información del runtime\n");
    printf("  gen <nivel> <svg>    Genera teselación de Penrose y la guarda en <svg>\n");
    printf("  save_demo <ruta.ax>  Crea y guarda un modelo demo (.ax)\n");
    printf("  load <ruta.ax>       Carga un modelo .ax y lista tensores\n");
    printf("  conv_demo <nivel>    Ejecuta demo de conv_penrose_grafo (nivel)\n");
}
