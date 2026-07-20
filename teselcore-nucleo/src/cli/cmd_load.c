/* src/cli/cmd_load.c — Comando: load <ruta.ax> */
#include "../../include/teselcore.h"
#include <stdio.h>

int _cmd_load(int argc, char** argv, const char* prog) {
    if (argc < 3) { fprintf(stderr, "Uso: %s load <ruta.ax>\n", prog); return 1; }
    const char* ruta = argv[2];
    tc_modelo* m = tc_cargar(ruta);
    if (!m) { fprintf(stderr, "Error cargando modelo: %s\n", ruta); return 1; }

    printf("Modelo cargado: %d tensores\n", m->num_tensores);
    for (int i = 0; i < m->num_tensores; i++) {
        printf(" - %s\n", m->tensores[i].nombre);
        if (m->tensores[i].tensor) tc_imprimir(m->tensores[i].tensor);
    }
    tc_liberar_modelo(m);
    return 0;
}
