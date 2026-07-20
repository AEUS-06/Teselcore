/* src/cli/cmd_save_demo.c — Comando: save_demo <ruta.ax> */
#include "../../include/teselcore.h"
#include <stdio.h>

int _cmd_save_demo(int argc, char** argv, const char* prog) {
    if (argc < 3) { fprintf(stderr, "Uso: %s save_demo <ruta.ax>\n", prog); return 1; }
    const char* ruta = argv[2];
    tc_modelo* m = tc_modelo_nuevo("{\"name\":\"demo\"}");
    if (!m) { fprintf(stderr, "No se pudo crear modelo\n"); return 1; }

    int forma[2] = {4, 4};
    tc_tensor* t = tc_ceros(forma, 2);
    tc_modelo_agregar_tensor(m, "demo/tensor", t);
    int ok = tc_guardar(m, ruta);
    tc_liberar_modelo(m);
    if (ok == 0) { printf("Modelo demo guardado: %s\n", ruta); return 0; }
    fprintf(stderr, "Error guardando modelo\n");
    return 1;
}
