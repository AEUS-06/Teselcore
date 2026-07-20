/* src/optimizador/gestion.c — Cero-gradientes y liberación del optimizador */
#include "../../include/internal/optimizador_interno.h"
#include <stdlib.h>

void tc_cero_gradientes_optimizador(tc_optimizador* opt) {
    for (int i = 0; i < opt->num_params; i++)
        tc_cero_gradiente(opt->parametros[i]);
}

void tc_liberar_optimizador(tc_optimizador* opt) {
    if (!opt) return;
    for (int i = 0; i < opt->num_params; i++) {
        if (opt->velocidades) free(opt->velocidades[i]);
        if (opt->momento1)    free(opt->momento1[i]);
        if (opt->momento2)    free(opt->momento2[i]);
    }
    free(opt->velocidades);
    free(opt->momento1);
    free(opt->momento2);
    free(opt);
}
