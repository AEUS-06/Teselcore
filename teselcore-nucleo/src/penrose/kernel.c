/* src/penrose/kernel.c — Kernel de Penrose: pesos + sesgo sobre una teselación */
#include "../../include/teselcore.h"
#include <stdlib.h>
#include <math.h>

tc_kernel_penrose* tc_crear_kernel_penrose(int canales_ent, int canales_sal,
                                            int nivel, float escala) {
    tc_kernel_penrose* k = (tc_kernel_penrose*)malloc(sizeof(tc_kernel_penrose));
    k->teselacion      = tc_crear_teselacion(nivel, escala);
    k->canales_entrada  = canales_ent;
    k->canales_salida   = canales_sal;

    int num_tejas = k->teselacion->num_tejas;
    int forma_pesos[] = {canales_sal, canales_ent, num_tejas};
    k->pesos = tc_aleatorio_normal(forma_pesos, 3);
    k->pesos->requiere_grad = 1;

    float std_he = sqrtf(2.0f / (canales_ent * num_tejas));
    float* d = (float*)k->pesos->datos;
    for (size_t i = 0; i < k->pesos->total; i++) d[i] *= std_he;

    int forma_sesgo[] = {canales_sal};
    k->sesgo = tc_ceros(forma_sesgo, 1);
    k->sesgo->requiere_grad = 1;
    return k;
}

void tc_liberar_kernel_penrose(tc_kernel_penrose* k) {
    if (!k) return;
    tc_liberar_teselacion(k->teselacion);
    tc_liberar(k->pesos);
    tc_liberar(k->sesgo);
    free(k);
}
