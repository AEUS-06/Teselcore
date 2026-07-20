/* src/cli/cmd_conv_demo.c — Comando: conv_demo <nivel> */
#include "../../include/teselcore.h"
#include <stdio.h>
#include <stdlib.h>

int _cmd_conv_demo(int argc, char** argv, const char* prog) {
    if (argc < 3) { fprintf(stderr, "Uso: %s conv_demo <nivel>\n", prog); return 1; }
    int nivel = atoi(argv[2]);
    float escala = 1.0f;

    tc_teselacion_penrose* t = tc_crear_teselacion(nivel, escala);
    if (!t) { fprintf(stderr, "Error creando teselacion\n"); return 1; }

    tc_kernel_penrose* k = tc_crear_kernel_penrose(1, 1, nivel, escala);
    if (!k) { fprintf(stderr, "Error creando kernel penrose\n"); tc_liberar_teselacion(t); return 1; }

    int forma_in[3] = {1, 1, t->num_tejas};
    tc_tensor* entrada = tc_aleatorio_normal(forma_in, 3);
    tc_tensor* salida = tc_conv_penrose_grafo(entrada, k->pesos, k->sesgo, t);

    printf("Salida de conv_demo:\n");
    if (salida) tc_imprimir(salida);

    tc_liberar_kernel_penrose(k);
    tc_liberar_teselacion(t);
    tc_liberar(entrada);
    tc_liberar(salida);
    return 0;
}
