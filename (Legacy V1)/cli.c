/* src/cli.c
 * Pequeño CLI portátil para TeselCore (comandos cortos, C puro)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "teselcore.h"

static void usage(const char* prog) {
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

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "h") == 0) {
        usage(argv[0]);
        return 0;
    }

    if (strcmp(cmd, "info") == 0) {
        tc_imprimir_info();
        return 0;
    }

    if (strcmp(cmd, "gen") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Uso: %s gen <nivel> <svg>\n", argv[0]);
            return 1;
        }
        int nivel = atoi(argv[2]);
        float escala = 100.0f;
        const char* svg = argv[3];
        tc_teselacion_penrose* t = tc_crear_teselacion(nivel, escala);
        if (!t) {
            fprintf(stderr, "Error creando teselacion\n");
            return 1;
        }
        int r = tc_exportar_svg_teselacion(t, svg);
        tc_liberar_teselacion(t);
        if (r) {
            printf("SVG guardado: %s\n", svg);
            return 0;
        }
        fprintf(stderr, "Error exportando SVG\n");
        return 1;
    }

    if (strcmp(cmd, "save_demo") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Uso: %s save_demo <ruta.ax>\n", argv[0]);
            return 1;
        }
        const char* ruta = argv[2];
        tc_modelo* m = tc_modelo_nuevo("{\"name\":\"demo\"}");
        if (!m) {
            fprintf(stderr, "No se pudo crear modelo\n");
            return 1;
        }
        int forma[2] = {4, 4};
        tc_tensor* t = tc_ceros(forma, 2);
        tc_modelo_agregar_tensor(m, "demo/tensor", t);
        int ok = tc_guardar(m, ruta);
        tc_liberar_modelo(m);
        if (ok == 0) {
            printf("Modelo demo guardado: %s\n", ruta);
            return 0;
        }
        fprintf(stderr, "Error guardando modelo\n");
        return 1;
    }

    if (strcmp(cmd, "load") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Uso: %s load <ruta.ax>\n", argv[0]);
            return 1;
        }
        const char* ruta = argv[2];
        tc_modelo* m = tc_cargar(ruta);
        if (!m) {
            fprintf(stderr, "Error cargando modelo: %s\n", ruta);
            return 1;
        }
        printf("Modelo cargado: %d tensores\n", m->num_tensores);
        for (int i = 0; i < m->num_tensores; ++i) {
            printf(" - %s\n", m->tensores[i].nombre);
            if (m->tensores[i].tensor) tc_imprimir(m->tensores[i].tensor);
        }
        tc_liberar_modelo(m);
        return 0;
    }

    if (strcmp(cmd, "conv_demo") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Uso: %s conv_demo <nivel>\n", argv[0]);
            return 1;
        }
        int nivel = atoi(argv[2]);
        float escala = 1.0f;
        tc_teselacion_penrose* t = tc_crear_teselacion(nivel, escala);
        if (!t) {
            fprintf(stderr, "Error creando teselacion\n");
            return 1;
        }
        int canales_ent = 1, canales_sal = 1;
        tc_kernel_penrose* k = tc_crear_kernel_penrose(canales_ent, canales_sal, nivel, escala);
        if (!k) {
            fprintf(stderr, "Error creando kernel penrose\n");
            tc_liberar_teselacion(t);
            return 1;
        }
        int forma_input[3];
        forma_input[0] = 1;
        forma_input[1] = canales_ent;
        forma_input[2] = t->num_tejas;
        tc_tensor* entrada = tc_aleatorio_normal(forma_input, 3);
        tc_tensor* salida = tc_conv_penrose_grafo(entrada, k->pesos, k->sesgo, t);
        printf("Salida de conv_demo:\n");
        if (salida) tc_imprimir(salida);
        tc_liberar_kernel_penrose(k);
        tc_liberar_teselacion(t);
        tc_liberar(entrada);
        tc_liberar(salida);
        return 0;
    }

    fprintf(stderr, "Comando desconocido: %s\n", cmd);
    usage(argv[0]);
    return 1;
}
