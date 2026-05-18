/* tests/test_basico.c */
#include <stdio.h>
#include <stdlib.h>
#include "../include/teselcore.h"

int main(void) {
    tc_imprimir_info();
    tc_semilla_aleatoria(12345ULL);

    int nivel = 2;
    float escala = 6.0f;
    int Cin = 3, Cout = 4;

    tc_kernel_penrose* k = tc_crear_kernel_penrose(Cin, Cout, nivel, escala);
    if (!k) { fprintf(stderr, "Error creando kernel\n"); return 1; }

    int N = k->teselacion->num_tejas;
    printf("Teselación nodos: %d\n", N);

    int forma_in[3] = {1, Cin, N};
    tc_tensor* entrada = tc_aleatorio_uniforme(forma_in, 3);
    entrada->requiere_grad = 1;
    k->pesos->requiere_grad = 1;

    tc_tensor* salida = tc_conv_penrose_grafo(entrada, k->pesos, k->sesgo, k->teselacion);
    if (!salida) { fprintf(stderr, "Error en forward\n"); return 1; }

    tc_tensor* perdida = tc_suma_dimension(salida, -1, 0);
    printf("Loss forward: %.6f\n", tc_elemento(perdida));

    tc_retropropagar(perdida);

    if (k->pesos->gradiente) printf("Grad kernel[0]=%.6f\n", k->pesos->gradiente[0]);
    if (entrada->gradiente)  printf("Grad entrada[0]=%.6f\n", entrada->gradiente[0]);
    if (k->sesgo->gradiente) printf("Grad sesgo[0]=%.6f\n",   k->sesgo->gradiente[0]);

    /* Guardar y cargar modelo */
    const char* ruta_modelo = "test_model.ax";

    tc_modelo* m = tc_modelo_nuevo("{\"desc\":\"test\"}");
    tc_modelo_agregar_tensor(m, "kernel_pesos", k->pesos);
    tc_modelo_agregar_tensor(m, "kernel_sesgo", k->sesgo);
    tc_guardar(m, ruta_modelo);

    /* BUG FIX en test: no liberar m con tc_liberar_modelo porque eso libera
       k->pesos y k->sesgo que todavía son propiedad del kernel. En vez de eso,
       liberar solo la estructura del modelo sin tocar los tensores. */
    free(m->tensores);
    free(m->metadatos);
    free(m);

    tc_modelo* m2 = tc_cargar(ruta_modelo);
    if (m2) {
        printf("Modelo cargado: %d tensores\n", m2->num_tensores);
        tc_liberar_modelo(m2);
    }

    remove(ruta_modelo);

    tc_liberar(entrada);
    tc_liberar(salida);
    tc_liberar(perdida);
    tc_liberar_kernel_penrose(k);

    printf("Prueba completada OK\n");
    return 0;
}