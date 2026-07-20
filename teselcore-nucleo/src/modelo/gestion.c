/* src/modelo/gestion.c — Ciclo de vida del modelo (nuevo / liberar) */
#define _POSIX_C_SOURCE 200809L
#include "../../include/teselcore.h"
#include <stdlib.h>
#include <string.h>

tc_modelo* tc_modelo_nuevo(const char* metadatos_json) {
    tc_modelo* m = (tc_modelo*)calloc(1, sizeof(tc_modelo));
    m->version_mayor = TC_VERSION_MAYOR;
    m->version_menor = TC_VERSION_MENOR;
    if (metadatos_json) m->metadatos = strdup(metadatos_json);
    return m;
}

void tc_liberar_modelo(tc_modelo* modelo) {
    if (!modelo) return;
    for (int i = 0; i < modelo->num_tensores; i++)
        tc_liberar(modelo->tensores[i].tensor);
    free(modelo->tensores);
    free(modelo->metadatos);
    free(modelo);
}

/* Libera solo la estructura del modelo (arreglo de tensores_nombrados +
   metadatos + el struct en sí) SIN liberar los tensores apuntados.
   Útil cuando los tensores agregados con tc_modelo_agregar_tensor
   siguen siendo propiedad de otra estructura (p.ej. un kernel de Penrose)
   que los liberará por su cuenta. Evita double-free en ese escenario. */
void tc_liberar_modelo_estructura(tc_modelo* modelo) {
    if (!modelo) return;
    free(modelo->tensores);
    free(modelo->metadatos);
    free(modelo);
}
