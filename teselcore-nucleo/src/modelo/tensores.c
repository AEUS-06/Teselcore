/* src/modelo/tensores.c — Agregar / obtener tensores nombrados del modelo */
#include "../../include/teselcore.h"
#include <stdlib.h>
#include <string.h>

int tc_modelo_agregar_tensor(tc_modelo* modelo, const char* nombre, tc_tensor* t) {
    modelo->tensores = (tc_tensor_nombrado*)realloc(
        modelo->tensores, (size_t)(modelo->num_tensores + 1) * sizeof(tc_tensor_nombrado));
    tc_tensor_nombrado* tn = &modelo->tensores[modelo->num_tensores++];
    strncpy(tn->nombre, nombre, 255);
    tn->nombre[255] = '\0';
    tn->tensor = t;
    return 0;
}

tc_tensor* tc_modelo_obtener_tensor(const tc_modelo* modelo, const char* nombre) {
    for (int i = 0; i < modelo->num_tensores; i++)
        if (strcmp(modelo->tensores[i].nombre, nombre) == 0)
            return modelo->tensores[i].tensor;
    return NULL;
}
