/* src/modelo/guardar_cuerpo.c — Escribe encabezado + tensores de un modelo */
#include "../../include/internal/modelo_interno.h"
#include "../../include/internal/io_binario.h"
#include <string.h>

void _escribir_cuerpo_modelo(FILE* archivo, const tc_modelo* modelo) {
    fwrite("AXON", 1, 4, archivo);
    ESCRIBIR_U16(archivo, modelo->version_mayor);
    ESCRIBIR_U16(archivo, modelo->version_menor);
    ESCRIBIR_U32(archivo, 0);

    uint32_t lm = modelo->metadatos ? (uint32_t)strlen(modelo->metadatos) : 0;
    ESCRIBIR_U32(archivo, lm);
    if (lm > 0) fwrite(modelo->metadatos, 1, lm, archivo);
    ESCRIBIR_U32(archivo, (uint32_t)modelo->num_tensores);

    for (int i = 0; i < modelo->num_tensores; i++) {
        const tc_tensor_nombrado* tn = &modelo->tensores[i];
        tc_tensor* t = tn->tensor;
        uint16_t ln = (uint16_t)strlen(tn->nombre);
        ESCRIBIR_U16(archivo, ln);
        fwrite(tn->nombre, 1, ln, archivo);
        ESCRIBIR_U8(archivo, (uint8_t)t->tipo);
        ESCRIBIR_U8(archivo, (uint8_t)t->ndim);
        for (int d = 0; d < t->ndim; d++) ESCRIBIR_U32(archivo, (uint32_t)t->forma[d]);
        uint64_t bytes = (uint64_t)(t->total * TC_BYTES_POR_TIPO[t->tipo]);
        ESCRIBIR_U64(archivo, bytes);
        fwrite(t->datos, 1, (size_t)bytes, archivo);
    }
}
