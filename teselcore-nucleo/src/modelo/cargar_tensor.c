/* src/modelo/cargar_tensor.c — Lee un tensor_nombrado del stream binario */
#include "../../include/internal/modelo_interno.h"
#include "../../include/internal/io_binario.h"
#include <stdlib.h>

int _leer_tensor_nombrado(FILE* archivo, tc_tensor_nombrado* tn) {
    uint16_t ln = _leer_u16(archivo);
    if (ln >= (uint16_t)sizeof(tn->nombre)) {
        size_t leer = sizeof(tn->nombre) - 1;
        if (fread(tn->nombre, 1, leer, archivo) != leer) return -1;
        tn->nombre[leer] = '\0';
        long saltar = (long)ln - (long)leer;
        if (saltar > 0) fseek(archivo, saltar, SEEK_CUR);
    } else {
        if (fread(tn->nombre, 1, ln, archivo) != ln) return -1;
        tn->nombre[ln] = '\0';
    }

    tc_tipo_dato tipo = (tc_tipo_dato)_leer_u8(archivo);
    int ndim = (int)_leer_u8(archivo);
    int forma[TC_MAX_DIMS];
    for (int d = 0; d < ndim; d++) forma[d] = (int)_leer_u32(archivo);

    uint64_t bytes = _leer_u64(archivo);
    void* datos = malloc((size_t)bytes);
    if (fread(datos, 1, (size_t)bytes, archivo) != (size_t)bytes) { free(datos); return -1; }
    tn->tensor = tc_desde_datos(datos, forma, ndim, tipo);
    free(datos);
    return 0;
}
