/* src/modelo/cargar_validar.c — Verifica el CRC32 de un archivo .ax */
#include "../../include/internal/modelo_interno.h"
#include "../../include/internal/crc32.h"
#include <stdlib.h>
#include <string.h>

int _validar_crc_archivo(const char* ruta) {
    FILE* f = fopen(ruta, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long tam = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (tam < 4) { fclose(f); return -1; }

    uint8_t* contenido = (uint8_t*)malloc((size_t)tam);
    if (!contenido) { fclose(f); return -1; }
    if (fread(contenido, 1, (size_t)tam, f) != (size_t)tam) { free(contenido); fclose(f); return -1; }
    fclose(f);

    uint32_t guardado, calculado;
    memcpy(&guardado, contenido + tam - 4, 4);
    calculado = _calcular_crc32(contenido, (size_t)(tam - 4));
    free(contenido);
    return (guardado == calculado) ? 0 : -1;
}
