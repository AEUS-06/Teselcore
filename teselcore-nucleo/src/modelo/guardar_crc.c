/* src/modelo/guardar_crc.c — Relee el archivo y añade el CRC32 final */
#include "../../include/internal/modelo_interno.h"
#include "../../include/internal/crc32.h"
#include <stdlib.h>

int _finalizar_con_crc(const char* ruta) {
    FILE* fr = fopen(ruta, "rb");
    if (!fr) { perror("tc_guardar: no se pudo reabrir para CRC"); return -1; }
    fseek(fr, 0, SEEK_END);
    long tam = ftell(fr);
    fseek(fr, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)tam);
    if (!buf) { fclose(fr); return -1; }
    if (fread(buf, 1, (size_t)tam, fr) != (size_t)tam) { free(buf); fclose(fr); return -1; }
    uint32_t crc = _calcular_crc32(buf, (size_t)tam);
    free(buf); fclose(fr);

    FILE* fa = fopen(ruta, "ab");
    if (!fa) { perror("tc_guardar: no se pudo abrir para append"); return -1; }
    uint32_t v = crc;
    fwrite(&v, 4, 1, fa);
    fclose(fa);
    return 0;
}
