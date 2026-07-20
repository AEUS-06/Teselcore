/* src/modelo/crc32.c — CRC32 estándar para verificación de integridad .ax */
#include "../../include/internal/crc32.h"

static uint32_t _tabla_crc32[256];
static int      _tabla_inicializada = 0;

static void _construir_tabla_crc32(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
        _tabla_crc32[i] = c;
    }
    _tabla_inicializada = 1;
}

uint32_t _calcular_crc32(const uint8_t* datos, size_t longitud) {
    if (!_tabla_inicializada) _construir_tabla_crc32();
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < longitud; i++)
        crc = _tabla_crc32[(crc ^ datos[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFU;
}
