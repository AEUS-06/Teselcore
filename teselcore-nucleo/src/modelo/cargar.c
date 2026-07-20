/* src/modelo/cargar.c — tc_cargar: valida CRC, lee encabezado y tensores */
#include "../../include/internal/modelo_interno.h"
#include "../../include/internal/io_binario.h"
#include <stdlib.h>
#include <string.h>

tc_modelo* tc_cargar(const char* ruta) {
    if (_validar_crc_archivo(ruta) != 0) {
        fprintf(stderr, "tc_cargar: CRC inválido o archivo inaccesible\n");
        return NULL;
    }
    FILE* archivo = fopen(ruta, "rb");
    if (!archivo) { perror("tc_cargar"); return NULL; }

    char magic[4];
    if (fread(magic, 1, 4, archivo) != 4) { fclose(archivo); return NULL; }
    if (memcmp(magic, "AXON", 4) != 0) {
        fprintf(stderr, "tc_cargar: no es un archivo .ax válido\n");
        fclose(archivo); return NULL;
    }

    tc_modelo* modelo = (tc_modelo*)calloc(1, sizeof(tc_modelo));
    modelo->version_mayor = _leer_u16(archivo);
    modelo->version_menor = _leer_u16(archivo);
    _leer_u32(archivo); /* banderas */

    uint32_t lm = _leer_u32(archivo);
    if (lm > 0) {
        modelo->metadatos = (char*)malloc(lm + 1);
        if (fread(modelo->metadatos, 1, lm, archivo) != lm) { fclose(archivo); return NULL; }
        modelo->metadatos[lm] = '\0';
    }

    uint32_t num = _leer_u32(archivo);
    modelo->num_tensores = (int)num;
    modelo->tensores = (tc_tensor_nombrado*)calloc(num, sizeof(tc_tensor_nombrado));
    for (uint32_t i = 0; i < num; i++) _leer_tensor_nombrado(archivo, &modelo->tensores[i]);

    fclose(archivo);
    printf("tc_cargar: '%s' cargado (%d tensores)\n", ruta, modelo->num_tensores);
    return modelo;
}
