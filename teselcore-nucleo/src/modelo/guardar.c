/* src/modelo/guardar.c — tc_guardar: orquesta cuerpo + CRC */
#include "../../include/internal/modelo_interno.h"
#include <stdio.h>

int tc_guardar(const tc_modelo* modelo, const char* ruta) {
    FILE* archivo = fopen(ruta, "wb");
    if (!archivo) { perror("tc_guardar: no se pudo abrir el archivo"); return -1; }

    _escribir_cuerpo_modelo(archivo, modelo);
    long posicion = ftell(archivo);
    fclose(archivo);

    if (_finalizar_con_crc(ruta) != 0) return -1;

    printf("tc_guardar: '%s' guardado (%.2f KB)\n", ruta, (posicion + 4) / 1024.0);
    return 0;
}
