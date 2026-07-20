#pragma once
#include <stdio.h>
#include "../teselcore.h"
/* guardar_cuerpo.c / guardar_crc.c */
void _escribir_cuerpo_modelo(FILE* archivo, const tc_modelo* modelo);
int  _finalizar_con_crc(const char* ruta);
/* cargar_validar.c / cargar_tensor.c */
int  _validar_crc_archivo(const char* ruta);
int  _leer_tensor_nombrado(FILE* archivo, tc_tensor_nombrado* tn);
