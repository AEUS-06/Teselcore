#pragma once
#include <stdio.h>
#include <stdint.h>

#define ESCRIBIR_U8(f,v)  { uint8_t  _v=(uint8_t)(v);  fwrite(&_v,1,1,f); }
#define ESCRIBIR_U16(f,v) { uint16_t _v=(uint16_t)(v); fwrite(&_v,2,1,f); }
#define ESCRIBIR_U32(f,v) { uint32_t _v=(uint32_t)(v); fwrite(&_v,4,1,f); }
#define ESCRIBIR_U64(f,v) { uint64_t _v=(uint64_t)(v); fwrite(&_v,8,1,f); }

uint8_t  _leer_u8 (FILE* f);
uint16_t _leer_u16(FILE* f);
uint32_t _leer_u32(FILE* f);
uint64_t _leer_u64(FILE* f);
