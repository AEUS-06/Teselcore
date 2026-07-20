/* src/modelo/io_binario.c — Primitivas de lectura binaria little-endian */
#include "../../include/internal/io_binario.h"

uint8_t  _leer_u8 (FILE* f) { uint8_t  v=0; if (fread(&v,1,1,f)!=1) return 0; return v; }
uint16_t _leer_u16(FILE* f) { uint16_t v=0; if (fread(&v,2,1,f)!=1) return 0; return v; }
uint32_t _leer_u32(FILE* f) { uint32_t v=0; if (fread(&v,4,1,f)!=1) return 0; return v; }
uint64_t _leer_u64(FILE* f) { uint64_t v=0; if (fread(&v,8,1,f)!=1) return 0; return v; }
