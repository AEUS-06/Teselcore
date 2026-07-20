/* src/core/rng.c — Generador aleatorio xoshiro256** */
#include "../../include/internal/rng.h"
#include <math.h>
#include <stdint.h>

static uint64_t _estado[4] = {
    0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL,
    0x0F1E2D3C4B5A6978ULL, 0x8796A5B4C3D2E1F0ULL
};

static inline uint64_t _rotar(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t _siguiente(void) {
    uint64_t r = _rotar(_estado[1] * 5, 7) * 9;
    uint64_t t = _estado[1] << 17;
    _estado[2] ^= _estado[0]; _estado[3] ^= _estado[1];
    _estado[1] ^= _estado[2]; _estado[0] ^= _estado[3];
    _estado[2] ^= t;          _estado[3]  = _rotar(_estado[3], 45);
    return r;
}

float _uniforme(void) {
    return (float)(_siguiente() >> 11) * (1.0f / (float)(1ULL << 53));
}

float _normal(void) {
    float u1 = _uniforme() + 1e-7f, u2 = _uniforme();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

void tc_semilla_aleatoria(uint64_t s) {
    _estado[0] = s; _estado[1] = s ^ 0xDEADBEEFULL;
    _estado[2] = s ^ 0xCAFEBABEULL; _estado[3] = s ^ 0xFEEDFACEULL;
}
