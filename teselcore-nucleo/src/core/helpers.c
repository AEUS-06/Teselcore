/* src/core/helpers.c — Utilidades internas compartidas */
#include "../../include/internal/helpers.h"
#include <stdlib.h>
#include <string.h>

void _calcular_pasos(const int* forma, int ndim, int* pasos) {
    pasos[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        pasos[i] = pasos[i + 1] * forma[i + 1];
}

void _asegurar_gradiente(tc_tensor* t) {
    if (!t->gradiente)
        t->gradiente = (float*)calloc(t->total, sizeof(float));
}

void _acumular_grad(tc_tensor* t, const float* delta) {
    if (!t->requiere_grad) return;
    _asegurar_gradiente(t);
    for (size_t i = 0; i < t->total; i++)
        t->gradiente[i] += delta[i];
}
