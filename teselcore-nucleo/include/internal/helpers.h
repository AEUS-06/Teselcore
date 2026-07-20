#pragma once
#include "../../include/teselcore.h"
void _calcular_pasos(const int* forma, int ndim, int* pasos);
void _asegurar_gradiente(tc_tensor* t);
void _acumular_grad(tc_tensor* t, const float* delta);
