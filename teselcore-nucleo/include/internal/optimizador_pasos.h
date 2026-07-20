#pragma once
#include "optimizador_interno.h"
void _paso_sgd (tc_optimizador* opt, tc_tensor* p, float* v);
void _paso_adam(tc_optimizador* opt, tc_tensor* p, float* m1, float* m2);
