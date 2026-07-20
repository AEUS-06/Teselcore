#pragma once
#include "../teselcore.h"

typedef enum { OPT_SGD=0, OPT_ADAM=1, OPT_ADAMW=2 } _tipo_opt;

struct tc_optimizador {
    tc_tensor** parametros;
    int         num_params;
    float       lr;
    _tipo_opt   tipo;

    float   momento;
    float** velocidades;

    float   beta1, beta2, eps, decaimiento_pesos;
    float** momento1;
    float** momento2;
    int     paso;
};
