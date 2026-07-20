/* src/optimizador/sgd.c — Descenso de gradiente estocástico con momento */
#include "../../include/internal/optimizador_interno.h"
#include <stdlib.h>

tc_optimizador* tc_sgd(tc_tensor** params, int n, float lr, float momento) {
    tc_optimizador* opt = (tc_optimizador*)calloc(1, sizeof(tc_optimizador));
    opt->parametros = params;
    opt->num_params = n;
    opt->lr = lr;
    opt->momento = momento;
    opt->tipo = OPT_SGD;
    opt->velocidades = (float**)calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++)
        opt->velocidades[i] = (float*)calloc(params[i]->total, sizeof(float));
    return opt;
}
