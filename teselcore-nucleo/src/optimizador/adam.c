/* src/optimizador/adam.c — Adam y AdamW (comparten estado de primer/segundo momento) */
#include "../../include/internal/optimizador_interno.h"
#include <stdlib.h>

tc_optimizador* tc_adam(tc_tensor** params, int n, float lr,
                        float beta1, float beta2, float eps) {
    tc_optimizador* opt = (tc_optimizador*)calloc(1, sizeof(tc_optimizador));
    opt->parametros = params; opt->num_params = n;
    opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->tipo = OPT_ADAM;
    opt->momento1 = (float**)calloc(n, sizeof(float*));
    opt->momento2 = (float**)calloc(n, sizeof(float*));
    for (int i = 0; i < n; i++) {
        opt->momento1[i] = (float*)calloc(params[i]->total, sizeof(float));
        opt->momento2[i] = (float*)calloc(params[i]->total, sizeof(float));
    }
    return opt;
}

tc_optimizador* tc_adamw(tc_tensor** params, int n, float lr,
                         float beta1, float beta2, float eps, float decaimiento) {
    tc_optimizador* opt = tc_adam(params, n, lr, beta1, beta2, eps);
    opt->tipo = OPT_ADAMW;
    opt->decaimiento_pesos = decaimiento;
    return opt;
}
