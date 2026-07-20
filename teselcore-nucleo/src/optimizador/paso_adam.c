/* src/optimizador/paso_adam.c — Un paso de actualización Adam / AdamW */
#include "../../include/internal/optimizador_interno.h"
#include <math.h>

void _paso_adam(tc_optimizador* opt, tc_tensor* p, float* m1, float* m2) {
    float* d = (float*)p->datos;
    float* g = p->gradiente;
    float c1 = 1.0f - powf(opt->beta1, (float)opt->paso);
    float c2 = 1.0f - powf(opt->beta2, (float)opt->paso);
    float lr_corr = opt->lr * sqrtf(c2) / c1;

    for (size_t j = 0; j < p->total; j++) {
        m1[j] = opt->beta1 * m1[j] + (1.0f - opt->beta1) * g[j];
        m2[j] = opt->beta2 * m2[j] + (1.0f - opt->beta2) * g[j] * g[j];
        float upd = lr_corr * m1[j] / (sqrtf(m2[j]) + opt->eps);
        if (opt->tipo == OPT_ADAMW) upd += opt->lr * opt->decaimiento_pesos * d[j];
        d[j] -= upd;
    }
}
