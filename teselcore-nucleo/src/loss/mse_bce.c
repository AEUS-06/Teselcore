/* src/loss/mse_bce.c — Error cuadrático medio y entropía cruzada binaria */
#include "../../include/teselcore.h"
#include "../../include/internal/helpers.h"
#include <math.h>
#include <assert.h>

tc_tensor* tc_error_cuadratico_medio(tc_tensor* pred, tc_tensor* objetivo) {
    assert(pred->total==objetivo->total);
    float *dp=(float*)pred->datos, *do_=(float*)objetivo->datos, perdida=0.0f;
    for (size_t i=0;i<pred->total;i++) { float d=dp[i]-do_[i]; perdida+=d*d; }
    perdida/=(float)pred->total;
    if (pred->requiere_grad) {
        _asegurar_gradiente(pred);
        float e=2.0f/(float)pred->total;
        for (size_t i=0;i<pred->total;i++) pred->gradiente[i]+=e*(dp[i]-do_[i]);
    }
    int forma[]={1};
    return tc_desde_datos(&perdida, forma, 1, TC_FLOAT32);
}

tc_tensor* tc_entropia_cruzada_binaria(tc_tensor* pred, tc_tensor* objetivo) {
    assert(pred->total==objetivo->total);
    float *dp=(float*)pred->datos, *do_=(float*)objetivo->datos, perdida=0.0f;
    for (size_t i=0;i<pred->total;i++) {
        float p=dp[i]<1e-7f?1e-7f:(dp[i]>1.0f-1e-7f?1.0f-1e-7f:dp[i]);
        perdida-=do_[i]*logf(p)+(1.0f-do_[i])*logf(1.0f-p);
    }
    perdida/=(float)pred->total;
    if (pred->requiere_grad) {
        _asegurar_gradiente(pred);
        float e=1.0f/(float)pred->total;
        for (size_t i=0;i<pred->total;i++) {
            float p=dp[i]<1e-7f?1e-7f:(dp[i]>1.0f-1e-7f?1.0f-1e-7f:dp[i]);
            pred->gradiente[i]+=e*(-do_[i]/p+(1.0f-do_[i])/(1.0f-p));
        }
    }
    int forma[]={1};
    return tc_desde_datos(&perdida, forma, 1, TC_FLOAT32);
}
