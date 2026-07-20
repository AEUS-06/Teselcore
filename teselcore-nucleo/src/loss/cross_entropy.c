/* src/loss/cross_entropy.c — Entropía cruzada categórica */
#include "../../include/teselcore.h"
#include "../../include/internal/helpers.h"
#include <math.h>

tc_tensor* tc_entropia_cruzada(tc_tensor* logits, tc_tensor* objetivos) {
    int L=logits->forma[0], C=logits->forma[1];
    tc_tensor* sm=tc_softmax(logits, -1);
    float* sp=(float*)sm->datos;
    int*   tp=(int*)objetivos->datos;

    float perdida=0.0f;
    for (int b=0;b<L;b++) perdida-=logf(sp[b*C+tp[b]]+1e-8f);
    perdida/=(float)L;

    if (logits->requiere_grad) {
        _asegurar_gradiente(logits);
        for (int b=0;b<L;b++) for (int c=0;c<C;c++) {
            float delta=sp[b*C+c]-(c==tp[b]?1.0f:0.0f);
            logits->gradiente[b*C+c]+=delta/(float)L;
        }
    }
    tc_liberar(sm);
    int forma[]={1};
    return tc_desde_datos(&perdida, forma, 1, TC_FLOAT32);
}
