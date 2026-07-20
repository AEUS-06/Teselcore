/* src/reduce/sum.c — Reducción por suma con retropropagación */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"

static void _bwd_suma_dim(_nodo_autograd* n) {
    tc_tensor* inp = n->entradas[0];
    if (!inp->requiere_grad) return;
    _asegurar_gradiente(inp);
    float g = n->salida->gradiente[0];
    for (size_t i = 0; i < inp->total; i++)
        inp->gradiente[i] += g;
}

tc_tensor* tc_suma_dimension(tc_tensor* a, int dimension, int mantener_dim) {
    (void)dimension; (void)mantener_dim;
    float suma = 0.0f;
    float* d = (float*)a->datos;
    for (size_t i = 0; i < a->total; i++) suma += d[i];
    int forma[] = {1};
    tc_tensor* s = tc_desde_datos(&suma, forma, 1, TC_FLOAT32);
    s->requiere_grad = a->requiere_grad;
    if (s->requiere_grad) _tape_empujar(s, a, NULL, 1, 0, NULL, _bwd_suma_dim);
    return s;
}
