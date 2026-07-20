/* src/optimizador/paso.c — tc_paso_optimizador: despacha según el tipo */
#include "../../include/internal/optimizador_pasos.h"

void tc_paso_optimizador(tc_optimizador* opt) {
    opt->paso++;
    for (int i = 0; i < opt->num_params; i++) {
        tc_tensor* p = opt->parametros[i];
        if (!p->requiere_grad || !p->gradiente) continue;
        if (opt->tipo == OPT_SGD)
            _paso_sgd(opt, p, opt->velocidades[i]);
        else
            _paso_adam(opt, p, opt->momento1[i], opt->momento2[i]);
    }
}
