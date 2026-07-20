/* src/layers/linear.c — Capa totalmente conectada (y = xWᵀ + b) */
#include "../../include/teselcore.h"

tc_tensor* tc_lineal(tc_tensor* x, tc_tensor* pesos, tc_tensor* sesgo) {
    tc_tensor* pt = tc_transponer(pesos, 0, 1);
    tc_tensor* xp = tc_multiplicacion_matricial(x, pt);
    if (!pt->requiere_grad) tc_liberar(pt);

    if (!sesgo) return xp;
    tc_tensor* sal = tc_sumar(xp, sesgo);
    if (!xp->requiere_grad) tc_liberar(xp);
    return sal;
}
