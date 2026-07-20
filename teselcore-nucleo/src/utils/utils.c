/* src/utils/utils.c — Impresión, info del sistema y gestión de dispositivo */
#include "../../include/teselcore.h"
#include <stdio.h>

void tc_imprimir(const tc_tensor* t) {
    printf("Tensor(forma=[");
    for (int i=0;i<t->ndim;i++) { printf("%d",t->forma[i]); if (i<t->ndim-1) printf(", "); }
    printf("], tipo=%d, requiere_grad=%d)\n", t->tipo, t->requiere_grad);
    float* d=(float*)t->datos;
    size_t lim=t->total<12?t->total:12;
    printf("datos: [");
    for (size_t i=0;i<lim;i++) { printf("%.4f",d[i]); if (i<lim-1) printf(", "); }
    if (t->total>12) printf(", ...");
    printf("]\n");
}

void tc_imprimir_info(void) {
    printf("TeselCore v%s\n", TC_VERSION_STR);
    printf("Compilado: %s %s\n", __DATE__, __TIME__);
#ifdef __AVX2__
    printf("AVX2: disponible\n");
#else
    printf("AVX2: no disponible\n");
#endif
    printf("Razón áurea φ: %.10f\n", TC_PHI);
}

tc_tensor* tc_a_dispositivo(tc_tensor* t, tc_dispositivo disp) {
    t->dispositivo = disp;
    return t;
}
