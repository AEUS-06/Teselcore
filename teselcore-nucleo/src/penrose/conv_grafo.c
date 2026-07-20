/* src/penrose/conv_grafo.c — Convolución de Penrose en modo grafo (forward) */
#include "../../include/teselcore.h"
#include <assert.h>

tc_tensor* tc_conv_penrose_grafo(tc_tensor* entrada, tc_tensor* kernel, tc_tensor* sesgo,
                                  const tc_teselacion_penrose* teselacion) {
    assert(entrada->ndim==3 && kernel->ndim==3);
    int B=entrada->forma[0], Cin=entrada->forma[1], N=entrada->forma[2];
    int Cout=kernel->forma[0], K=kernel->forma[2];
    assert(Cin==kernel->forma[1] && N==teselacion->num_tejas);

    int forma[]={B,Cout,N};
    tc_tensor* sal=tc_ceros(forma, 3);
    float *ds=(float*)sal->datos, *de=(float*)entrada->datos, *dk=(float*)kernel->datos;
    float *db = sesgo ? (float*)sesgo->datos : NULL;

    for (int b=0;b<B;b++) for (int co=0;co<Cout;co++) for (int i=0;i<N;i++) {
        float suma = db ? db[co] : 0.0f;
        const tc_teja_penrose* tj=&teselacion->tejas[i];
        for (int ci=0;ci<Cin;ci++) {
            if (0<K) suma += dk[co*Cin*K+ci*K+0]*de[b*Cin*N+ci*N+i];
            for (int nv=0;nv<tj->num_vecinos && (nv+1)<K;nv++) {
                int vecino=tj->vecinos[nv];
                suma += dk[co*Cin*K+ci*K+(nv+1)]*de[b*Cin*N+ci*N+vecino];
            }
        }
        ds[b*Cout*N+co*N+i]=suma;
    }

    sal->requiere_grad = entrada->requiere_grad || kernel->requiere_grad;
    if (sal->requiere_grad)
        tc_tape_empujar_conv_penrose_grafo(sal, entrada, kernel, sesgo, teselacion);
    return sal;
}
