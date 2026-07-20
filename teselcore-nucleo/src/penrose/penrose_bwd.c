/* src/penrose/penrose_bwd.c — Retropropagación en conv sobre teselación de Penrose */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"
#include <stdlib.h>

typedef struct { const tc_teselacion_penrose* tes; tc_tensor* sesgo; } _ctx_penrose;

static void _bwd_penrose(_nodo_autograd* n) {
    _ctx_penrose* ctx=((_ctx_penrose*)n->aux);
    const tc_teselacion_penrose* tes=ctx?ctx->tes:NULL;
    tc_tensor *ent=n->entradas[0], *ker=n->entradas[1], *sal=n->salida, *sesgo=ctx?ctx->sesgo:NULL;
    if (!ent||!ker||!sal||!tes) return;
    int B=ent->forma[0],Cin=ent->forma[1],N=ent->forma[2],Cout=ker->forma[0],K=ker->forma[2];
    float *de=(float*)ent->datos, *dk=(float*)ker->datos, *g=sal->gradiente;
    if (ent->requiere_grad)  _asegurar_gradiente(ent);
    if (ker->requiere_grad)  _asegurar_gradiente(ker);
    if (sesgo&&sesgo->requiere_grad) _asegurar_gradiente(sesgo);

    for (int b=0;b<B;b++) for (int co=0;co<Cout;co++) for (int i=0;i<N;i++) {
        float go=g[b*Cout*N+co*N+i];
        if (sesgo&&sesgo->requiere_grad) sesgo->gradiente[co]+=go;
        for (int ci=0;ci<Cin;ci++) {
            if (K>0) {
                if (ker->requiere_grad) ker->gradiente[co*Cin*K+ci*K+0]+=go*de[b*Cin*N+ci*N+i];
                if (ent->requiere_grad) ent->gradiente[b*Cin*N+ci*N+i]+=go*dk[co*Cin*K+ci*K+0];
            }
            const tc_teja_penrose* tj=&tes->tejas[i];
            for (int nv=0;nv<tj->num_vecinos&&(nv+1)<K;nv++) {
                int v=tj->vecinos[nv];
                if (ker->requiere_grad) ker->gradiente[co*Cin*K+ci*K+(nv+1)]+=go*de[b*Cin*N+ci*N+v];
                if (ent->requiere_grad) ent->gradiente[b*Cin*N+ci*N+v]+=go*dk[co*Cin*K+ci*K+(nv+1)];
            }
        }
    }
}

void tc_tape_empujar_conv_penrose_grafo(tc_tensor* sal, tc_tensor* ent, tc_tensor* ker,
                                        tc_tensor* sesgo, const tc_teselacion_penrose* tes) {
    if (!sal||!sal->requiere_grad) return;
    _ctx_penrose* ctx=(_ctx_penrose*)malloc(sizeof(_ctx_penrose));
    ctx->tes=tes; ctx->sesgo=sesgo;
    _tape_empujar(sal, ent, ker, 2, 0.0f, ctx, _bwd_penrose);
}
