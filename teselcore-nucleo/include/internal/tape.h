#pragma once
#include "../../include/teselcore.h"

typedef struct _nodo_autograd {
    tc_tensor*  salida;
    tc_tensor*  entradas[4];
    int         num_entradas;
    float       escalar_guardado;
    void*       aux;
    void (*retropropagar)(struct _nodo_autograd*);
    struct _nodo_autograd* siguiente;
} _nodo_autograd;

extern _nodo_autograd* _tape_cabeza;
extern int             _tape_activa;

_nodo_autograd* _tape_empujar(
    tc_tensor* salida,  tc_tensor* entrada_a, tc_tensor* entrada_b,
    int num_entradas,   float escalar,        void* aux,
    void (*retroprop)(_nodo_autograd*)
);
void _limpiar_tape(void);
