/* src/core/tape.c — Gestión de la cinta de autograd */
#include "../../include/internal/tape.h"
#include <stdlib.h>

_nodo_autograd* _tape_cabeza = NULL;
int             _tape_activa = 1;

_nodo_autograd* _tape_empujar(
    tc_tensor* salida,  tc_tensor* entrada_a, tc_tensor* entrada_b,
    int num_entradas,   float escalar,        void* aux,
    void (*retroprop)(_nodo_autograd*)
) {
    if (!_tape_activa) return NULL;
    _nodo_autograd* nodo = (_nodo_autograd*)malloc(sizeof(_nodo_autograd));
    nodo->salida           = salida;
    nodo->entradas[0]      = entrada_a;
    nodo->entradas[1]      = entrada_b;
    nodo->entradas[2]      = NULL;
    nodo->entradas[3]      = NULL;
    nodo->num_entradas     = num_entradas;
    nodo->escalar_guardado = escalar;
    nodo->aux              = aux;
    nodo->retropropagar    = retroprop;
    nodo->siguiente        = _tape_cabeza;
    _tape_cabeza           = nodo;
    salida->_nodo_grad     = nodo;
    return nodo;
}

void _limpiar_tape(void) {
    _nodo_autograd* actual = _tape_cabeza;
    while (actual) {
        _nodo_autograd* sig = actual->siguiente;
        if (actual->aux) free(actual->aux);
        free(actual);
        actual = sig;
    }
    _tape_cabeza = NULL;
}
