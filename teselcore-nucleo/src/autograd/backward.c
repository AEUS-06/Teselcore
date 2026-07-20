/* src/autograd/backward.c — Retropropagación a través de la cinta */
#include "../../include/teselcore.h"
#include "../../include/internal/tape.h"
#include "../../include/internal/helpers.h"

void tc_retropropagar(tc_tensor* perdida) {
    _asegurar_gradiente(perdida);
    perdida->gradiente[0] = 1.0f;

    _nodo_autograd* nodo = _tape_cabeza;
    while (nodo) {
        if (nodo->salida->gradiente && nodo->retropropagar)
            nodo->retropropagar(nodo);
        nodo = nodo->siguiente;
    }
    _limpiar_tape();
}
