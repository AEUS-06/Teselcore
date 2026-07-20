/* src/cli/main.c — Punto de entrada del CLI: despacha por subcomando */
#include "../../include/internal/cli_comandos.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { _usage(argv[0]); return 1; }
    const char* cmd = argv[1];

    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "h") == 0) { _usage(argv[0]); return 0; }
    if (strcmp(cmd, "info") == 0)       return _cmd_info();
    if (strcmp(cmd, "gen") == 0)        return _cmd_gen(argc, argv, argv[0]);
    if (strcmp(cmd, "save_demo") == 0)  return _cmd_save_demo(argc, argv, argv[0]);
    if (strcmp(cmd, "load") == 0)       return _cmd_load(argc, argv, argv[0]);
    if (strcmp(cmd, "conv_demo") == 0)  return _cmd_conv_demo(argc, argv, argv[0]);

    fprintf(stderr, "Comando desconocido: %s\n", cmd);
    _usage(argv[0]);
    return 1;
}
