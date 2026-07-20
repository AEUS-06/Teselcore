# ══════════════════════════════════════════════════════
# Makefile — TeselCore (arquitectura modular)
# Cada módulo vive en su carpeta; agregar uno nuevo =
# añadir su .c aquí. No tocar nada más.
# ══════════════════════════════════════════════════════

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iinclude
LDFLAGS = -lm

BUILD   = build
LIB     = $(BUILD)/libteselcore.a

# ── Fuentes agrupadas por módulo ──────────────────────
SRCS_CORE   = src/core/rng.c src/core/tape.c src/core/helpers.c
SRCS_TENSOR = src/tensor/create_basic.c src/tensor/create_from.c \
              src/tensor/create_random.c src/tensor/lifecycle.c
SRCS_OPS    = src/ops/add.c src/ops/sub.c src/ops/mul.c src/ops/div.c \
              src/ops/neg.c src/ops/pow.c src/ops/exp_log.c src/ops/sqrt_abs.c
SRCS_REDUCE = src/reduce/sum.c src/reduce/stats.c
SRCS_LINALG = src/linalg/matmul.c src/linalg/transform.c src/linalg/concat.c
SRCS_ACT    = src/activations/relu.c src/activations/sigmoid_tanh.c \
              src/activations/softmax.c src/activations/gelu.c src/activations/silu.c
SRCS_LAYER  = src/layers/linear.c src/layers/conv2d.c src/layers/pooling.c \
              src/layers/batchnorm.c src/layers/dropout_embed.c
SRCS_LOSS   = src/loss/cross_entropy.c src/loss/mse_bce.c
SRCS_PENR   = src/penrose/buffers.c src/penrose/subdivide.c src/penrose/semilla.c \
              src/penrose/vecindad.c src/penrose/bbox.c src/penrose/teselacion.c \
              src/penrose/kernel.c src/penrose/interp.c src/penrose/conv_imagen.c \
              src/penrose/conv_grafo.c src/penrose/imagen_teselacion.c \
              src/penrose/teselacion_imagen.c src/penrose/svg_export.c \
              src/penrose/penrose_bwd.c
SRCS_MODELO = src/modelo/crc32.c src/modelo/io_binario.c src/modelo/guardar_cuerpo.c \
              src/modelo/guardar_crc.c src/modelo/guardar.c src/modelo/cargar_validar.c \
              src/modelo/cargar_tensor.c src/modelo/cargar.c src/modelo/gestion.c \
              src/modelo/tensores.c
SRCS_OPTIM  = src/optimizador/sgd.c src/optimizador/adam.c src/optimizador/paso_sgd.c \
              src/optimizador/paso_adam.c src/optimizador/paso.c src/optimizador/gestion.c
SRCS_AUTO   = src/autograd/backward.c
SRCS_UTILS  = src/utils/utils.c
SRCS_CLI    = src/cli/usage.c src/cli/cmd_info.c src/cli/cmd_gen.c \
              src/cli/cmd_save_demo.c src/cli/cmd_load.c src/cli/cmd_conv_demo.c \
              src/cli/main.c

ALL_SRCS = $(SRCS_CORE) $(SRCS_TENSOR) $(SRCS_OPS) $(SRCS_REDUCE) \
           $(SRCS_LINALG) $(SRCS_ACT) $(SRCS_LAYER) $(SRCS_LOSS) \
           $(SRCS_PENR) $(SRCS_MODELO) $(SRCS_OPTIM) $(SRCS_AUTO) $(SRCS_UTILS)

OBJS     = $(patsubst src/%.c, $(BUILD)/%.o, $(ALL_SRCS))
CLI_OBJS = $(patsubst src/%.c, $(BUILD)/%.o, $(SRCS_CLI))
SO       = $(BUILD)/libteselcore.so
CLI_BIN  = $(BUILD)/teselcore-cli

# ── Reglas ────────────────────────────────────────────
.PHONY: all clean cli so test

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(BUILD)
	ar rcs $@ $^
	@echo "✓ Biblioteca lista: $@"

so: $(SO)
$(SO): $(ALL_SRCS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared $(ALL_SRCS) -o $@ $(LDFLAGS)
	@echo "✓ Biblioteca compartida lista: $@ (para bindings Python)"

cli: $(CLI_BIN)
$(CLI_BIN): $(LIB) $(CLI_OBJS)
	$(CC) $(CFLAGS) $(CLI_OBJS) -L$(BUILD) -lteselcore -o $@ $(LDFLAGS)
	@echo "✓ CLI lista: $@"

test: $(LIB)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) tests/integration/test_penrose_completo.c -L$(BUILD) -l:libteselcore.a -o $(BUILD)/test_penrose $(LDFLAGS)
	./$(BUILD)/test_penrose

$(BUILD)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)
	@echo "✓ Limpio"