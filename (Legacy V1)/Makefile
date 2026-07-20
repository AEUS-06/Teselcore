# teselcore/Makefile
# Compila TeselCore para prácticamente cualquier plataforma.
#"no se porque me enfoque mucho en esto en lugar de refactorizar
#bien el codigo del nucleo, pero hey sirve para compilar en muchas partes"
# ── Targets disponibles ───────────────────────────────────────────────────────
#   make                → biblioteca nativa (Linux .so / macOS .dylib)
#   make estatica       → libteselcore.a  (linkeo estático, cualquier plataforma)
#   make windows        → teselcore.dll   (x86-64, mingw-w64)
#   make windows-arm    → teselcore-arm64.dll (ARM64 Windows, mingw-w64)
#   make macos          → libteselcore.dylib universal (x86-64 + arm64)
#   make ios            → libteselcore-ios.a (arm64, iPhone/iPad)
#   make android-arm64  → libteselcore-android-arm64.so
#   make android-arm32  → libteselcore-android-arm32.so
#   make android-x86    → libteselcore-android-x86_64.so
#   make android        → las tres variantes Android juntas
#   make wasm           → teselcore.wasm  (navegador, requiere emscripten)
#   make wasm-threads   → teselcore-threads.wasm (WASM con pthreads)
#   make rpi            → libteselcore-rpi.so (Raspberry Pi, ARMv7 NEON)
#   make embedded-arm   → libteselcore-embedded.a (Cortex-M4/M7, bare-metal)
#   make embedded-rv    → libteselcore-embedded-rv.a (RISC-V, bare-metal)
#   make desktop        → nativa + windows + estatica
#   make movil          → ios + android
#   make svg_demo       → genera teselacion_penrose.svg de demostración
#   make test           → tests en la plataforma nativa
#   make instalar       → copiar a /usr/local
#   make info           → mostrar compiladores disponibles
#   make clean          → limpiar todos los artefactos
# ─────────────────────────────────────────────────────────────────────────────

SRC = src/tensor.c src/penrose.c src/formato_ax.c
OBJ = $(SRC:.c=.o)

# Flags base compartidos por todas las plataformas
CFLAGS_BASE = -O3 -ffast-math -Wall -Wextra -I./include -std=c11

# ── Detección de plataforma nativa ───────────────────────────────────────────
UNAME := $(shell uname -sm)

ifeq ($(findstring Darwin,$(UNAME)),Darwin)
    CC      = clang
    CFLAGS  = $(CFLAGS_BASE) -fPIC
    LIB     = libteselcore.dylib
    LDFLAGS = -shared -dynamiclib -lm
    ARCH_FLAGS = -arch x86_64 -arch arm64
else ifeq ($(findstring Linux,$(UNAME)),Linux)
    CC      = gcc
    CFLAGS  = $(CFLAGS_BASE) -march=native -fPIC
    LIB     = libteselcore.so
    LDFLAGS = -shared -Wl,-soname,libteselcore.so -lm
else
    # Fallback genérico (BSD, Haiku, etc.)
    CC      = cc
    CFLAGS  = $(CFLAGS_BASE) -fPIC
    LIB     = libteselcore.so
    LDFLAGS = -shared -lm
endif

# ── Principal (nativa) ───────────────────────────────────────────────────────
all: $(LIB)

$(LIB): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ -lm
	@echo "✓ $(LIB)"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Estática genérica ─────────────────────────────────────────────────────────
# Sirve para cualquier plataforma donde se compile con el CC nativo.
estatica:
	$(CC) $(CFLAGS_BASE) -fPIC -march=native -I./include -std=c11 \
	    -c $(SRC)
	ar rcs libteselcore.a $(notdir $(OBJ))
	ranlib libteselcore.a
	@echo "✓ libteselcore.a"

# ── macOS universal (x86-64 + Apple Silicon) ─────────────────────────────────
macos:
	clang $(CFLAGS_BASE) -fPIC $(ARCH_FLAGS) \
	    -shared -dynamiclib -lm \
	    -o libteselcore.dylib $(SRC)
	@echo "✓ libteselcore.dylib (universal x86-64 + arm64)"

# ── iOS (arm64, bare-metal / App Store) ──────────────────────────────────────
# Requiere Xcode instalado en macOS.
IOS_SDK    := $(shell xcrun --sdk iphoneos --show-sdk-path 2>/dev/null)
IOS_CC     = $(shell xcrun -f clang 2>/dev/null)
IOS_CFLAGS = $(CFLAGS_BASE) -arch arm64 -miphoneos-version-min=13.0 \
             -isysroot $(IOS_SDK) -fembed-bitcode

ios:
	@test -n "$(IOS_SDK)" || (echo "ERROR: Xcode no encontrado"; exit 1)
	$(IOS_CC) $(IOS_CFLAGS) -c $(SRC)
	ar rcs libteselcore-ios.a $(notdir $(OBJ))
	ranlib libteselcore-ios.a
	@echo "✓ libteselcore-ios.a (arm64, iOS >= 13)"

# ── Windows x86-64 (mingw-w64) ───────────────────────────────────────────────
WIN_CC    = x86_64-w64-mingw32-gcc
WIN_FLAGS = $(CFLAGS_BASE) -march=x86-64 -I./include

windows:
	@which $(WIN_CC) > /dev/null || (echo "ERROR: mingw-w64 no encontrado"; exit 1)
	$(WIN_CC) $(WIN_FLAGS) -shared -o teselcore.dll $(SRC) -lm
	$(WIN_CC) $(WIN_FLAGS) -c $(SRC)
	ar rcs libteselcore-windows.a $(notdir $(OBJ))
	@echo "✓ teselcore.dll + libteselcore-windows.a (x86-64)"

# ── Windows ARM64 (mingw-w64 aarch64) ────────────────────────────────────────
WINARM_CC = aarch64-w64-mingw32-gcc

windows-arm:
	@which $(WINARM_CC) > /dev/null || (echo "ERROR: aarch64-w64-mingw32-gcc no encontrado"; exit 1)
	$(WINARM_CC) $(CFLAGS_BASE) -march=armv8-a -shared \
	    -o teselcore-arm64.dll $(SRC) -lm
	@echo "✓ teselcore-arm64.dll (ARM64 Windows)"

# ── Android (NDK) ────────────────────────────────────────────────────────────
# Requiere Android NDK. Ajusta NDK_PATH si es necesario.
NDK_PATH   ?= $(HOME)/Android/Sdk/ndk/$(shell ls $(HOME)/Android/Sdk/ndk 2>/dev/null | sort -V | tail -1)
NDK_HOST   := $(shell uname -s | tr '[:upper:]' '[:lower:]')-x86_64
NDK_TOOLS   = $(NDK_PATH)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin
ANDROID_API = 21

android-arm64:
	@test -d "$(NDK_TOOLS)" || (echo "ERROR: Android NDK no encontrado en $(NDK_PATH)"; exit 1)
	$(NDK_TOOLS)/aarch64-linux-android$(ANDROID_API)-clang \
	    $(CFLAGS_BASE) -march=armv8-a -fPIC -shared \
	    -o libteselcore-android-arm64.so $(SRC) -lm
	@echo "✓ libteselcore-android-arm64.so (API $(ANDROID_API)+)"

android-arm32:
	@test -d "$(NDK_TOOLS)" || (echo "ERROR: Android NDK no encontrado en $(NDK_PATH)"; exit 1)
	$(NDK_TOOLS)/armv7a-linux-androideabi$(ANDROID_API)-clang \
	    $(CFLAGS_BASE) -march=armv7-a -mfpu=neon -mfloat-abi=softfp \
	    -fPIC -shared \
	    -o libteselcore-android-arm32.so $(SRC) -lm
	@echo "✓ libteselcore-android-arm32.so (API $(ANDROID_API)+, ARMv7)"

android-x86:
	@test -d "$(NDK_TOOLS)" || (echo "ERROR: Android NDK no encontrado en $(NDK_PATH)"; exit 1)
	$(NDK_TOOLS)/x86_64-linux-android$(ANDROID_API)-clang \
	    $(CFLAGS_BASE) -march=x86-64 -fPIC -shared \
	    -o libteselcore-android-x86_64.so $(SRC) -lm
	@echo "✓ libteselcore-android-x86_64.so (API $(ANDROID_API)+, x86-64)"

android: android-arm64 android-arm32 android-x86
	@echo "✓ Todas las variantes Android compiladas"

# ── WebAssembly (navegador / Node.js) ────────────────────────────────────────
WASM_EXPORTS = '["_tc_vacio","_tc_ceros","_tc_unos","_tc_escalar",\
"_tc_aleatorio_uniforme","_tc_aleatorio_normal","_tc_desde_datos","_tc_clonar",\
"_tc_liberar","_tc_ref","_tc_unref","_tc_elemento",\
"_tc_sumar","_tc_restar","_tc_multiplicar","_tc_dividir","_tc_negar",\
"_tc_potencia","_tc_logaritmo","_tc_exponencial","_tc_raiz","_tc_valor_absoluto",\
"_tc_suma_dimension","_tc_media_dimension","_tc_maximo_dimension","_tc_minimo_dimension",\
"_tc_multiplicacion_matricial","_tc_transponer","_tc_reformar","_tc_aplanar","_tc_concatenar",\
"_tc_relu","_tc_relu_con_fuga","_tc_sigmoide","_tc_tangente_hiperbolica",\
"_tc_softmax","_tc_gelu","_tc_silu","_tc_embedding",\
"_tc_lineal","_tc_conv2d","_tc_agrupacion_max","_tc_agrupacion_promedio",\
"_tc_normalizacion_lote","_tc_abandono",\
"_tc_entropia_cruzada","_tc_error_cuadratico_medio","_tc_entropia_cruzada_binaria",\
"_tc_retropropagar","_tc_cero_gradiente",\
"_tc_crear_teselacion","_tc_liberar_teselacion",\
"_tc_crear_kernel_penrose","_tc_liberar_kernel_penrose",\
"_tc_conv_penrose","_tc_conv_penrose_grafo",\
"_tc_imagen_a_teselacion","_tc_teselacion_a_imagen",\
"_tc_exportar_svg_teselacion",\
"_tc_guardar","_tc_cargar","_tc_liberar_modelo",\
"_tc_modelo_nuevo","_tc_modelo_agregar_tensor","_tc_modelo_obtener_tensor",\
"_tc_sgd","_tc_adam","_tc_adamw",\
"_tc_paso_optimizador","_tc_cero_gradientes_optimizador","_tc_liberar_optimizador",\
"_tc_imprimir","_tc_imprimir_info","_tc_semilla_aleatoria"]'

wasm:
	@which emcc > /dev/null || (echo "ERROR: emscripten no encontrado (source emsdk_env.sh)"; exit 1)
	emcc $(SRC) -I./include -O3 \
	    -s WASM=1 \
	    -s EXPORTED_FUNCTIONS=$(WASM_EXPORTS) \
	    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' \
	    -s FORCE_FILESYSTEM=1 \
	    -s ALLOW_MEMORY_GROWTH=1 \
	    -s INITIAL_MEMORY=33554432 \
	    -o teselcore.wasm
	@echo "✓ teselcore.wasm"

# WASM con soporte de hilos (SharedArrayBuffer, requiere COOP/COEP en el servidor)
wasm-threads:
	@which emcc > /dev/null || (echo "ERROR: emscripten no encontrado"; exit 1)
	emcc $(SRC) -I./include -O3 \
	    -s WASM=1 \
	    -s USE_PTHREADS=1 \
	    -s PTHREAD_POOL_SIZE=4 \
	    -s EXPORTED_FUNCTIONS=$(WASM_EXPORTS) \
	    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' \
	    -s FORCE_FILESYSTEM=1 \
	    -s ALLOW_MEMORY_GROWTH=1 \
	    -o teselcore-threads.wasm
	@echo "✓ teselcore-threads.wasm (pthreads)"

# ── Raspberry Pi (ARMv7 hard-float, cross desde Linux x86-64) ────────────────
RPI_CC = arm-linux-gnueabihf-gcc

rpi:
	@which $(RPI_CC) > /dev/null || (echo "ERROR: arm-linux-gnueabihf-gcc no encontrado"; exit 1)
	$(RPI_CC) $(CFLAGS_BASE) \
	    -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard \
	    -fPIC -shared -o libteselcore-rpi.so $(SRC) -lm
	@echo "✓ libteselcore-rpi.so (Raspberry Pi, ARMv7 NEON)"

# ── Embedded ARM Cortex-M (bare-metal, sin sistema operativo) ────────────────
# Target: STM32F4 / STM32H7 / RP2040 (Cortex-M4/M7)
EMB_CC    = arm-none-eabi-gcc
EMB_FLAGS = $(CFLAGS_BASE) \
            -march=armv7e-m -mtune=cortex-m4 \
            -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
            -mthumb \
            -fno-exceptions \
            -ffreestanding \
            -DTARGET_EMBEDDED \
            -nostdlib

embedded-arm:
	@which $(EMB_CC) > /dev/null || (echo "ERROR: arm-none-eabi-gcc no encontrado"; exit 1)
	$(EMB_CC) $(EMB_FLAGS) -c $(SRC)
	arm-none-eabi-ar rcs libteselcore-embedded.a $(notdir $(OBJ))
	arm-none-eabi-ranlib libteselcore-embedded.a
	@echo "✓ libteselcore-embedded.a (Cortex-M4/M7, bare-metal)"

# ── Embedded RISC-V (bare-metal) ─────────────────────────────────────────────
# Target: ESP32-C3, CH32V, GD32VF, etc.
RV_CC    = riscv32-unknown-elf-gcc
RV_FLAGS = $(CFLAGS_BASE) \
           -march=rv32imfc -mabi=ilp32f \
           -ffreestanding \
           -DTARGET_EMBEDDED \
           -nostdlib

embedded-rv:
	@which $(RV_CC) > /dev/null || (echo "ERROR: riscv32-unknown-elf-gcc no encontrado"; exit 1)
	$(RV_CC) $(RV_FLAGS) -c $(SRC)
	riscv32-unknown-elf-ar rcs libteselcore-embedded-rv.a $(notdir $(OBJ))
	@echo "✓ libteselcore-embedded-rv.a (RISC-V rv32imfc, bare-metal)"

# ── Grupos de targets ─────────────────────────────────────────────────────────
desktop: all macos windows estatica
	@echo "✓ Todos los targets de escritorio compilados"

movil: ios android
	@echo "✓ Todos los targets móviles compilados"

# ── Demo SVG ──────────────────────────────────────────────────────────────────
svg_demo: $(LIB)
	$(CC) -O2 -I./include demos/generar_svg.c \
	    -L. -lteselcore -Wl,-rpath,. -lm -o demo_svg
	LD_LIBRARY_PATH=. ./demo_svg
	@echo "✓ teselacion_penrose.svg generado"

# ── CLI (ejecutable de utilidades) ───────────────────────────────────────────
cli:
	$(CC) $(CFLAGS_BASE) -I./include -std=c11 src/cli.c $(SRC) -lm -o tesel
	@echo "✓ tesel (CLI) compilado"

# ── Tests ──────────────────────────────────────────────────────────────────────
test: $(LIB)
	$(CC) -O2 -I./include -L. tests/test_basico.c \
	    -lteselcore -Wl,-rpath,. -lm -o test_runner
	LD_LIBRARY_PATH=. ./test_runner
	@echo "✓ Tests completados"

# ── Instalar ───────────────────────────────────────────────────────────────────
instalar: $(LIB)
	cp $(LIB) /usr/local/lib/
	cp include/teselcore.h /usr/local/include/
	ldconfig 2>/dev/null || true
	@echo "✓ TeselCore instalado en /usr/local"

# ── Info de compiladores disponibles ─────────────────────────────────────────
info:
	@echo "── Compiladores detectados ──────────────────────────────"
	@which gcc                     && echo "  ✓ gcc                  (Linux nativo)"        || echo "  ✗ gcc"
	@which clang                   && echo "  ✓ clang                (macOS/Linux)"         || echo "  ✗ clang"
	@which x86_64-w64-mingw32-gcc  && echo "  ✓ mingw-w64 x86-64     (Windows 64-bit)"     || echo "  ✗ mingw-w64 x86-64"
	@which aarch64-w64-mingw32-gcc && echo "  ✓ mingw-w64 aarch64    (Windows ARM64)"      || echo "  ✗ mingw-w64 aarch64"
	@which arm-linux-gnueabihf-gcc && echo "  ✓ arm-linux-gnueabihf  (Raspberry Pi)"       || echo "  ✗ arm-linux-gnueabihf"
	@which arm-none-eabi-gcc       && echo "  ✓ arm-none-eabi        (Cortex-M bare-metal)" || echo "  ✗ arm-none-eabi"
	@which riscv32-unknown-elf-gcc && echo "  ✓ riscv32-unknown-elf  (RISC-V bare-metal)"  || echo "  ✗ riscv32-unknown-elf"
	@which emcc                    && echo "  ✓ emcc                 (WebAssembly)"         || echo "  ✗ emcc"
	@test -n "$(IOS_SDK)"          && echo "  ✓ Xcode                (iOS)"                || echo "  ✗ Xcode/iOS SDK"
	@test -d "$(NDK_TOOLS)"        && echo "  ✓ Android NDK          ($(NDK_PATH))"        || echo "  ✗ Android NDK"
	@echo "─────────────────────────────────────────────────────────"

# ── Limpieza ───────────────────────────────────────────────────────────────────
clean:
	rm -f $(OBJ) \
	      libteselcore.so libteselcore.dylib libteselcore.a \
	      libteselcore-windows.a \
	      libteselcore-ios.a \
	      libteselcore-android-arm64.so \
	      libteselcore-android-arm32.so \
	      libteselcore-android-x86_64.so \
	      libteselcore-rpi.so \
	      libteselcore-embedded.a \
	      libteselcore-embedded-rv.a \
	      teselcore.dll teselcore-arm64.dll \
	      teselcore.wasm teselcore-threads.wasm \
	      test_runner demo_svg *.svg *.o

.PHONY: all estatica macos ios windows windows-arm \
        android android-arm64 android-arm32 android-x86 \
        wasm wasm-threads rpi embedded-arm embedded-rv \
        desktop movil svg_demo test instalar info clean