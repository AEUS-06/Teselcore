```
████████╗███████╗███████╗███████╗██╗      ██████╗ ██████╗ ██████╗ ███████╗
╚══██╔══╝██╔════╝██╔════╝██╔════╝██║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║   █████╗  ███████╗█████╗  ██║     ██║     ██║   ██║██████╔╝█████╗  
   ██║   ██╔══╝  ╚════██║██╔══╝  ██║     ██║     ██║   ██║██╔══██╗██╔══╝  
   ██║   ███████╗███████║███████╗███████╗╚██████╗╚██████╔╝██║  ██║███████╗
   ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

<div align="center">

### *Un kernel experimental donde las teselaciones de Penrose se convierten en redes neuronales. Sí, es tan raro como suena.*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blueviolet.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![C](https://img.shields.io/badge/Core-C99-orange?logo=c&logoColor=white)](https://en.wikipedia.org/wiki/C99)
[![Status](https://img.shields.io/badge/Estado-Experimental%20🧪-yellow)](https://github.com/AEUS-06/TeselCore)
[![Contributions](https://img.shields.io/badge/Ideas%20Locas-Bienvenidas-ff69b4)](https://github.com/AEUS-06/TeselCore/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/AEUS-06/TeselCore/pulls)

</div>

---

> **⚠️ Advertencia:** Este no es tu framework de ML de todos los días.  
> Si buscas algo estable, predecible y aburrido — este no es tu lugar.  
> Pero si alguna vez te preguntaste *¿qué pasa si uso geometría imposible como base de una red neuronal?* — entonces bienvenido a casa.

---

## ¿Qué rayos es TeselCore?

Imagina que Roger Penrose y un ingeniero de redes neuronales se juntan a tomar café y deciden que las convoluciones cuadradas son demasiado convencionales. Eso, más o menos, es TeselCore.

En su núcleo, TeselCore propone usar **teselaciones de Penrose** — esos patrones geométricos aperiódicos que llenan el plano sin repetirse jamás — como base para operaciones de convolución en redes neuronales ligeras. No es un framework listo para producción. Es un **laboratorio de ideas extrañas**, un lugar donde la matemática rara se encuentra con el machine learning y nadie sabe exactamente qué va a salir.

Por ahora es experimental. Y eso es exactamente el punto.

---

## 🧠 La filosofía del proyecto

```
   ◈  matemáticas         ◈  física curiosa         ◈  ideas de las 2am
      que nadie usa          sin terminar               que no te dejan
      en producción          de entender                dormir
           │                      │                          │
           └──────────────────────┴──────────────────────────┘
                                  │
                                  ▼
                    ╔═════════════════════════╗
                    ║                         ║
                    ║      T E S E L C O R E  ║
                    ║                         ║
                    ╚═════════════════════════╝
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       tal vez funciona     tal vez explota      tal vez ambas
       y no sabes por qué   espectacularmente    cosas a la vez
```

TeselCore no nació para competir con PyTorch ni con JAX. Nació porque algunas preguntas merecen ser preguntadas aunque nadie sepa todavía la respuesta:

- ¿Puede una convolución aperiódica capturar patrones que una convolución estándar pierde?
- ¿Qué propiedades emergentes aparecen cuando la geometría del kernel no tiene simetría traslacional?
- ¿Y si la próxima gran idea en redes neuronales viene de la cristalografía cuántica o de un mosaico árabe del siglo XIII?

Esas son las preguntas que viven aquí.

---

## ✨ ¿Qué hay dentro?

```
TeselCore/
│
├── 🔧 teselcore-nucleo/          ← Motor en C. Crudo, modular, sin adornos.
│   ├── src/                      ── 56 archivos fuente en 14 módulos ──
│   │   ├── activations/          ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax
│   │   ├── autograd/             Backward automático (tape graph)
│   │   ├── cli/                  Interfaz CLI: main, info, gen, load, save_demo, conv_demo
│   │   ├── core/                 Helpers internos, RNG (xoshiro256**), gestión de tape
│   │   ├── layers/               Linear, Conv2D, BatchNorm, Dropout, Embedding, Pooling
│   │   ├── linalg/               MatMul, Transpose, Reshape, Concat
│   │   ├── loss/                 Cross-entropy, MSE, Binary Cross-entropy
│   │   ├── modelo/               Serialización .ax: guardar, cargar, CRC32, I/O binario
│   │   ├── ops/                  Add, Sub, Mul, Div, Neg, Pow, Exp, Log, Sqrt, Abs
│   │   ├── optimizador/          SGD, Adam, AdamW — creación, paso, gestión
│   │   ├── penrose/              ⬡ El corazón: teselación, subdivisión, semilla,
│   │   │                           vecindad, kernel, interpolación, conv_grafo,
│   │   │                           conv_imagen, proyección imagen↔teselación,
│   │   │                           SVG export, backward de Penrose, bbox
│   │   ├── reduce/               Sum, Min, Max, Mean
│   │   ├── tensor/               Creación: zeros, ones, random, from_data, lifecycle
│   │   └── utils/                Utilidades varias
│   │
│   ├── include/
│   │   ├── teselcore.h           ← API pública (200 líneas, toda la interfaz)
│   │   └── internal/             Headers internos
│   │
│   ├── demos/
│   │   └── visualizar_kernel.c   Genera SVG del kernel de Penrose
│   │
│   ├── tests/
│   │   └── integration/
│   │       └── test_penrose_completo.c  Prueba end-to-end: forward+backward+save/load
│   │
│   └── Makefile                  Build: lib, .so, CLI, test
│
├── 🐍 teselcore-python/          ← Bindings Python (ctypes al C nativo)
│   └── teselcore/
│       ├── tensor.py             Tensor con autograd y operadores sobrecargados
│       ├── penrose.py            Teselacion, KernelPenrose, convolución en grafo/imagen
│       ├── modelo.py             Modelo: guardar/cargar .ax
│       ├── optimizador.py        SGD, Adam, AdamW
│       └── perdidas.py           Entropía cruzada, MSE, BCE
│
├── 🧪 experimentos/              ← Donde las ideas chocan contra la realidad
│   └── mnist/                    PenroseNet vs CNNClasico en MNIST
│       ├── penrose_net.py        PenroseNet: teselación→grafo→bloques residuales
│       ├── entrenar_penrose.py   Entrenamiento híbrido: grad numérico + autograd
│       ├── probar_modelos.py     Evaluación y matrices de confusión
│       └── graficar_resultados.py  Visualización comparativa
│
├── 👴 (Legacy V1)/               ← El prototipo monolítico original. Historia viva.
│
└── 📄 document/                  ← Paper principal en PDF
```

---

## ⚙️ Cómo funciona esto por dentro

### 🧮 El algoritmo de teselación

Penrose no se dibuja. Se *cultiva*. El núcleo implementa subdivisión por **deflación**: empiezas con una semilla y en cada iteración cada baldosa se parte en piezas más pequeñas siguiendo reglas geométricas fijas, gobernadas por φ. Tras `n` iteraciones tienes `O(φ²ⁿ)` baldosas formando una teselación que cubre el plano sin repetirse jamás.

```
      Iteración 0          Iteración 1          Iteración 2

        ╱▔╲                ╱╲  ╱╲             ╱╲╱╲╱╲╱╲
       ╱   ╲              ╱  ╲╱  ╲           ╱  ╲╱  ╲╱
      ╱     ╲            ╱  ╱╲  ╱╲          ╱ ╱╲ ╱╲ ╱╲
      ╲     ╱            ╲ ╱  ╲╱  ╱         ╲╱ ╲╱ ╲╱ ╲
       ╲   ╱              ╲╱  ╲╱            ╲╱╲╱╲╱╲╱╲
        ╲╱              kite    dart         ╲╱╲╱╲╱╲╱

      semilla            1 deflación         2 deflaciones
```

### ⬡ Convolución aperiódica — la idea central

Una convolución normal desliza un kernel cuadrado sobre una cuadrícula regular. TeselCore hace algo distinto:

```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   CONVOLUCIÓN ESTÁNDAR          CONVOLUCIÓN TESELADA   │
  │                                                         │
  │   ┌───┬───┬───┐                ⬗ ⬔ ⬕ ⬗ ⬔ ⬕ ⬗         │
  │   │ w │ w │ w │               ⬕ ⬗ ⬔ ⬕ ⬗ ⬔ ⬕ ⬗        │
  │   ├───┼───┼───┤               ⬔ ⬕ ⬗ ⬔ ⬕ ⬗ ⬔ ⬕        │
  │   │ w │ w │ w │               ⬗ ⬔ ⬕ ⬗ ⬔ ⬕ ⬗ ⬔        │
  │   ├───┼───┼───┤               ⬕ ⬗ ⬔ ⬕ ⬗ ⬔ ⬕ ⬗        │
  │   │ w │ w │ w │               ⬔ ⬕ ⬗ ⬔ ⬕ ⬗ ⬔ ⬕        │
  │   └───┴───┴───┘               ⬗ ⬔ ⬕ ⬗ ⬔ ⬕ ⬗ ⬔        │
  │                                                         │
  │   Kernel 3×3                  Kernel Penrose nivel 3   │
  │   → 9 pesos fijos             → ~160 nodos en grafo    │
  │   → simetría traslacional     → sin periodicidad       │
  │   → vecinos cardinales        → vecinos geométricos φ  │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

**Dos modos de convolución:**

| Modo | Descripción | Complejidad |
|------|-------------|-------------|
| `conv_grafo` | Cada nodo se agrega con sus vecinos geométricos | `O(N · d · Cᵢₙ · C_out)`, `d ≈ 6.5` |
| `conv_imagen` | Proyecta la teselación sobre la imagen con interpolación bilineal | `O(N · A · Cᵢₙ · C_out)` |

### 🧬 Autograd en C puro — sin PyTorch

El núcleo tiene su propio sistema de diferenciación automática escrito en C. Cada operación registra su función de gradiente en un tape. Cuando llamas a `tc_backward()`, el grafo se recorre en orden topológico inverso aplicando la regla de la cadena. Sin recolector de basura, sin magia. Solo C, punteros y una pila.

```
  ┌───────────────────────────────┐
  │  tc_tensor                    │
  │  ├── data      ← float*      │
  │  ├── grad      ← float*      │
  │  └── autograd  ─────────────►│──► _nodo_autograd
  └───────────────────────────────┘         │
                                             ├── backward_fn  ← regla de la cadena
                                             ├── inputs[]     ← tensores padre
                                             └── saved        ← intermedios
```

Operaciones con gradiente soportadas: Add, Sub, Mul, Div, Neg, Pow, Exp, Log, Sqrt, MatMul, Conv2D, Pooling, Reshape, Transpose, ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, BatchNorm, Dropout, CrossEntropy, MSE, BCE — y **ConvPenrose**.

### 📦 Formato .ax — serialización binaria propia

Los modelos se guardan en `.ax`, un formato binario compacto con checksum CRC32. Si el archivo está corrupto, el núcleo se niega a cargarlo. Sin preguntas.

```
  ┌──────────────────────────────────────────┐
  │  "TESELCORE_AX"   ← magic bytes         │
  │  version          ← u32                 │
  │  n_tensores       ← u64                 │
  │  [ nombre · tipo · shape · data ] × N   │
  │  crc32            ← checksum final      │
  └──────────────────────────────────────────┘
```

### 🧠 PenroseNet — la arquitectura experimental

```
  ┌──────────┐   ┌────────────┐   ┌───────────────┐   ┌──────────┐
  │  Entrada │──►│ imagen →   │──►│ PenroseBlock  │──►│ Dual     │──►
  │  28×28   │   │ teselación │   │ (residual)    │   │ Pool     │
  └──────────┘   └────────────┘   └───────────────┘   └────┬─────┘
                                                            │
                         ┌──────────┐   ┌──────────┐       │
                         │  Salida  │◄──│  FC      │◄──────┘
                         │  10      │   │  256→10  │
                         └──────────┘   └──────────┘

  Optimización híbrida:
  ┌─────────────────────────────────────────────────┐
  │  Penrose  →  gradientes numéricos (diferencias  │
  │              centrales, LR × 0.5)               │
  │  Denso    →  autograd estándar (LR normal)      │
  │  Scheduler coseno sobre ambos grupos            │
  └─────────────────────────────────────────────────┘
```

**PenroseBlock:** LayerNorm → ConvPenroseGrafo → GELU → Dropout → conexión residual.  
**DualPool:** pooling independiente por tipo de baldosa (kite vs dart), aprovechando la dualidad geométrica de Penrose.

### ⚡ Tabla de complejidad

| Operación | Tiempo | Espacio |
|-----------|--------|---------|
| Generar teselación nivel L | `O(φ²ᴸ)` | `O(φ²ᴸ)` |
| Construir grafo de vecindad | `O(N log N)` | `O(N · d)` |
| ConvPenrose forward | `O(N · d · Cᵢₙ · C_out)` | `O(N · C_out)` |
| ConvPenrose backward | `O(N · d · Cᵢₙ · C_out)` | `O(N · d)` |
| Guardar / cargar .ax | `O(params)` | `O(params)` |

Para MNIST (28×28): PenroseNet tiene ~15× menos parámetros que una CNN clásica comparable — 70μs/imagen vs 2400μs. La precisión (32% vs 83%) todavía no está a la par, pero la geometría es prometedora. Por eso esto es un laboratorio, no un producto.

---

## 🚀 Instalación

Necesitas lo básico:

- Un compilador C (`gcc`, `clang` o MSVC)
- `make` (en Windows usa MSYS2 o similar)
- Python 3.8+

**1. Compilar el núcleo C:**

```bash
cd teselcore-nucleo
make
```

**2. Instalar el paquete Python:**

```bash
cd teselcore-python
pip install -e .
```

**3. (Opcional) Entorno para experimentos:**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
pip install -r experimentos/requirements.txt
```

---

## ⚡ Antes de entrenar — lee esto

> [!WARNING]
> **La complejidad crece con φ, no con N. Eso importa.**
>
> El costo del forward pass completo es:
>
> **O(B · C_out · C_in · Tₙ · Hₛ · Wₛ)**
>
> donde **Tₙ ≈ φ²ⁿ · 10** es el número de tejas al nivel `n`, y **φ ≈ 1.618**.
> Cada nivel no suma tejas — las multiplica por φ² ≈ 2.618.
> Encima, el gradiente numérico ejecuta **2 forward passes por cada parámetro**.
> El costo se apila rápido.

> [!TIP]
> **Configuración mínima — la que realmente funciona:**
> ```bash
> python entrenar_penrose.py \
>   --nivel   1   \
>   --batch   256 \
>   --canales 16  \
>   --bloques 2   \
>   --K       3   \
>   --eps_num 1e-3
> ```
> Con `--nivel 1` tienes ~20 tejas. Con `--batch 256` el gradiente numérico
> tiene suficientes muestras para no ser puro ruido.
> Es el punto de entrada más bajo que tiene sentido intentar.

> [!NOTE]
> Si ni con esa configuración arranca en tu máquina — abre un issue o escríbeme.
> Sé que la complejidad tiene solución, todavía no encontré la forma de bajarla
> sin romper el núcleo. Es trabajo en progreso, y si tienes ideas, más razón para abrir ese PR.
>
> 📧 [axedus06@gmail.com](mailto:axedus06@gmail.com) · [Issues](https://github.com/AEUS-06/TeselCore/issues)

---

## ⚡ Uso rápido

**Desde la CLI:**

```bash
cd teselcore-nucleo
./teselcore_cli conv_demo 2          # Unix
.\teselcore_cli.exe conv_demo 2      # Windows
```

**Desde Python:**

```python
from teselcore import TeselCore
core = TeselCore()
core.run_demo()
```

**Experimentos MNIST:**

> [!WARNING]
> **La complejidad crece con φ — no con N. Eso importa.**
>
> El costo del forward pass está dado por:
>
> $$O(B \cdot C_{out} \cdot C_{in} \cdot T_n \cdot H_s \cdot W_s)$$
>
> donde $T_n \sim \varphi^{2n} \cdot 10$ es el número de tejas al nivel $n$, y $\varphi \approx 1.618$.
> Cada nivel no suma tejas — las multiplica por $\varphi^2 \approx 2.618$.
> Encima, el gradiente numérico ejecuta **2 forward passes por cada parámetro**.
> El costo se apila rápido.

> [!TIP]
> **Configuración mínima — la que realmente funciona:**
> ```bash
> python entrenar.py \
>   --nivel   1   \
>   --batch   256 \
>   --canales 16  \
>   --bloques 2   \
>   --K       3   \
>   --eps_num 1e-3
> ```
> Con `--nivel 1` tienes ~20 tejas. Con `--batch 256` el gradiente numérico
> tiene suficientes muestras para no ser puro ruido. Es el punto de entrada más bajo
> que tiene sentido intentar.

> [!NOTE]
> Si ni con esa configuración arranca en tu máquina — abre un issue o escríbeme directo.
> Sé que la complejidad tiene solución, todavía no encontré la forma de bajarla
> sin romper el núcleo. Es trabajo en progreso, y si tienes ideas, más razón para abrir ese PR.
>
> 📧 [axedus06@gmail.com](mailto:axedus06@gmail.com) · [GitHub Issues](https://github.com/AEUS-06/TeselCore/issues)

```bash
cd experimentos/mnist
python entrenar_penrose.py
python probar_modelos.py
python graficar_resultados.py
```

---

## 🧪 Tests

```bash
# Núcleo C
cd teselcore-nucleo
make test

# Python
pytest
```

---

## 🌀 El mapa de ideas locas

> Esto no es un roadmap. No tiene fechas, no tiene orden, no tiene garantías.  
> Es una lista de cosas que me quitan el sueño y que en algún momento  
> voy a intentar meter en código. Pura curiosidad, sin más.

```
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                                                                      ║
  ║    🌌 QFT   ·   🎵 Música   ·   🍄 Micelio   ·   ⚡ EM   ·   🌡️ TD  ║
  ║                                                                      ║
  ║              🪐 Gravedad   ·   🔗 Topología   ·   💡 ???             ║
  ║                                                                      ║
  ╚══════════════════════════════════════════════════════════════════════╝
```

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🌌  TEORÍA CUÁNTICA DE CAMPOS                      │
  └─────────────────────────────────────────────────────┘
```
Los físicos llevan décadas usando QFT para describir partículas como vibraciones en campos que llenan todo el espacio. Yo llevo un rato pensando que los pesos de una red neuronal se parecen sospechosamente a eso. ¿Qué pasaría si la propagación hacia adelante tuviera una formulación lagrangiana? ¿Si el backprop fuera una integral de camino? No sé la respuesta todavía. Eso es exactamente por qué quiero intentarlo.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🎵  RESONANCIAS Y MÚSICA                           │
  └─────────────────────────────────────────────────────┘
```
El sonido es matemática que puedes escuchar. Y la música que suena *bien* tiene razones físicas y geométricas detrás que van mucho más allá de "es bonito". Los armónicos, las razones de frecuencia, la estructura de los intervalos — todo eso es geometría disfrazada. ¿Puede una red aprender a representar información en términos de resonancias en lugar de píxeles o tokens? No tengo idea. Pero me parece una pregunta demasiado buena para no hacerla.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🍄  EL MICELIO DE LOS HONGOS                       │
  └─────────────────────────────────────────────────────┘
```
Las redes de micelio llevan millones de años resolviendo problemas de optimización que nosotros apenas estamos aprendiendo a formalizar. Sin un nodo central que "piense", sin instrucciones explícitas — solo química y geometría. Es un grafo de cómputo que la evolución diseñó sin saber que estaba diseñando un grafo de cómputo. Hay algo ahí. No sé exactamente qué. Pero hay algo.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  ⚡  ELECTROMAGNETISMO                              │
  └─────────────────────────────────────────────────────┘
```
Las ecuaciones de Maxwell describen cómo los campos eléctricos y magnéticos se propagan, se acoplan y se influencian mutuamente a través del espacio. Hay una estructura ahí — ondas que viajan, campos que se inducen, energía que fluye — que se parece bastante a cómo la información viaja en una red profunda. ¿Tiene sentido construir capas que se comporten como conductores, inductores o capacitores? Probablemente sí. ¿Alguien lo ha hecho bien? Todavía no estoy convencido.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🪐  GRAVEDAD                                       │
  └─────────────────────────────────────────────────────┘
```
La relatividad general describe la gravedad no como una fuerza sino como curvatura del espacio-tiempo. La geometría manda. Y eso es exactamente lo que hace interesante pensar en redes que operen sobre espacios curvos en lugar de espacios planos. El espacio hiperbólico ya se usa en algunos modelos para representar jerarquías — pero eso es apenas la superficie. La pregunta de verdad es qué más esconde la geometría riemanniana que todavía no hemos metido a un modelo.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🌡️  TERMODINÁMICA                                  │
  └─────────────────────────────────────────────────────┘
```
Entropía, temperatura, equilibrio, transiciones de fase. La termodinámica lleva dos siglos describiendo cómo los sistemas complejos evolucionan hacia estados de mínima energía — que es, curiosamente, lo mismo que hace un optimizador de gradiente descendente. No es una analogía superficial. Es casi la misma matemática vista desde ángulos distintos. Los modelos de Ising, la energía libre de Helmholtz, el principio de máxima entropía — todo eso tiene algo que decirle a una función de pérdida.

<br>

```
  ┌─────────────────────────────────────────────────────┐
  │  🔗  TOPOLOGÍA ALGEBRAICA                           │
  └─────────────────────────────────────────────────────┘
```
¿Y si en lugar de medir distancias medimos *formas*? La homología persistente puede decirte cómo es la estructura de un conjunto de datos sin coordenadas, sin métrica, sin nada euclidiano. Para ciertos tipos de datos eso es exactamente lo que necesitas — y aún así casi nadie lo usa en ML. Yo creo que eso es porque es difícil, no porque sea una mala idea.

<br>

```
  ╔═════════════════════════════════════════════════════╗
  ║  💡  TU IDEA AQUÍ                                   ║
  ╚═════════════════════════════════════════════════════╝
```
Si llegaste leyendo hasta acá, probablemente tienes algo raro dando vueltas en la cabeza. Abre un issue. En serio, no importa qué tan descabellado suene — en este repositorio eso es exactamente un punto a favor.

---

## 📖 Citación

Si usas TeselCore en trabajos académicos o proyectos derivados:

```bibtex
@misc{atekokoliAEUS2026teselcore,
  author       = {Axel Eduardo Urbina Secundino},
  title        = {Fundamentos Matemáticos de la Convolución Aperiódica
                  basada en Teselación de Penrose para Redes Neuronales Ligeras},
  year         = {2026},
  note         = {Preprint técnico},
  howpublished = {\url{https://github.com/AEUS-06/TeselCore}}
}
```

---

## 🤝 ¿Quieres contribuir?

Si llegaste hasta aquí es porque algo de esto te pareció interesante. Eso ya es suficiente razón.

1. Haz un fork y crea una rama descriptiva
2. Agrega tests para cualquier cambio funcional
3. Abre un Pull Request a `dev` — no a `main` — con contexto y pasos para reproducir

¿Tienes una idea rara que no sabes si encaja? Abre un issue de todas formas. Las ideas raras son exactamente lo que este proyecto necesita.

Y si usas TeselCore en algún proyecto, **menciónanoslo en redes** — me encantaría ver qué construyes con esto.

---

## 📜 Licencia

TeselCore se distribuye bajo **GNU GPL v3.0**.  
Consulta el archivo `LICENSE` para más detalles.

---

## 📬 Contacto

¿Tienes una idea rara, una pregunta o simplemente quieres hablar de geometría aperiódica a las 2am?

<div align="center">

| Canal | Link |
|-------|------|
| 📧 **Email** | [axedus06@gmail.com](mailto:axedus06@gmail.com) |
| 🐙 **GitHub** | [@AEUS-06](https://github.com/AEUS-06) |
| 📸 **Instagram** | [@atekokoli01](https://www.instagram.com/atekokoli01/) |
| 💼 **LinkedIn** | [atkokoli](https://www.linkedin.com/in/atkokoli) |

</div>

O simplemente abre un [issue](https://github.com/AEUS-06/TeselCore/issues) directamente en el repositorio.

---

<div align="center">

*"La geometría no es aburrida. Solo la enseñamos aburrida."*

⬡ ⬡ ⬡

**TeselCore** — Hecho con matemáticas raras y curiosidad genuina.

</div>
