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
                    ║   T E S E L C O R E  ║
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
├── 🔧 teselcore-nucleo/     ← Motor en C. Rápido, crudo, sin adornos.
│   ├── src/                 ← Código fuente del kernel
│   ├── include/teselcore.h  ← Interfaz pública
│   ├── tests/               ← Tests del núcleo
│   └── Makefile
│
├── 🐍 teselcore-python/     ← Bindings Python para que no tengas que sufrir con C
│   └── teselcore.py
│
├── 🧪 experimentos/         ← Aquí es donde la magia (y el caos) ocurre
│   └── mnist/               ← ¿Penrose vs CNN? Sí, eso existe aquí.
│       ├── entrenar.py
│       ├── probar_modelos.py
│       └── graficar_resultados.py
│
└── 📄 document/             ← Documentación, notas, artefactos matemáticos
```

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

En Windows esto genera `teselcore_cli.exe`. En Unix, el binario correspondiente.

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

## ⚡ Uso rápido

**Desde la CLI del núcleo:**

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
python entrenar.py
python probar_modelos.py
python graficar_resultados.py
```

---

## 🧪 Tests

Para el núcleo C:

```bash
cd teselcore-nucleo
make test
```

Para Python:

```bash
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

Si usas TeselCore en trabajos académicos, investigaciones o proyectos derivados, por favor cita:

```bibtex
@misc{atekokoliAEUS2026teselcore,
  author = {Axel Eduardo Urbina Secundino},
  title  = {Fundamentos Matemáticos de la Convolución Aperiódica basada en
            Teselación de Penrose para Redes Neuronales Ligeras},
  year   = {2026},
  note   = {Preprint técnico},
  howpublished = {\url{https://github.com/AEUS-06/TeselCore}}
}
```

---

## 🤝 ¿Quieres contribuir?

Si llegaste hasta aquí es porque algo de esto te pareció interesante. Eso ya es suficiente razón para contribuir.

1. Haz un fork y crea una rama con nombre descriptivo
2. Agrega tests para cualquier cambio funcional
3. Abre un Pull Request con contexto y pasos para reproducir

¿Tienes una idea rara que no sabes si encaja? Abre un issue de todas formas. Las ideas raras son exactamente lo que este proyecto necesita.

Y si usas TeselCore en algún proyecto o experimento, **menciónanoslo en redes** — me encantaría ver qué construyes con esto.

---

## 📜 Licencia

TeselCore se distribuye bajo **GNU GPL v3.0**.  
Consulta el archivo `LICENSE` para más detalles.

---

## 📬 Contacto

¿Tienes una idea rara, una pregunta o simplemente quieres hablar de geometría aperiódica a las 2am? Aquí me encuentras:

<div align="center">

| Canal | Link |
|-------|------|
| 📧 **Email** | [axedus06@gmail.com](mailto:axedus06@gmail.com) |
| 🐙 **GitHub** | [@AEUS-06](https://github.com/AEUS-06) |
| 📸 **Instagram** | [@atekokoli01](https://www.instagram.com/atekokoli01/) |
| 💼 **LinkedIn** | [atkokoli](https://www.linkedin.com/in/atkokoli) |

</div>

O simplemente abre un [issue](https://github.com/AEUS-06/TeselCore/issues) o un Pull Request directamente en el repositorio.

---

<div align="center">

*"La geometría no es aburrida. Solo la enseñamos aburrida."*

⬡ ⬡ ⬡

**TeselCore** — Hecho con matemáticas raras y curiosidad genuina.

</div>
