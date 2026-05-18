/* teselcore-nucleo/demos/visualizar_kernel.c
 * Genera un SVG que muestra visualmente cómo fluye la información
 * a través del kernel de Penrose:
 *   - Los puntos de muestreo sobre una imagen de entrada
 *   - La intensidad de cada peso (grosor del contorno de cada teja)
 *   - Las conexiones entre tejas vecinas (aristas del grafo)
 *   - La activación resultante en el modo grafo
 *   - Una visualización del campo receptivo por nivel de deflación
 *
 * Uso:
 *   gcc -O2 demos/visualizar_kernel.c src/tensor.c src/penrose.c \
 *       src/formato_ax.c -Iinclude -lm -o vis_kernel
 *   ./vis_kernel
 *   (genera kernel_flow.svg — abrirlo en el browser)
 */

#include "../include/teselcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ════════════════════════════════════════════════════
 * CONSTANTES DE VISUALIZACIÓN
 * ════════════════════════════════════════════════════ */
#define SVG_W   1200.0f
#define SVG_H    900.0f
#define MARGEN   60.0f
#define PAD_TEXTO 8.0f

/* Paleta de colores */
static const char* COLOR_FONDO    = "#0a0a14";
static const char* COLOR_KITE_F   = "#0d2a40";
static const char* COLOR_KITE_S   = "#00c8e0";
static const char* COLOR_DART_F   = "#2a0d0d";
static const char* COLOR_DART_S   = "#e05000";
static const char* COLOR_ARISTA   = "#2a2a4a";
static const char* COLOR_MUESTREO = "#ffffff";
static const char* COLOR_ACTIVO   = "#7fff00";
static const char* COLOR_TEXTO    = "#aaaacc";
static const char* COLOR_TITULO   = "#00e5ff";

/* ════════════════════════════════════════════════════
 * HELPERS SVG
 * ════════════════════════════════════════════════════ */
static float _interpolar(float a, float b, float t) { return a + (b - a) * t; }
static float _clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

/* Color de gradiente: azul frío (bajo) → amarillo-verde (alto) */
static void _color_calor(float t, char* out) {
    t = _clamp01(t);
    int r, g, b;
    if (t < 0.5f) {
        float s = t * 2.0f;
        r = (int)_interpolar(20,  80, s);
        g = (int)_interpolar(40, 180, s);
        b = (int)_interpolar(180, 60, s);
    } else {
        float s = (t - 0.5f) * 2.0f;
        r = (int)_interpolar(80,  255, s);
        g = (int)_interpolar(180, 220, s);
        b = (int)_interpolar(60,   0, s);
    }
    snprintf(out, 16, "#%02x%02x%02x", r, g, b);
}

/* ════════════════════════════════════════════════════
 * SECCIÓN 1: GEOMETRÍA DE LA TESELACIÓN
 * Muestra la teselación con colores según tipo de teja.
 * ════════════════════════════════════════════════════ */
static void _dibujar_teselacion(FILE* f, const tc_teselacion_penrose* ts,
                                 float cx, float cy, float radio,
                                 const float* pesos_norm,   /* peso normalizado por teja */
                                 int          mostrar_pesos) {
    /* Calcular bounding box */
    float xmin=1e9f, xmax=-1e9f, ymin=1e9f, ymax=-1e9f;
    for (int i = 0; i < ts->num_tejas; i++)
        for (int v = 0; v < 4; v++) {
            float px = ts->tejas[i].puntos[v][0], py = ts->tejas[i].puntos[v][1];
            if (px<xmin) xmin=px; if (px>xmax) xmax=px;
            if (py<ymin) ymin=py; if (py>ymax) ymax=py;
        }
    float escala_x = (radio * 2.0f) / (xmax - xmin + 1e-8f);
    float escala_y = (radio * 2.0f) / (ymax - ymin + 1e-8f);
    float escala   = escala_x < escala_y ? escala_x : escala_y;
    float ox = cx - (xmin + xmax) * 0.5f * escala;
    float oy = cy - (ymin + ymax) * 0.5f * escala;

    /* Dibujar aristas del grafo */
    for (int i = 0; i < ts->num_tejas; i++) {
        float ci_x = ts->tejas[i].centro_x * escala + ox;
        float ci_y = ts->tejas[i].centro_y * escala + oy;
        for (int k = 0; k < ts->tejas[i].num_vecinos; k++) {
            int j = ts->tejas[i].vecinos[k];
            if (j > i) {
                float cj_x = ts->tejas[j].centro_x * escala + ox;
                float cj_y = ts->tejas[j].centro_y * escala + oy;
                fprintf(f,
                    "  <line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\""
                    " stroke=\"%s\" stroke-width=\"0.6\" opacity=\"0.4\"/>\n",
                    ci_x, ci_y, cj_x, cj_y, COLOR_ARISTA);
            }
        }
    }

    /* Dibujar polígonos de cada teja */
    for (int i = 0; i < ts->num_tejas; i++) {
        const tc_teja_penrose* tj = &ts->tejas[i];
        const char* cf = tj->tipo == TC_PENROSE_KITE ? COLOR_KITE_F : COLOR_DART_F;
        const char* cs = tj->tipo == TC_PENROSE_KITE ? COLOR_KITE_S : COLOR_DART_S;

        float grosor = 0.5f;
        char color_relleno[20];
        snprintf(color_relleno, sizeof(color_relleno), "%s", cf);

        if (mostrar_pesos && pesos_norm) {
            _color_calor(pesos_norm[i], color_relleno);
            grosor = 0.5f + pesos_norm[i] * 2.0f;
        }

        fprintf(f, "  <polygon points=\"");
        for (int v = 0; v < 4; v++) {
            float px = tj->puntos[v][0] * escala + ox;
            float py = tj->puntos[v][1] * escala + oy;
            fprintf(f, "%.2f,%.2f%s", px, py, v < 3 ? " " : "");
        }
        fprintf(f,
            "\" fill=\"%s\" stroke=\"%s\" stroke-width=\"%.2f\" opacity=\"0.88\"/>\n",
            color_relleno, cs, grosor);

        /* Punto de muestreo central */
        float px = tj->centro_x * escala + ox;
        float py = tj->centro_y * escala + oy;
        fprintf(f,
            "  <circle cx=\"%.2f\" cy=\"%.2f\" r=\"1.5\""
            " fill=\"%s\" opacity=\"0.8\"/>\n",
            px, py, COLOR_MUESTREO);
    }
}

/* ════════════════════════════════════════════════════
 * SECCIÓN 2: CAMPO RECEPTIVO POR NIVEL
 * Muestra cómo el campo receptivo del kernel crece con el nivel.
 * ════════════════════════════════════════════════════ */
static void _dibujar_comparativa_niveles(FILE* f, float y_base) {
    const char* etiquetas[] = {"Nivel 0\n10 tejas", "Nivel 1\n~26 tejas", "Nivel 2\n~68 tejas"};
    float xs[] = {200.0f, 600.0f, 1000.0f};
    float radios[] = {65.0f, 100.0f, 130.0f};

    for (int n = 0; n <= 2; n++) {
        tc_teselacion_penrose* ts = tc_crear_teselacion(n, 10.0f);
        _dibujar_teselacion(f, ts, xs[n], y_base, radios[n], NULL, 0);

        /* Etiqueta */
        fprintf(f,
            "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
            " fill=\"%s\" font-size=\"11\" font-family=\"monospace\">"
            "Nivel %d  (%d tejas)</text>\n",
            xs[n], y_base + radios[n] + 20.0f, COLOR_TEXTO, n, ts->num_tejas);

        tc_liberar_teselacion(ts);
    }
}

/* ════════════════════════════════════════════════════
 * SECCIÓN 3: CALOR DE ACTIVACIONES
 * Muestra la activación por teja después de un forward pass.
 * ════════════════════════════════════════════════════ */
static void _dibujar_activaciones(FILE* f,
                                   const tc_teselacion_penrose* ts,
                                   const float* activaciones,
                                   float cx, float cy, float radio) {
    int N = ts->num_tejas;

    /* Normalizar activaciones a [0,1] */
    float mn = activaciones[0], mx = activaciones[0];
    for (int i = 1; i < N; i++) {
        if (activaciones[i] < mn) mn = activaciones[i];
        if (activaciones[i] > mx) mx = activaciones[i];
    }
    float rng = mx - mn + 1e-8f;
    float* norm = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) norm[i] = (activaciones[i] - mn) / rng;

    _dibujar_teselacion(f, ts, cx, cy, radio, norm, 1);
    free(norm);
}

/* ════════════════════════════════════════════════════
 * SECCIÓN 4: LEYENDA Y ANOTACIONES
 * ════════════════════════════════════════════════════ */
static void _dibujar_leyenda(FILE* f, float x, float y) {
    float dy = 22.0f;

    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"13\""
        " font-family=\"monospace\" font-weight=\"bold\">"
        "Leyenda</text>\n", x, y, COLOR_TITULO);
    y += dy;

    /* Kite */
    fprintf(f,
        "  <rect x=\"%.2f\" y=\"%.2f\" width=\"20\" height=\"12\""
        " fill=\"%s\" stroke=\"%s\" stroke-width=\"1\"/>\n",
        x, y - 10, COLOR_KITE_F, COLOR_KITE_S);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"10\""
        " font-family=\"monospace\">Kite (cometa) — φ y 1</text>\n",
        x + 26, y, COLOR_TEXTO);
    y += dy;

    /* Dart */
    fprintf(f,
        "  <rect x=\"%.2f\" y=\"%.2f\" width=\"20\" height=\"12\""
        " fill=\"%s\" stroke=\"%s\" stroke-width=\"1\"/>\n",
        x, y - 10, COLOR_DART_F, COLOR_DART_S);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"10\""
        " font-family=\"monospace\">Dart (dardo)  — φ y 1</text>\n",
        x + 26, y, COLOR_TEXTO);
    y += dy;

    /* Punto de muestreo */
    fprintf(f,
        "  <circle cx=\"%.2f\" cy=\"%.2f\" r=\"4\" fill=\"%s\"/>\n",
        x + 10, y - 5, COLOR_MUESTREO);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"10\""
        " font-family=\"monospace\">Punto de muestreo</text>\n",
        x + 26, y, COLOR_TEXTO);
    y += dy;

    /* Arista */
    fprintf(f,
        "  <line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\""
        " stroke=\"%s\" stroke-width=\"1.5\"/>\n",
        x, y - 4, x + 20, y - 4, COLOR_ARISTA);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"10\""
        " font-family=\"monospace\">Conexión vecinos (grafo)</text>\n",
        x + 26, y, COLOR_TEXTO);
    y += dy * 1.5f;

    /* Gradiente de calor */
    fprintf(f, "  <defs><linearGradient id=\"grad_calor\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">"
               "<stop offset=\"0\" stop-color=\"#142a50\"/>"
               "<stop offset=\"0.5\" stop-color=\"#10b0a0\"/>"
               "<stop offset=\"1\" stop-color=\"#ffc800\"/>"
               "</linearGradient></defs>\n");
    fprintf(f,
        "  <rect x=\"%.2f\" y=\"%.2f\" width=\"80\" height=\"10\""
        " fill=\"url(#grad_calor)\"/>\n", x, y - 8);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"9\""
        " font-family=\"monospace\">bajo</text>\n", x, y + 6, COLOR_TEXTO);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"9\""
        " font-family=\"monospace\">alto</text>\n", x + 62, y + 6, COLOR_TEXTO);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"10\""
        " font-family=\"monospace\">Activación / peso</text>\n",
        x + 90, y, COLOR_TEXTO);
}

/* ════════════════════════════════════════════════════
 * FUNCIÓN PRINCIPAL
 * ════════════════════════════════════════════════════ */
int main(void) {
    tc_semilla_aleatoria((uint64_t)time(NULL));

    const char* ruta_svg = "kernel_flow.svg";
    FILE* f = fopen(ruta_svg, "w");
    if (!f) { perror("No se pudo abrir kernel_flow.svg"); return 1; }

    /* Cabecera SVG */
    fprintf(f,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\""
        " width=\"%.0f\" height=\"%.0f\""
        " viewBox=\"0 0 %.0f %.0f\">\n",
        SVG_W, SVG_H, SVG_W, SVG_H);

    /* Fondo */
    fprintf(f,
        "  <rect width=\"100%%\" height=\"100%%\" fill=\"%s\"/>\n",
        COLOR_FONDO);

    /* ── Título ──────────────────────────────────────── */
    fprintf(f,
        "  <text x=\"%.2f\" y=\"34\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"16\" font-family=\"monospace\""
        " font-weight=\"bold\">"
        "Kernel de Penrose — Flujo de información</text>\n",
        SVG_W * 0.5f, COLOR_TITULO);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"52\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"10\" font-family=\"monospace\">"
        "TeselCore v0.1.0 — Investigación experimental</text>\n",
        SVG_W * 0.5f, COLOR_TEXTO);

    /* ── ZONA 1: Comparativa de niveles (fila superior) ── */
    fprintf(f,
        "  <text x=\"%.2f\" y=\"90\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"12\" font-family=\"monospace\">"
        "Crecimiento del campo receptivo por nivel de deflación</text>\n",
        SVG_W * 0.5f, COLOR_TITULO);

    _dibujar_comparativa_niveles(f, 240.0f);

    /* ── ZONA 2: Kernel con pesos y activaciones ──────── */
    float titulo_y2 = 440.0f;
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"12\" font-family=\"monospace\">"
        "Pesos del kernel (izq) y activaciones tras forward pass (der)</text>\n",
        SVG_W * 0.5f, titulo_y2, COLOR_TITULO);

    /* Crear kernel de nivel 2 para la visualización detallada */
    int nivel_vis = 2;
    float escala_vis = 8.0f;
    tc_kernel_penrose* kernel = tc_crear_kernel_penrose(1, 1, nivel_vis, escala_vis);
    int N = kernel->teselacion->num_tejas;

    /* Pesos normalizados */
    float* pesos_np = (float*)kernel->pesos->datos;
    float p_min = pesos_np[0], p_max = pesos_np[0];
    for (int i = 1; i < N; i++) {
        if (pesos_np[i] < p_min) p_min = pesos_np[i];
        if (pesos_np[i] > p_max) p_max = pesos_np[i];
    }
    float p_rng = p_max - p_min + 1e-8f;
    float* pesos_norm = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) pesos_norm[i] = (pesos_np[i] - p_min) / p_rng;

    /* Panel izquierdo: pesos del kernel */
    float cx_izq = SVG_W * 0.27f, cy_vis = titulo_y2 + 190.0f, radio_vis = 155.0f;
    _dibujar_teselacion(f, kernel->teselacion, cx_izq, cy_vis, radio_vis, pesos_norm, 1);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"10\" font-family=\"monospace\">"
        "Pesos W[t]  (calor = magnitud)</text>\n",
        cx_izq, cy_vis + radio_vis + 18.0f, COLOR_TEXTO);

    /* Simular un forward pass con entrada aleatoria */
    int forma_in[] = {1, 1, N};
    tc_tensor* entrada = tc_aleatorio_uniforme(forma_in, 3);
    tc_tensor* salida  = tc_conv_penrose_grafo(
        entrada, kernel->pesos, kernel->sesgo, kernel->teselacion);

    /* Panel derecho: activaciones */
    float cx_der = SVG_W * 0.68f;
    float* act_datos = (float*)salida->datos;
    _dibujar_activaciones(f, kernel->teselacion, act_datos, cx_der, cy_vis, radio_vis);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"10\" font-family=\"monospace\">"
        "Activaciones Y[t]  (tras un forward pass)</text>\n",
        cx_der, cy_vis + radio_vis + 18.0f, COLOR_TEXTO);

    /* Flecha entre paneles */
    float ax1 = cx_izq + radio_vis * 0.85f;
    float ax2 = cx_der - radio_vis * 0.85f;
    float ay  = cy_vis;
    fprintf(f,
        "  <defs><marker id=\"fl\" markerWidth=\"8\" markerHeight=\"8\""
        " refX=\"6\" refY=\"3\" orient=\"auto\">"
        "<path d=\"M0,0 L0,6 L8,3 z\" fill=\"%s\"/>"
        "</marker></defs>\n", COLOR_KITE_S);
    fprintf(f,
        "  <line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\""
        " stroke=\"%s\" stroke-width=\"2\""
        " marker-end=\"url(#fl)\"/>\n",
        ax1, ay, ax2, ay, COLOR_KITE_S);
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"9\" font-family=\"monospace\">"
        "conv_penrose_grafo</text>\n",
        (ax1 + ax2) * 0.5f, ay - 8.0f, COLOR_KITE_S);

    /* ── Anotaciones matemáticas ──────────────────────── */
    float ann_x = SVG_W * 0.42f, ann_y = cy_vis - 60.0f;
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"10\" font-family=\"monospace\">"
        "Y[b,co,i] = b[co] + Σ_c Σ_k  W[co,c,k] · X[b,c,ν_k(i)]"
        "</text>\n",
        SVG_W * 0.5f, ann_y, COLOR_ACTIVO);

    /* Estadísticas */
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\""
        " fill=\"%s\" font-size=\"9\" font-family=\"monospace\">"
        "Nivel %d | %d tejas | φ = 1.6180…</text>\n",
        cx_izq - radio_vis, cy_vis + radio_vis + 35.0f,
        COLOR_TEXTO, nivel_vis, N);

    /* ── Leyenda ──────────────────────────────────────── */
    _dibujar_leyenda(f, MARGEN, SVG_H - 140.0f);

    /* ── Fórmulas clave (esquina derecha) ─────────────── */
    float fx = SVG_W - 260.0f, fy = SVG_H - 150.0f, fdy = 18.0f;
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"11\""
        " font-family=\"monospace\" font-weight=\"bold\">"
        "Fundamentos</text>\n", fx, fy, COLOR_TITULO);
    fy += fdy;
    const char* formulas[] = {
        "φ = (1+√5)/2 ≈ 1.618",
        "φ² = φ+1",
        "Kite → 2·Kite + 1·Dart",
        "Dart → 1·Kite + 1·Dart",
        "T_n ~ φ^(2n) · 10",
        "T_K/T_D → φ  (n→∞)",
        "σ_w = √(2 / C·T_n)  [He]",
        NULL
    };
    for (int i = 0; formulas[i]; i++) {
        fprintf(f,
            "  <text x=\"%.2f\" y=\"%.2f\" fill=\"%s\" font-size=\"9\""
            " font-family=\"monospace\">%s</text>\n",
            fx, fy + i * fdy, COLOR_TEXTO, formulas[i]);
    }

    /* ── Pie de página ────────────────────────────────── */
    fprintf(f,
        "  <text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\""
        " fill=\"%s\" font-size=\"8\" font-family=\"monospace\">"
        "TeselCore — GNU LGPL v3 — Investigación abierta y reproducible"
        "</text>\n",
        SVG_W * 0.5f, SVG_H - 8.0f, COLOR_TEXTO);

    fprintf(f, "</svg>\n");
    fclose(f);

    /* Limpiar */
    free(pesos_norm);
    tc_liberar(entrada);
    tc_liberar(salida);
    tc_liberar_kernel_penrose(kernel);

    printf("Visualización generada: %s\n", ruta_svg);
    printf("  Ábrelo en el browser para ver el flujo de información.\n");
    printf("\n  Contenido del SVG:\n");
    printf("  1. Crecimiento del campo receptivo por nivel (0, 1, 2)\n");
    printf("  2. Pesos del kernel de nivel 2 (mapa de calor por teja)\n");
    printf("  3. Activaciones tras un forward pass (conv_penrose_grafo)\n");
    printf("  4. Fórmulas matemáticas clave\n");
    printf("  5. Leyenda completa\n");

    return 0;
}