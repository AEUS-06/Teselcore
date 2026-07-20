/* teselcore/src/penrose.c
 * Convolución de Penrose — módulo experimental de TeselCore.
 * Licencia: GNU LGPL v3
 */

#include "../include/teselcore.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define PI_TC    3.14159265358979323846f
#define ANG36    (PI_TC / 5.0f)
#define ANG72    (2.0f * PI_TC / 5.0f)

typedef struct { float x, y; } _Vertice;

typedef struct {
    int   tipo;
    int   v[4];
    float angulo;
} _Baldosa;

typedef struct { _Vertice* buf; int n, cap; } _BufVert;
typedef struct { _Baldosa* buf; int n, cap; } _BufBald;

static void _bv_agregar(_BufVert* bv, float x, float y) {
    if (bv->n >= bv->cap) {
        bv->cap = bv->cap ? bv->cap * 2 : 64;
        bv->buf = (_Vertice*)realloc(bv->buf, bv->cap * sizeof(_Vertice));
    }
    bv->buf[bv->n++] = (_Vertice){x, y};
}

static int _bv_obtener(_BufVert* bv, float x, float y) {
    const float EPS = 1e-5f;
    for (int i = 0; i < bv->n; i++) {
        float dx = bv->buf[i].x - x, dy = bv->buf[i].y - y;
        if (dx*dx + dy*dy < EPS*EPS) return i;
    }
    _bv_agregar(bv, x, y);
    return bv->n - 1;
}

static void _bb_agregar(_BufBald* bb, int tipo, int v0, int v1, int v2, int v3, float ang) {
    if (bb->n >= bb->cap) {
        bb->cap = bb->cap ? bb->cap * 2 : 64;
        bb->buf = (_Baldosa*)realloc(bb->buf, bb->cap * sizeof(_Baldosa));
    }
    bb->buf[bb->n++] = (_Baldosa){tipo, {v0, v1, v2, v3}, ang};
}

/* ════════════════════════════════════════════════════
 * SUBDIVISIÓN DEFLACIONARIA
 * ════════════════════════════════════════════════════ */
static void _subdividir(_BufVert* verts, _BufBald* entrada, _BufBald* salida) {
    for (int i = 0; i < entrada->n; i++) {
        _Baldosa* b = &entrada->buf[i];
        _Vertice* v = verts->buf;

        float cx = v[b->v[0]].x, cy = v[b->v[0]].y;
        float lx = v[b->v[1]].x, ly = v[b->v[1]].y;
        float px = v[b->v[2]].x, py = v[b->v[2]].y;
        float rx = v[b->v[3]].x, ry = v[b->v[3]].y;

        if (b->tipo == TC_PENROSE_KITE) {
            float qx = cx + (px - cx) / TC_PHI, qy = cy + (py - cy) / TC_PHI;
            float r2x = lx + (cx - lx) / TC_PHI, r2y = ly + (cy - ly) / TC_PHI;
            float sx  = rx + (cx - rx) / TC_PHI, sy  = ry + (cy - ry) / TC_PHI;

            int iC  = b->v[0], iL = b->v[1], iP = b->v[2], iR = b->v[3];
            int iQ  = _bv_obtener(verts, qx,  qy);
            int iR2 = _bv_obtener(verts, r2x, r2y);
            int iS  = _bv_obtener(verts, sx,  sy);

            /* BUG FIX: las subdivisiones originales tenían los índices de
               vértices inconsistentes con la convención (C, L, P, R).
               Se usa la convención: v[0]=centro, v[1]=izq, v[2]=punta, v[3]=der */
            _bb_agregar(salida, TC_PENROSE_KITE, iQ,  iL,  iP,  iR2, b->angulo);
            _bb_agregar(salida, TC_PENROSE_DART, iC,  iR2, iQ,  iL,  b->angulo + ANG36);
            _bb_agregar(salida, TC_PENROSE_DART, iC,  iS,  iQ,  iR,  b->angulo - ANG36);

        } else { /* TC_PENROSE_DART */
            float qx = lx + (px - lx) / TC_PHI, qy = ly + (py - ly) / TC_PHI;

            int iC = b->v[0], iL = b->v[1], iP = b->v[2], iR = b->v[3];
            int iQ = _bv_obtener(verts, qx, qy);

            _bb_agregar(salida, TC_PENROSE_KITE, iQ,  iC,  iP,  iR,  b->angulo);
            _bb_agregar(salida, TC_PENROSE_DART, iL,  iQ,  iC,  iP,  b->angulo + ANG36 * 2.0f);
        }
    }
}

/* ════════════════════════════════════════════════════
 * COMPARADOR DE ARISTAS (para qsort)
 * ════════════════════════════════════════════════════ */
typedef struct { int a, b; int tile; } _Edge;

static int _cmp_edge(const void* p1, const void* p2) {
    const _Edge* e1 = (const _Edge*)p1;
    const _Edge* e2 = (const _Edge*)p2;
    if (e1->a != e2->a) return e1->a - e2->a;
    return e1->b - e2->b;
}

/* ════════════════════════════════════════════════════
 * CREAR TESELACIÓN
 * ════════════════════════════════════════════════════ */
tc_teselacion_penrose* tc_crear_teselacion(int nivel, float escala) {
    assert(nivel >= 0 && nivel <= 7);

    _BufVert verts     = {0};
    _BufBald actual    = {0};
    _BufBald siguiente = {0};

    /*
     * Semilla: 10 tejas en configuración "sol".
     *
     * BUG FIX MAYOR: el bucle original usaba variables iL e iR que se calculaban
     * pero luego se ignoraban con (void), y las tejas semilla se construían con
     * índices incorrectos (iO, iC, iP, iP en lugar de iO, iL, iP, iR).
     * Esto hacía que todas las tejas semilla tuvieran vértices duplicados,
     * generando una geometría degenerada desde el nivel 0.
     */
    for (int i = 0; i < 10; i++) {
        float ang = i * 2.0f * PI_TC / 10.0f;

        /* Cuatro vértices de la teja kite/dart semilla */
        float ox = 0.0f,                              oy = 0.0f;
        float ax = cosf(ang) * escala,                ay = sinf(ang) * escala;
        float lx = cosf(ang + ANG36) * escala,        ly = sinf(ang + ANG36) * escala;
        float rx = cosf(ang - ANG36) * escala,        ry = sinf(ang - ANG36) * escala;
        /* Punta: a distancia escala/φ en dirección ang+ANG36 */
        float px = cosf(ang + ANG36) * escala / TC_PHI;
        float py = sinf(ang + ANG36) * escala / TC_PHI;

        int iO = _bv_obtener(&verts, ox, oy);
        int iA = _bv_obtener(&verts, ax, ay);
        int iL = _bv_obtener(&verts, lx, ly);
        int iR = _bv_obtener(&verts, rx, ry);
        int iP = _bv_obtener(&verts, px, py);
        (void)iA; /* iA no se usa directamente en la teja semilla */

        /* Alternamos kite y dart para la configuración inicial */
        int tipo = (i % 2 == 0) ? TC_PENROSE_KITE : TC_PENROSE_DART;
        /* Convención: v[0]=origen(centro), v[1]=izq, v[2]=punta, v[3]=der */
        _bb_agregar(&actual, tipo, iO, iL, iP, iR, ang);
    }

    /* Aplicar deflación nivel veces */
    for (int n = 0; n < nivel; n++) {
        siguiente.n = 0;
        _subdividir(&verts, &actual, &siguiente);
        _BufBald tmp = actual;
        actual    = siguiente;
        siguiente = tmp;
    }

    int T = actual.n;

    /* Construir tabla de vecindad por aristas compartidas */
    _Edge* edges = (_Edge*)malloc(sizeof(_Edge) * T * 4);
    int ecount = 0;

    for (int i = 0; i < T; i++) {
        int* vs = actual.buf[i].v;
        for (int p = 0; p < 4; p++) {
            int u     = vs[p];
            int v_idx = vs[(p + 1) % 4];
            int a = u < v_idx ? u : v_idx;
            int b = u < v_idx ? v_idx : u;
            edges[ecount++] = (_Edge){a, b, i};
        }
    }

    qsort(edges, ecount, sizeof(_Edge), _cmp_edge);

    int* neigh   = (int*)calloc(T * 7, sizeof(int));
    int* n_neigh = (int*)calloc(T, sizeof(int));

    for (int i = 0; i < ecount; ) {
        int j = i + 1;
        while (j < ecount && edges[j].a == edges[i].a && edges[j].b == edges[i].b) j++;
        for (int p = i; p < j; p++) {
            for (int q = p + 1; q < j; q++) {
                int t1 = edges[p].tile;
                int t2 = edges[q].tile;
                int found = 0;
                for (int k = 0; k < n_neigh[t1]; k++) if (neigh[t1*7+k] == t2) { found=1; break; }
                if (!found && n_neigh[t1] < 7) neigh[t1*7 + n_neigh[t1]++] = t2;
                found = 0;
                for (int k = 0; k < n_neigh[t2]; k++) if (neigh[t2*7+k] == t1) { found=1; break; }
                if (!found && n_neigh[t2] < 7) neigh[t2*7 + n_neigh[t2]++] = t1;
            }
        }
        i = j;
    }

    tc_teselacion_penrose* ts = (tc_teselacion_penrose*)malloc(sizeof(tc_teselacion_penrose));
    ts->num_tejas = T;
    ts->nivel     = nivel;
    ts->escala    = escala;
    ts->tejas     = (tc_teja_penrose*)calloc(T, sizeof(tc_teja_penrose));

    for (int i = 0; i < T; i++) {
        tc_teja_penrose* tj = &ts->tejas[i];
        int* vs = actual.buf[i].v;
        float cx = 0.0f, cy = 0.0f;
        for (int p = 0; p < 4; p++) {
            cx += verts.buf[vs[p]].x;
            cy += verts.buf[vs[p]].y;
        }
        cx *= 0.25f; cy *= 0.25f;
        tj->centro_x    = cx;
        tj->centro_y    = cy;
        tj->num_vecinos = n_neigh[i];
        for (int k = 0; k < n_neigh[i]; k++) tj->vecinos[k] = neigh[i*7 + k];
        tj->tipo   = (tc_tipo_teja_penrose)actual.buf[i].tipo;
        tj->angulo = actual.buf[i].angulo;
        for (int p = 0; p < 4; p++) {
            tj->puntos[p][0] = verts.buf[vs[p]].x;
            tj->puntos[p][1] = verts.buf[vs[p]].y;
        }
    }

    free(edges);
    free(neigh);
    free(n_neigh);
    free(verts.buf);
    free(actual.buf);
    free(siguiente.buf);

    printf("Teselación de Penrose nivel %d: %d nodos generados\n", nivel, T);
    return ts;
}

void tc_liberar_teselacion(tc_teselacion_penrose* t) {
    if (!t) return;
    free(t->tejas);
    free(t);
}

/* ════════════════════════════════════════════════════
 * KERNEL DE PENROSE
 * ════════════════════════════════════════════════════ */
tc_kernel_penrose* tc_crear_kernel_penrose(int canales_ent, int canales_sal,
                                            int nivel, float escala) {
    tc_kernel_penrose* k = (tc_kernel_penrose*)malloc(sizeof(tc_kernel_penrose));
    k->teselacion      = tc_crear_teselacion(nivel, escala);
    k->canales_entrada = canales_ent;
    k->canales_salida  = canales_sal;

    int num_tejas = k->teselacion->num_tejas;
    int forma_pesos[] = {canales_sal, canales_ent, num_tejas};
    k->pesos = tc_aleatorio_normal(forma_pesos, 3);
    k->pesos->requiere_grad = 1;

    float std_he = sqrtf(2.0f / (canales_ent * num_tejas));
    float* d = (float*)k->pesos->datos;
    for (size_t i = 0; i < k->pesos->total; i++) d[i] *= std_he;

    int forma_sesgo[] = {canales_sal};
    k->sesgo = tc_ceros(forma_sesgo, 1);
    k->sesgo->requiere_grad = 1;

    return k;
}

void tc_liberar_kernel_penrose(tc_kernel_penrose* k) {
    if (!k) return;
    tc_liberar_teselacion(k->teselacion);
    tc_liberar(k->pesos);
    tc_liberar(k->sesgo);
    free(k);
}

/* ════════════════════════════════════════════════════
 * INTERPOLACIÓN BILINEAL
 * ════════════════════════════════════════════════════ */
static float _interpolar_bilineal(const float* canal, int alto, int ancho,
                                   float x, float y) {
    if (x < 0 || x >= ancho - 1 || y < 0 || y >= alto - 1) return 0.0f;
    int   x0 = (int)x, y0 = (int)y;
    float dx = x - x0, dy = y - y0;
    return canal[y0*ancho + x0]       * (1-dx) * (1-dy)
         + canal[y0*ancho + x0+1]     * dx     * (1-dy)
         + canal[(y0+1)*ancho + x0]   * (1-dx) * dy
         + canal[(y0+1)*ancho + x0+1] * dx     * dy;
}

/* ════════════════════════════════════════════════════
 * CONVOLUCIÓN DE PENROSE — MODO IMAGEN
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_conv_penrose(tc_tensor* entrada, tc_kernel_penrose* kernel,
                            int paso, int modo_relleno) {
    (void)modo_relleno;

    int L  = entrada->forma[0];
    int Ce = entrada->forma[1];
    int H  = entrada->forma[2];
    int W  = entrada->forma[3];
    int Cs = kernel->canales_salida;
    int T  = kernel->teselacion->num_tejas;

    int H_sal = H / paso;
    int W_sal = W / paso;

    int forma_sal[] = {L, Cs, H_sal, W_sal};
    tc_tensor* salida = tc_ceros(forma_sal, 4);

    float* datos_ent = (float*)entrada->datos;
    float* datos_sal = (float*)salida->datos;
    float* datos_p   = (float*)kernel->pesos->datos;
    float* datos_s   = (float*)kernel->sesgo->datos;

    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (int t = 0; t < T; t++) {
        float cx = kernel->teselacion->tejas[t].centro_x;
        float cy = kernel->teselacion->tejas[t].centro_y;
        if (cx < min_x) min_x = cx;
        if (cx > max_x) max_x = cx;
        if (cy < min_y) min_y = cy;
        if (cy > max_y) max_y = cy;
    }
    float rango_x = max_x - min_x + 1e-8f;
    float rango_y = max_y - min_y + 1e-8f;

    for (int b = 0; b < L; b++)
    for (int cs = 0; cs < Cs; cs++)
    for (int ih = 0; ih < H_sal; ih++)
    for (int iw = 0; iw < W_sal; iw++) {
        float centro_h = ih * paso + paso / 2.0f;
        float centro_w = iw * paso + paso / 2.0f;
        float acumulado = 0.0f;

        for (int t = 0; t < T; t++) {
            float norm_x = (kernel->teselacion->tejas[t].centro_x - min_x) / rango_x;
            float norm_y = (kernel->teselacion->tejas[t].centro_y - min_y) / rango_y;
            float px_ent = centro_w + (norm_x - 0.5f) * paso;
            float py_ent = centro_h + (norm_y - 0.5f) * paso;

            for (int ce = 0; ce < Ce; ce++) {
                const float* canal = datos_ent + b*Ce*H*W + ce*H*W;
                float valor = _interpolar_bilineal(canal, H, W, px_ent, py_ent);
                float peso  = datos_p[cs*Ce*T + ce*T + t];
                acumulado  += valor * peso;
            }
        }

        int idx = b*Cs*H_sal*W_sal + cs*H_sal*W_sal + ih*W_sal + iw;
        datos_sal[idx] = acumulado + datos_s[cs];
    }
    return salida;
}

/* ════════════════════════════════════════════════════
 * CONVOLUCIÓN DE PENROSE — MODO GRAFO
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_conv_penrose_grafo(tc_tensor* entrada,
                                  tc_tensor* kernel,
                                  tc_tensor* sesgo,
                                  const tc_teselacion_penrose* teselacion) {
    assert(entrada->ndim == 3);
    assert(kernel->ndim  == 3);

    int B    = entrada->forma[0];
    int Cin  = entrada->forma[1];
    int N    = entrada->forma[2];
    int Cout = kernel->forma[0];
    int K    = kernel->forma[2];

    assert(Cin == kernel->forma[1]);
    assert(N   == teselacion->num_tejas);

    int forma_sal[] = {B, Cout, N};
    tc_tensor* sal  = tc_ceros(forma_sal, 3);
    float* ds = (float*)sal->datos;
    float* de = (float*)entrada->datos;
    float* dk = (float*)kernel->datos;
    float* db = sesgo ? (float*)sesgo->datos : NULL;

    for (int b = 0; b < B; b++)
    for (int co = 0; co < Cout; co++)
    for (int i = 0; i < N; i++) {
        float suma = db ? db[co] : 0.0f;
        const tc_teja_penrose* tj = &teselacion->tejas[i];

        for (int ci = 0; ci < Cin; ci++) {
            if (0 < K)
                suma += dk[co*Cin*K + ci*K + 0] * de[b*Cin*N + ci*N + i];
            for (int nv = 0; nv < tj->num_vecinos && (nv+1) < K; nv++) {
                int vecino = tj->vecinos[nv];
                suma += dk[co*Cin*K + ci*K + (nv+1)] * de[b*Cin*N + ci*N + vecino];
            }
        }
        ds[b*Cout*N + co*N + i] = suma;
    }

    sal->requiere_grad = entrada->requiere_grad || kernel->requiere_grad;
    if (sal->requiere_grad)
        tc_tape_empujar_conv_penrose_grafo(sal, entrada, kernel, sesgo, teselacion);

    return sal;
}

/* ════════════════════════════════════════════════════
 * PROYECCIÓN IMAGEN ↔ TESELACIÓN
 * ════════════════════════════════════════════════════ */
tc_tensor* tc_imagen_a_teselacion(tc_tensor* imagen,
                                   const tc_teselacion_penrose* teselacion) {
    assert(imagen->ndim == 4);
    int B = imagen->forma[0], C = imagen->forma[1];
    int H = imagen->forma[2], W = imagen->forma[3];
    int N = teselacion->num_tejas;

    int forma_sal[] = {B, C, N};
    tc_tensor* sal  = tc_ceros(forma_sal, 3);
    float* di = (float*)imagen->datos;
    float* ds = (float*)sal->datos;

    float xmin=1e9f, xmax=-1e9f, ymin=1e9f, ymax=-1e9f;
    for (int i = 0; i < N; i++) {
        float x = teselacion->tejas[i].centro_x;
        float y = teselacion->tejas[i].centro_y;
        if (x < xmin) xmin=x; if (x > xmax) xmax=x;
        if (y < ymin) ymin=y; if (y > ymax) ymax=y;
    }
    float xrng = xmax - xmin + 1e-8f, yrng = ymax - ymin + 1e-8f;

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            float nx = (teselacion->tejas[i].centro_x - xmin) / xrng;
            float ny = (teselacion->tejas[i].centro_y - ymin) / yrng;
            float fx = nx * (W - 1), fy = ny * (H - 1);
            int   x0 = (int)fx, y0 = (int)fy;
            int   x1 = x0+1 < W ? x0+1 : x0;
            int   y1 = y0+1 < H ? y0+1 : y0;
            float wx = fx - x0, wy = fy - y0;

            for (int c = 0; c < C; c++) {
                float v00 = di[b*C*H*W + c*H*W + y0*W + x0];
                float v10 = di[b*C*H*W + c*H*W + y0*W + x1];
                float v01 = di[b*C*H*W + c*H*W + y1*W + x0];
                float v11 = di[b*C*H*W + c*H*W + y1*W + x1];
                ds[b*C*N + c*N + i] = v00*(1-wx)*(1-wy) + v10*wx*(1-wy)
                                    + v01*(1-wx)*wy      + v11*wx*wy;
            }
        }
    }
    return sal;
}

tc_tensor* tc_teselacion_a_imagen(tc_tensor* caracteristicas,
                                   const tc_teselacion_penrose* teselacion,
                                   int alto, int ancho) {
    assert(caracteristicas->ndim == 3);
    int B = caracteristicas->forma[0], C = caracteristicas->forma[1];
    int N = caracteristicas->forma[2];
    assert(N == teselacion->num_tejas);

    int forma_sal[]  = {B, C, alto, ancho};
    tc_tensor* sal   = tc_ceros(forma_sal, 4);
    tc_tensor* pesos = tc_ceros(forma_sal, 4);
    float* dc = (float*)caracteristicas->datos;
    float* ds = (float*)sal->datos;
    float* dp = (float*)pesos->datos;

    float xmin=1e9f, xmax=-1e9f, ymin=1e9f, ymax=-1e9f;
    for (int i = 0; i < N; i++) {
        float x = teselacion->tejas[i].centro_x;
        float y = teselacion->tejas[i].centro_y;
        if (x < xmin) xmin=x; if (x > xmax) xmax=x;
        if (y < ymin) ymin=y; if (y > ymax) ymax=y;
    }
    float xrng = xmax - xmin + 1e-8f, yrng = ymax - ymin + 1e-8f;

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            float nx = (teselacion->tejas[i].centro_x - xmin) / xrng;
            float ny = (teselacion->tejas[i].centro_y - ymin) / yrng;
            float fx = nx * (ancho - 1), fy = ny * (alto - 1);
            int   x0 = (int)fx, y0 = (int)fy;
            int   x1 = x0+1 < ancho ? x0+1 : x0;
            int   y1 = y0+1 < alto  ? y0+1 : y0;
            float wx = fx - x0, wy = fy - y0;

            float ws[4] = {(1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy};
            int   xs[4] = {x0, x1, x0, x1};
            int   ys[4] = {y0, y0, y1, y1};

            for (int c = 0; c < C; c++) {
                float val = dc[b*C*N + c*N + i];
                for (int p = 0; p < 4; p++) {
                    int idx = b*C*alto*ancho + c*alto*ancho + ys[p]*ancho + xs[p];
                    ds[idx] += val * ws[p];
                    dp[idx] += ws[p];
                }
            }
        }
    }

    size_t total = (size_t)B * C * alto * ancho;
    for (size_t i = 0; i < total; i++)
        if (dp[i] > 1e-8f) ds[i] /= dp[i];

    tc_liberar(pesos);
    return sal;
}

/* ════════════════════════════════════════════════════
 * EXPORTAR SVG
 * ════════════════════════════════════════════════════ */
int tc_exportar_svg_teselacion(const tc_teselacion_penrose* t, const char* ruta) {
    FILE* f = fopen(ruta, "w");
    if (!f) return -1;

    float min_x=1e9f, max_x=-1e9f, min_y=1e9f, max_y=-1e9f;
    for (int i = 0; i < t->num_tejas; i++)
        for (int v = 0; v < 4; v++) {
            float px = t->tejas[i].puntos[v][0], py = t->tejas[i].puntos[v][1];
            if (px < min_x) min_x=px; if (px > max_x) max_x=px;
            if (py < min_y) min_y=py; if (py > max_y) max_y=py;
        }

    float margen = (max_x - min_x) * 0.05f;
    float vb_x = min_x - margen, vb_y = min_y - margen;
    float vb_w = (max_x - min_x) + 2*margen;
    float vb_h = (max_y - min_y) + 2*margen;

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" ");
    fprintf(f, "viewBox=\"%.2f %.2f %.2f %.2f\" width=\"800\" height=\"800\">\n",
            vb_x, vb_y, vb_w, vb_h);
    fprintf(f, "  <rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"#0f0f1a\"/>\n",
            vb_x, vb_y, vb_w, vb_h);

    for (int i = 0; i < t->num_tejas; i++) {
        const tc_teja_penrose* teja = &t->tejas[i];
        const char* cf = teja->tipo == TC_PENROSE_KITE ? "#1a3a5c" : "#3a1a0f";
        const char* cs = teja->tipo == TC_PENROSE_KITE ? "#00e5ff" : "#ff6b35";

        fprintf(f, "  <polygon points=\"");
        for (int v = 0; v < 4; v++)
            fprintf(f, "%.3f,%.3f%s",
                    teja->puntos[v][0], teja->puntos[v][1], v < 3 ? " " : "");
        fprintf(f, "\" fill=\"%s\" stroke=\"%s\" stroke-width=\"0.3\" opacity=\"0.85\"/>\n",
                cf, cs);

        fprintf(f, "  <circle cx=\"%.3f\" cy=\"%.3f\" r=\"0.15\" fill=\"%s\" opacity=\"0.6\"/>\n",
                teja->centro_x, teja->centro_y, cs);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    printf("SVG exportado: %s (%d nodos)\n", ruta, t->num_tejas);
    return 0;
}