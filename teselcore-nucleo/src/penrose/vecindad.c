/* src/penrose/vecindad.c — Tabla de vecindad por aristas compartidas */
#include "../../include/internal/penrose_interno.h"
#include <stdlib.h>

typedef struct { int a, b; int tile; } _Edge;

static int _cmp_edge(const void* p1, const void* p2) {
    const _Edge* e1=(const _Edge*)p1, *e2=(const _Edge*)p2;
    if (e1->a != e2->a) return e1->a - e2->a;
    return e1->b - e2->b;
}

static void _vincular(int t1, int t2, int* neigh, int* n_neigh) {
    int found=0;
    for (int k=0;k<n_neigh[t1];k++) if (neigh[t1*7+k]==t2) { found=1; break; }
    if (!found && n_neigh[t1]<7) neigh[t1*7+n_neigh[t1]++]=t2;
}

void _construir_vecindad(_BufBald* tejas, int T, int* neigh, int* n_neigh) {
    _Edge* edges=(_Edge*)malloc(sizeof(_Edge)*T*4);
    int ecount=0;
    for (int i=0;i<T;i++) {
        int* vs=tejas->buf[i].v;
        for (int p=0;p<4;p++) {
            int u=vs[p], w=vs[(p+1)%4];
            int a=u<w?u:w, b=u<w?w:u;
            edges[ecount++]=(_Edge){a,b,i};
        }
    }
    qsort(edges, ecount, sizeof(_Edge), _cmp_edge);

    for (int i=0;i<ecount;) {
        int j=i+1;
        while (j<ecount && edges[j].a==edges[i].a && edges[j].b==edges[i].b) j++;
        for (int p=i;p<j;p++) for (int q=p+1;q<j;q++) {
            _vincular(edges[p].tile, edges[q].tile, neigh, n_neigh);
            _vincular(edges[q].tile, edges[p].tile, neigh, n_neigh);
        }
        i=j;
    }
    free(edges);
}
