// vc2_xdma_model: the software stand-in for the VC-2 decoder card behind the vc2_xdma API (vc2x_open("model:...")).
// SPDX-License-Identifier: MIT
//
// It takes the H2C stream of format v0 (docs/FPGA.md section 9), rebuilds each access unit with the card model
// (vc2_fpga_model_au), decodes it with the CPU decoder straight into interleaved NV20 (Y line, CbCr line, ...), and
// hands the bytes out through vc2x_recv, as the card's C2H window does. It behaves like the card where that is
// visible to the host:
//   - back-pressure: vc2x_send blocks while QUEUE pictures wait to be read, vc2x_recv blocks until bytes exist;
//   - the status block: picture count and luma lines of the picture being output, pictures completed, the flags;
//   - VC2X_ERR_MAGIC (no descriptor where one must start), VC2X_ERR_FORMAT (a picture outside the subset) and
//     VC2X_ERR_OVERRUN (the slices run past the descriptor's slice bytes): the card stops at that descriptor and
//     puts nothing out for it; the pictures before it come out complete; it takes no H2C data until vc2x_reset
//     (a send blocks until the xdma driver's 10 s timeout). The model fails such sends at once with EIO.
//   - VC2X_ERR_GEOM: more than 8 level-4 rows per slice row (the card's band buffers). The card puts out a
//     full-length picture with undefined content, the model a correct one; the flag stays until reset.
//   VC2X_SATURATED is never set: the CPU decoder computes at full width.
// Arguments after "model:", comma separated: threads=N (decoder threads, default 2), cpus=a/b/c (pin them),
// queue=N (pictures held before vc2x_send blocks, default 2).
#define _GNU_SOURCE
#define VC2X_HAVE_MODEL
#include "vc2_xdma.h"
#include "vc2_fpga.h"
#include "vc2_priv.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct pic { struct pic *next; uint8_t *data; size_t len, pos, pair; uint32_t count; } pic;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    vc2_dec *dec;
    uint8_t *in, *au;      // H2C bytes not yet used; the rebuilt access unit
    size_t in_len, in_cap, au_cap;
    pic *head, *tail;
    int queued, queue, lost;
    uint32_t flags, pictures_done, last_count, gen;
} model;

static int geom_ok(const vc2_fpga_desc *d)
{
    int ph = (d->height + 15) & ~15; // padded to 2^depth
    return (ph / 2) / d->slices_y <= 8; // level-4 band rows per slice row
}

static const char *subset(const vc2_fpga_desc *d)
{
    vc2_seq s = { .major = 2, .width = d->width, .height = d->height, .chroma = d->chroma, .bit_depth = d->bit_depth };
    vc2_pic p = { .wavelet = d->wavelet, .depth = d->depth, .slices_x = d->slices_x, .slices_y = d->slices_y,
                  .prefix_bytes = d->prefix_bytes, .size_scaler = d->size_scaler };
    if (d->slices_x < 1 || d->slices_y < 1 || d->depth < 1 || d->depth > 5) return "bad geometry";
    return vc2_fpga_subset(&s, &p);
}

// one access unit from the head of m->in, called with the lock held; returns the bytes used, 0 = need more
static size_t one(model *m, uint32_t gen)
{
    vc2_fpga_desc d;
    if (m->in_len < VC2_FPGA_DESC_LEN) return 0;
    if (vc2_fpga_desc_read(m->in, m->in_len, &d)) { m->lost = 1; m->flags |= VC2X_ERR_MAGIC; return m->in_len; }
    size_t total = (VC2_FPGA_DESC_LEN + (size_t)d.slice_bytes + VC2_FPGA_ALIGN - 1) & ~(size_t)(VC2_FPGA_ALIGN - 1);
    if (m->in_len < total) return 0;
    if (subset(&d)) { m->lost = 1; m->flags |= VC2X_ERR_FORMAT; return m->in_len; }
    if (!geom_ok(&d)) m->flags |= VC2X_ERR_GEOM;
    if (m->au_cap < d.slice_bytes + 256) {
        free(m->au);
        m->au_cap = d.slice_bytes + 256;
        if (!(m->au = calloc(1, m->au_cap + VC2_INPUT_PADDING))) { m->au_cap = 0; return 0; }
    }
    size_t n = vc2_fpga_model_au(m->in, total, m->au, m->au_cap);
    size_t line = vc2x_line_bytes(d.width), size = vc2x_picture_bytes(d.width, d.height);
    pic *p = calloc(1, sizeof *p);
    if (p) p->data = malloc(size);
    if (!n || !p || !p->data) { if (p) free(p->data); free(p); m->lost = 1; m->flags |= VC2X_ERR_OVERRUN; return m->in_len; }
    // decode without the lock, so that vc2x_recv and vc2x_status go on meanwhile
    pthread_mutex_unlock(&m->mu);
    vc2_dec_set_nv20(m->dec, p->data, p->data + line, 2 * line);
    vc2_frame f;
    int bad = vc2_dec_decode(m->dec, m->au, n, &f);
    pthread_mutex_lock(&m->mu);
    if (bad || gen != m->gen) { // decoder error, or a reset meanwhile
        free(p->data); free(p);
        if (gen != m->gen) return 0;
        m->lost = 1; m->flags |= VC2X_ERR_OVERRUN;
        return m->in_len;
    }
    p->len = size; p->pair = 2 * line; p->count = d.au_index;
    if (m->tail) m->tail->next = p; else m->head = p;
    m->tail = p;
    m->queued++;
    pthread_cond_broadcast(&m->cv);
    return total;
}

static int m_send(vc2x *x, const void *buf, size_t len)
{
    model *m = x->priv;
    if (len % 16) { errno = EINVAL; return -1; }
    pthread_mutex_lock(&m->mu);
    uint32_t gen = m->gen;
    if (m->lost) { pthread_mutex_unlock(&m->mu); errno = EIO; return -1; } // the card takes nothing until a reset
    if (m->in_len + len > m->in_cap) {
        size_t cap = (m->in_len + len) * 2;
        uint8_t *n = realloc(m->in, cap);
        if (!n) { pthread_mutex_unlock(&m->mu); errno = ENOMEM; return -1; }
        m->in = n; m->in_cap = cap;
    }
    memcpy(m->in + m->in_len, buf, len);
    m->in_len += len;
    for (;;) {
        while (m->queued >= m->queue && gen == m->gen) pthread_cond_wait(&m->cv, &m->mu); // back-pressure
        if (gen != m->gen) break;
        size_t used = one(m, gen);
        if (!used || gen != m->gen) break;
        memmove(m->in, m->in + used, m->in_len - used);
        m->in_len -= used;
        if (m->lost) { m->in_len = 0; break; }
    }
    pthread_mutex_unlock(&m->mu);
    return 0;
}

static int m_recv(vc2x *x, void *buf, size_t len)
{
    model *m = x->priv;
    uint8_t *o = buf;
    pthread_mutex_lock(&m->mu);
    uint32_t gen = m->gen;
    while (len) {
        while (!m->head && gen == m->gen) pthread_cond_wait(&m->cv, &m->mu);
        if (gen != m->gen) { pthread_mutex_unlock(&m->mu); errno = ECANCELED; return -1; }
        pic *p = m->head;
        if (len % p->pair) { pthread_mutex_unlock(&m->mu); errno = EINVAL; return -1; }
        size_t n = p->len - p->pos < len ? p->len - p->pos : len;
        memcpy(o, p->data + p->pos, n);
        o += n; len -= n; p->pos += n;
        m->last_count = p->count;
        if (p->pos == p->len) {
            m->head = p->next;
            if (!m->head) m->tail = 0;
            m->queued--; m->pictures_done++;
            free(p->data); free(p);
            pthread_cond_broadcast(&m->cv);
        }
    }
    pthread_mutex_unlock(&m->mu);
    return 0;
}

static int m_status(vc2x *x, struct vc2x_status *s)
{
    model *m = x->priv;
    pthread_mutex_lock(&m->mu);
    memset(s, 0, sizeof *s);
    s->id = VC2X_ID;
    s->version = 0x10000; // the model, version 1.0
    s->flags = m->flags | (!m->head && !m->in_len ? VC2X_IDLE : 0);
    s->picture_count = m->head ? m->head->count : m->last_count;
    s->lines_done = m->head ? (uint32_t)(m->head->pos / m->head->pair) : 0;
    s->pictures_done = m->pictures_done;
    pthread_mutex_unlock(&m->mu);
    return 0;
}

static int m_check(vc2x *x) { (void)x; return 0; }

static int m_reset(vc2x *x)
{
    model *m = x->priv;
    pthread_mutex_lock(&m->mu);
    m->gen++; // blocked send / recv calls return
    while (m->head) { pic *p = m->head; m->head = p->next; free(p->data); free(p); }
    m->tail = 0; m->queued = 0; m->in_len = 0; m->lost = 0; m->flags = 0;
    pthread_cond_broadcast(&m->cv);
    pthread_mutex_unlock(&m->mu);
    return 0;
}

static void m_close(vc2x *x)
{
    model *m = x->priv;
    m_reset(x);
    vc2_dec_free(m->dec);
    free(m->in); free(m->au);
    pthread_mutex_destroy(&m->mu); pthread_cond_destroy(&m->cv);
    free(m); free(x);
}

static const struct vc2x_ops model_ops = { m_check, m_reset, m_send, m_recv, m_status, m_close };

vc2x *vc2x_model_open(const char *arg)
{
    int threads = 2, queue = 2, cpus[64], ncpu = 0;
    for (const char *a = arg; a && *a;) {
        if (!strncmp(a, "threads=", 8)) threads = atoi(a + 8);
        else if (!strncmp(a, "queue=", 6)) queue = atoi(a + 6);
        else if (!strncmp(a, "cpus=", 5)) {
            for (const char *c = a + 5; *c && *c != ',' && ncpu < 64;) {
                cpus[ncpu++] = atoi(c);
                while (*c && *c != '/' && *c != ',') c++;
                if (*c == '/') c++;
            }
        } else { errno = EINVAL; return NULL; }
        a = strchr(a, ',');
        if (a) a++;
    }
    if (threads < 1 || threads > 64 || queue < 1 || (ncpu && ncpu < threads)) { errno = EINVAL; return NULL; }
    vc2x *x = calloc(1, sizeof *x);
    model *m = calloc(1, sizeof *m);
    if (!x || !m || !(m->dec = vc2_dec_new(threads, ncpu ? cpus : 0))) { free(x); free(m); errno = ENOMEM; return NULL; }
    pthread_mutex_init(&m->mu, 0);
    pthread_cond_init(&m->cv, 0);
    m->queue = queue;
    x->ops = &model_ops;
    x->priv = m;
    return x;
}
