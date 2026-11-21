// vc2_xdma: the hardware backend over the Xilinx xdma driver's character devices (see vc2_xdma.h).
// SPDX-License-Identifier: MIT
#define _GNU_SOURCE
#include "vc2_xdma.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// status block registers (byte offsets from VC2X_STATUS_BASE)
enum { R_ID = 0x00, R_VERSION = 0x04, R_FLAGS = 0x08, R_PICCOUNT = 0x0C, R_LINES = 0x10, R_PICS = 0x14, R_CTRL = 0x18 };

typedef struct {
    int h2c, c2h, user;
} hw;

static int rd32(int fd, uint32_t off, uint32_t *v)
{
    ssize_t n = pread(fd, v, 4, VC2X_STATUS_BASE + off);
    if (n == 4) return 0;
    if (n >= 0) errno = EIO;
    return -1;
}

static int wr32(int fd, uint32_t off, uint32_t v)
{
    ssize_t n = pwrite(fd, &v, 4, VC2X_STATUS_BASE + off);
    if (n == 4) return 0;
    if (n >= 0) errno = EIO;
    return -1;
}

// a whole transfer, starting at the window base every time (the window is a byte stream)
static int xfer(int fd, void *buf, size_t len, uint64_t base, int write)
{
    uint8_t *p = buf;
    while (len) {
        ssize_t n = write ? pwrite(fd, p, len, (off_t)base) : pread(fd, p, len, (off_t)base);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) { errno = EIO; return -1; }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int hw_status(vc2x *x, struct vc2x_status *s)
{
    hw *h = x->priv;
    if (rd32(h->user, R_ID, &s->id) || rd32(h->user, R_VERSION, &s->version) || rd32(h->user, R_FLAGS, &s->flags)
        || rd32(h->user, R_PICCOUNT, &s->picture_count) || rd32(h->user, R_LINES, &s->lines_done)
        || rd32(h->user, R_PICS, &s->pictures_done))
        return -1;
    return 0;
}

static int hw_check(vc2x *x)
{
    hw *h = x->priv;
    uint32_t id;
    if (rd32(h->user, R_ID, &id)) return -1;
    if (id != VC2X_ID) { errno = ENODEV; return -1; }
    return 0;
}

static int hw_reset(vc2x *x)
{
    hw *h = x->priv;
    if (hw_check(x) || wr32(h->user, R_CTRL, 1)) return -1;
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        uint32_t f;
        if (rd32(h->user, R_FLAGS, &f)) return -1;
        if ((f & VC2X_IDLE) && !(f & VC2X_ERRORS)) return 0;
        clock_gettime(CLOCK_MONOTONIC, &t);
        if ((t.tv_sec - t0.tv_sec) * 1000000000L + (t.tv_nsec - t0.tv_nsec) > 1000000000L) { errno = ETIMEDOUT; return -1; }
        usleep(1000);
    }
}

static int hw_send(vc2x *x, const void *buf, size_t len)
{
    if (len % 16) { errno = EINVAL; return -1; }
    return xfer(((hw *)x->priv)->h2c, (void *)buf, len, VC2X_H2C_BASE, 1);
}

static int hw_recv(vc2x *x, void *buf, size_t len)
{
    return xfer(((hw *)x->priv)->c2h, buf, len, VC2X_C2H_BASE, 0);
}

static void hw_close(vc2x *x)
{
    hw *h = x->priv;
    if (h->h2c >= 0) close(h->h2c);
    if (h->c2h >= 0) close(h->c2h);
    if (h->user >= 0) close(h->user);
    free(h);
    free(x);
}

static const struct vc2x_ops hw_ops = { hw_check, hw_reset, hw_send, hw_recv, hw_status, hw_close };

// Without a model backend: NULL / ENOSYS. Builds that link the model (vc2-endpoint: lib/vc2_xdma_model.c) define
// VC2X_HAVE_MODEL, because in a static library a weak default would win over an archive member that defines it.
#ifndef VC2X_HAVE_MODEL
__attribute__((weak)) vc2x *vc2x_model_open(const char *arg)
{
    (void)arg;
    errno = ENOSYS;
    return NULL;
}
#endif

vc2x *vc2x_open(const char *dev)
{
    if (!strncmp(dev, "model:", 6)) return vc2x_model_open(dev + 6);
    vc2x *x = calloc(1, sizeof *x);
    hw *h = calloc(1, sizeof *h);
    if (!x || !h) { free(x); free(h); errno = ENOMEM; return NULL; }
    h->h2c = h->c2h = h->user = -1;
    x->ops = &hw_ops;
    x->priv = h;
    char path[256];
    snprintf(path, sizeof path, "%s_h2c_0", dev);
    h->h2c = open(path, O_WRONLY);
    snprintf(path, sizeof path, "%s_c2h_0", dev);
    h->c2h = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%s_user", dev);
    h->user = open(path, O_RDWR);
    if (h->h2c < 0 || h->c2h < 0 || h->user < 0) {
        int e = errno;
        hw_close(x);
        errno = e;
        return NULL;
    }
    return x;
}

void vc2x_close(vc2x *x) { if (x) x->ops->close(x); }
int vc2x_check(vc2x *x) { return x->ops->check(x); }
int vc2x_reset(vc2x *x) { return x->ops->reset(x); }
int vc2x_send(vc2x *x, const void *buf, size_t len) { return x->ops->send(x, buf, len); }
int vc2x_recv(vc2x *x, void *buf, size_t len) { return x->ops->recv(x, buf, len); }
int vc2x_status(vc2x *x, struct vc2x_status *s) { return x->ops->status(x, s); }
