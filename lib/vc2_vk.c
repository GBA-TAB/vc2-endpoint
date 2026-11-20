// Vulkan compute backend of the GPU path: the shader of vc2_idwt.comp on host-visible storage buffers.
// Plain Vulkan 1.0, nothing vendor-specific: for the Mesa drivers (v3dv, panvk, Intel, AMD) and lavapipe.
// Planar output only (the decoder packs NV20 itself with this backend), no dma-buf targets, no GPU timing.
// VC2_VK_DEVICE=<part of the device name> picks a device (default: the first one with a compute queue).
#define _GNU_SOURCE
#include "vc2_gpu.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

// the shader, compiled to SPIR-V by the Makefile
__asm__(".pushsection .rodata\n.balign 4\n.global vc2_idwt_spv\nvc2_idwt_spv:\n.incbin \"" VC2_IDWT_SPV "\"\n.global vc2_idwt_spv_end\nvc2_idwt_spv_end:\n.popsection\n");
extern const uint32_t vc2_idwt_spv[];
extern const char vc2_idwt_spv_end[];

#define TW 4  // as in the shader
#define RB 16
#define LSX 16 // its local_size_x
#define MAX_CB 256

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize size; int coherent; } vbuf;
typedef struct { int32_t off[3][4], bw[4], oofs[4], ostride[4], bh, oh, blk0; } push_t;

struct vc2_gpu {
    vc2_gpu_cfg cfg;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkShaderModule shader;
    VkPipeline pipe;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    VkCommandPool cpool;
    VkCommandBuffer cb[MAX_CB];
    int ncb, failed;
    vbuf coef, pix;
    VkDeviceSize atom;
    push_t pc;
    pthread_mutex_t mu; // launches come from several threads
};

static int make_buf(vc2_gpu *g, vbuf *b, VkDeviceSize size)
{
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(g->dev, &bi, 0, &b->buf)) return -1;
    VkMemoryRequirements mr;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetBufferMemoryRequirements(g->dev, b->buf, &mr);
    vkGetPhysicalDeviceMemoryProperties(g->phys, &mp);
    int best = -1, score = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (!(mr.memoryTypeBits & (1u << i)) || !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        int s = !!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) * 2 + !!(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (s > score) { score = s; best = (int)i; }
    }
    if (best < 0) return -1;
    b->coherent = score >= 2;
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = (uint32_t)best };
    if (vkAllocateMemory(g->dev, &ai, 0, &b->mem) || vkBindBufferMemory(g->dev, b->buf, b->mem, 0) ||
        vkMapMemory(g->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map)) return -1;
    b->size = size;
    memset(b->map, 0, size);
    return 0;
}

static void vk_destroy(vc2_gpu *g)
{
    if (!g) return;
    if (g->dev) {
        vkDeviceWaitIdle(g->dev);
        vbuf *bs[2] = { &g->coef, &g->pix };
        for (int i = 0; i < 2; i++) {
            if (bs[i]->buf) vkDestroyBuffer(g->dev, bs[i]->buf, 0);
            if (bs[i]->mem) vkFreeMemory(g->dev, bs[i]->mem, 0);
        }
        if (g->cpool) vkDestroyCommandPool(g->dev, g->cpool, 0);
        if (g->dpool) vkDestroyDescriptorPool(g->dev, g->dpool, 0);
        if (g->pipe) vkDestroyPipeline(g->dev, g->pipe, 0);
        if (g->shader) vkDestroyShaderModule(g->dev, g->shader, 0);
        if (g->layout) vkDestroyPipelineLayout(g->dev, g->layout, 0);
        if (g->dsl) vkDestroyDescriptorSetLayout(g->dev, g->dsl, 0);
        vkDestroyDevice(g->dev, 0);
    }
    if (g->inst) vkDestroyInstance(g->inst, 0);
    pthread_mutex_destroy(&g->mu);
    free(g);
}

static vc2_gpu *vk_create(const vc2_gpu_cfg *cfg, vc2_gpu_bufs *bufs)
{
    vc2_gpu *g = calloc(1, sizeof *g);
    g->cfg = *cfg;
    pthread_mutex_init(&g->mu, 0);
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vc2", .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    if (vkCreateInstance(&ici, 0, &g->inst)) { g->inst = 0; goto fail; }
    VkPhysicalDevice pd[16];
    uint32_t npd = 16;
    vkEnumeratePhysicalDevices(g->inst, &npd, pd);
    const char *want = getenv("VC2_VK_DEVICE");
    for (uint32_t i = 0; i < npd && !g->phys; i++) {
        VkPhysicalDeviceProperties pr;
        VkQueueFamilyProperties qf[16];
        uint32_t nq = 16;
        vkGetPhysicalDeviceProperties(pd[i], &pr);
        if (want && !strstr(pr.deviceName, want)) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qf);
        for (uint32_t q = 0; q < nq; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                g->phys = pd[i]; g->qfam = q; g->atom = pr.limits.nonCoherentAtomSize;
                if (getenv("VC2_GPU_DEBUG")) fprintf(stderr, "vc2: Vulkan device %s\n", pr.deviceName);
                break;
            }
    }
    if (!g->phys) { fprintf(stderr, "vc2: no Vulkan device with a compute queue%s\n", want ? " of that name" : ""); goto fail; }
    float prio = 1;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g->qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    if (vkCreateDevice(g->phys, &dci, 0, &g->dev)) { g->dev = 0; goto fail; }
    vkGetDeviceQueue(g->dev, g->qfam, 0, &g->queue);

    // buffers: the same layout as the OpenCL backend (room before and after every plane for the lane loads)
    size_t pos = 2048;
    for (int c = 0; c < 3; c++)
        for (int o = 0; o < 4; o++) { g->pc.off[c][o] = (int32_t)pos; pos += ((size_t)cfg->bw[c] * cfg->bh + 64 + 31) & ~(size_t)31; }
    if (make_buf(g, &g->coef, (pos * 2 + 8191) & ~(size_t)4095)) goto fail;
    pos = 0;
    for (int c = 0; c < 3; c++) {
        g->pc.bw[c] = cfg->bw[c];
        g->pc.ostride[c] = (2 * cfg->bw[c] + 15) & ~15;
        g->pc.oofs[c] = (int32_t)pos;
        pos += (size_t)g->pc.ostride[c] * 2 * cfg->bh + 32;
    }
    g->pc.bh = cfg->bh; g->pc.oh = cfg->h;
    if (make_buf(g, &g->pix, (pos * 2 + 8191) & ~(size_t)4095)) goto fail;
    for (int c = 0; c < 3; c++) {
        for (int o = 0; o < 4; o++) bufs->band[c][o] = (int16_t *)g->coef.map + g->pc.off[c][o];
        bufs->pix[c] = (uint16_t *)g->pix.map + g->pc.oofs[c];
        bufs->pstride[c] = g->pc.ostride[c];
    }
    bufs->rb = RB;
    bufs->nv20 = 0;

    VkDescriptorSetLayoutBinding bind[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
    VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bind };
    if (vkCreateDescriptorSetLayout(g->dev, &dli, 0, &g->dsl)) goto fail;
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_t) };
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g->dsl,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    if (vkCreatePipelineLayout(g->dev, &pli, 0, &g->layout)) goto fail;
    VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)(vc2_idwt_spv_end - (const char *)vc2_idwt_spv),
                                     .pCode = vc2_idwt_spv };
    if (vkCreateShaderModule(g->dev, &smi, 0, &g->shader)) goto fail;
    const int32_t spec[3] = { cfg->wavelet == VC2_WL_LEGALL5_3, 1 << (cfg->bit_depth - 1), (1 << cfg->bit_depth) - 1 };
    const VkSpecializationMapEntry sme[3] = { { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 } };
    const VkSpecializationInfo si = { 3, sme, sizeof spec, spec };
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = g->layout,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = g->shader,
                   .pName = "main", .pSpecializationInfo = &si } };
    if (vkCreateComputePipelines(g->dev, VK_NULL_HANDLE, 1, &cpi, 0, &g->pipe)) goto fail;
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorSetCount = 1, .pSetLayouts = &g->dsl };
    if (vkCreateDescriptorPool(g->dev, &dpi, 0, &g->dpool)) goto fail;
    dai.descriptorPool = g->dpool;
    if (vkAllocateDescriptorSets(g->dev, &dai, &g->dset)) goto fail;
    VkDescriptorBufferInfo dbi[2] = { { g->coef.buf, 0, VK_WHOLE_SIZE }, { g->pix.buf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wr[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g->dset, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g->dset, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[1] } };
    vkUpdateDescriptorSets(g->dev, 2, wr, 0, 0);
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = g->qfam };
    if (vkCreateCommandPool(g->dev, &cpci, 0, &g->cpool)) goto fail;
    return g;
fail:
    vk_destroy(g);
    return 0;
}

static int vk_begin(vc2_gpu *g, const vc2_gpu_target *t)
{
    g->failed = 0;
    return t->nv20 ? -1 : 0;
}

// host-visible memory that is not coherent wants to be told about writes (and reads, in finish)
static void vk_wrote(vc2_gpu *g, const void *p, size_t bytes)
{
    if (g->coef.coherent || !bytes) return;
    VkDeviceSize a = g->atom ? g->atom : 1, o = (VkDeviceSize)((const char *)p - (const char *)g->coef.map), e = o + bytes;
    o -= o % a; e = (e + a - 1) / a * a;
    if (e > g->coef.size) e = g->coef.size;
    VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = g->coef.mem, .offset = o, .size = e - o };
    vkFlushMappedMemoryRanges(g->dev, 1, &r);
}

static int vk_launch(vc2_gpu *g, int thread, int blk0, int blk1)
{
    (void)thread;
    const vc2_gpu_cfg *f = &g->cfg;
    const uint32_t tiles = (uint32_t)((f->bw[0] + TW - 1) / TW + (f->bw[1] + TW - 1) / TW + (f->bw[2] + TW - 1) / TW);
    int rc = -1;
    pthread_mutex_lock(&g->mu);
    if (g->ncb < MAX_CB) {
        VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g->cpool,
                                           .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        VkCommandBuffer cb;
        push_t pc = g->pc;
        pc.blk0 = blk0;
        if (!vkAllocateCommandBuffers(g->dev, &ai, &cb)) {
            g->cb[g->ncb++] = cb;
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
            if (!vkBeginCommandBuffer(cb, &bi)) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->dset, 0, 0);
                vkCmdPushConstants(cb, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
                vkCmdDispatch(cb, (tiles + LSX - 1) / LSX, (uint32_t)(blk1 - blk0), 1);
                if (!vkEndCommandBuffer(cb) && !vkQueueSubmit(g->queue, 1, &si, VK_NULL_HANDLE)) rc = 0;
            }
        }
    }
    if (rc) g->failed = 1;
    pthread_mutex_unlock(&g->mu);
    return rc;
}

static int vk_finish(vc2_gpu *g, double *gpu_seconds)
{
    pthread_mutex_lock(&g->mu);
    int rc = g->failed ? -1 : 0;
    if (vkQueueWaitIdle(g->queue)) rc = -1;
    if (g->ncb) vkFreeCommandBuffers(g->dev, g->cpool, (uint32_t)g->ncb, g->cb);
    g->ncb = 0;
    vkResetCommandPool(g->dev, g->cpool, 0);
    pthread_mutex_unlock(&g->mu);
    if (!g->pix.coherent) {
        VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = g->pix.mem, .offset = 0, .size = VK_WHOLE_SIZE };
        vkInvalidateMappedMemoryRanges(g->dev, 1, &r);
    }
    if (gpu_seconds) *gpu_seconds = 0; // not measured
    return rc;
}

const vc2_gpu_api vc2_vk_api = { "Vulkan", vk_create, vk_destroy, vk_begin, vk_wrote, vk_launch, vk_finish };
