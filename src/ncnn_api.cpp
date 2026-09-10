// Thin C shim over ncnn::Net for dart:ffi — generalized inference API.
//
// - Every extern "C" entry is wrapped in try/catch: ncnn throws C++
//   exceptions (corrupted param/bin files), and an exception crossing
//   the C boundary into Dart FFI is UB — surface as HN_ERR_EXCEPTION.
// - Output blobs are discovered by parsing the param file (blobs
//   produced but never consumed; Noop-consumed as the onnx2ncnn
//   fallback), so callers never hardcode blob names.
// - A 64x64 warm-up records output shapes as hints; shapes may vary
//   with input size, so hn_extract reports the CURRENT run's shapes and
//   supports a capacity-retry via *required_out.
#include "ncnn_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "ncnn/net.h"
#if NCNN_VULKAN
#include "ncnn/gpu.h"
#include <mutex>
#if defined(_WIN32)
#include <cstdlib>
#endif
#endif

namespace {

constexpr int kWarmupSize = 64;

struct hn_net {
    ncnn::Net net;
    bool loaded = false;
    int pixel_format = HN_PIX_RGB;
    float mean[3] = {0.f, 0.f, 0.f};
    float norm[3] = {1.f, 1.f, 1.f};
    std::string input_blob = "in0"; // strdup'd copy — owns the lifetime
    std::vector<int> output_blobs;
    // 5 ints per output: [dims, w, h, d, c]; dims == 0 = no hint.
    std::vector<int> output_shape_hints;
};

hn_net* as_net(hn_net_t net) {
    return static_cast<hn_net*>(net);
}

#if NCNN_VULKAN && defined(_WIN32)

// EVIDENCE (2026-09, RTX 4060 + Intel Arc laptop, ncnn 20260526):
//   hardware Vulkan + Flutter engine process  -> crash AFTER full device
//     enumeration (all device dumps print, incl. NVIDIA), inside the tail
//     of create_gpu_instance / early device use
//   hardware Vulkan + plain native / python   -> works
//   SwiftShader ICD + Flutter engine process  -> works
//   CPU                                        -> always works
// Policy: in-process on Windows the default is ZERO Vulkan devices (CPU
// inference) so apps never touch the crashing path. GPU inference goes
// through the standalone ncnn_helper child process instead. Opt back in
// with NCNN_DART_ENABLE_VK=1 (or pin VK_ICD_FILENAMES yourself).
int windows_vulkan_allowed() {
    if (getenv("NCNN_DART_ENABLE_VK") != nullptr) return 1;
    if (getenv("VK_ICD_FILENAMES") != nullptr) return 1;
    return 0;
}

std::once_flag gpu_init_once;

void warm_up_gpu_instance() {
    std::call_once(gpu_init_once, []() {
        if (!windows_vulkan_allowed()) return;
        // Synchronous init: a detached thread races with the caller's
        // get_gpu_count and can tear down mid-read (observed crash).
        ncnn::create_gpu_instance();
    });
}

// NOTE: non-Windows platforms previously had a detached-thread + 200ms
// sleep warm-up here. Removed: ncnn creates the gpu instance lazily,
// idempotently and under its own lock (get_gpu_count), so the sleep was
// both unnecessary and racy.
#endif

int pixel_type(int fmt) {
    switch (fmt) {
        case HN_PIX_BGR: return ncnn::Mat::PIXEL_BGR;
        case HN_PIX_RGBA: return ncnn::Mat::PIXEL_RGBA;
        case HN_PIX_BGRA: return ncnn::Mat::PIXEL_BGRA;
        case HN_PIX_GRAY: return ncnn::Mat::PIXEL_GRAY;
        case HN_PIX_RGB:
        default: return ncnn::Mat::PIXEL_RGB;
    }
}

int channel_count(int fmt) {
    return fmt == HN_PIX_GRAY ? 1 : 3;
}

void set_options(ncnn::Net& net, int use_vulkan, int device_index) {
    net.opt.num_threads = 4;
#if NCNN_VULKAN
#if defined(_WIN32)
    if (use_vulkan && !windows_vulkan_allowed()) use_vulkan = 0;
    // warm_up_gpu_instance only exists on Windows (see NOTE above);
    // elsewhere ncnn inits the gpu instance lazily under its own lock.
    if (use_vulkan) warm_up_gpu_instance();
#endif
    net.opt.use_vulkan_compute = use_vulkan != 0;
    if (use_vulkan && device_index >= 0) net.set_vulkan_device(device_index);
#else
    (void)use_vulkan;
    (void)device_index;
#endif
}

// Parse a param file for graph output blobs: ids produced but never
// consumed by any layer. Empty result falls back to blobs consumed by
// Noop layers (onnx2ncnn marks outputs that way). Ascending blob-id
// order for deterministic output indexing.
bool parse_output_blobs(const char* path, std::vector<int>* out) {
    FILE* f = fopen(path, "rb");
    if (f == nullptr) return false;
    char line[1024];
    if (fgets(line, sizeof(line), f) == nullptr) { fclose(f); return false; }
    int layer_count = 0, blob_count = 0;
    if (fscanf(f, "%d %d", &layer_count, &blob_count) != 2) {
        fclose(f);
        return false;
    }
    fgets(line, sizeof(line), f); // consume EOL after the counts
    std::unordered_set<int> produced, consumed, noop_consumed;
    bool ok = true;
    for (int i = 0; i < layer_count && ok; i++) {
        char type[256], name[256];
        int in_n = 0, out_n = 0;
        if (fscanf(f, "%255s %255s %d %d", type, name, &in_n, &out_n) != 4) {
            ok = false;
            break;
        }
        const bool is_noop = strcmp(type, "Noop") == 0;
        for (int k = 0; k < in_n; k++) {
            int b;
            if (fscanf(f, "%d", &b) != 1) { ok = false; break; }
            consumed.insert(b);
            if (is_noop) noop_consumed.insert(b);
        }
        for (int k = 0; ok && k < out_n; k++) {
            int b;
            if (fscanf(f, "%d", &b) != 1) { ok = false; break; }
            produced.insert(b);
        }
    }
    fclose(f);
    if (!ok) return false;
    for (int b : produced) {
        if (consumed.find(b) == consumed.end()) out->push_back(b);
    }
    if (out->empty()) {
        for (int b : noop_consumed) out->push_back(b);
    }
    std::sort(out->begin(), out->end());
    return !out->empty();
}

// Total element count of an ncnn Mat (logical elements, channels included).
size_t mat_elems(const ncnn::Mat& m) {
    if (m.dims == 1) return (size_t)m.w;
    if (m.dims == 2) return (size_t)m.w * m.h;
    return (size_t)m.c * m.w * m.h * (m.dims == 4 ? m.d : 1);
}

// Fill shape[0..3] = (w, h, d|1, c), padded with 1s. Returns dims.
int fill_shape(const ncnn::Mat& m, int32_t* shape) {
    shape[0] = m.dims >= 1 ? m.w : 1;
    shape[1] = m.dims >= 2 ? m.h : 1;
    shape[2] = m.dims == 4 ? m.d : 1;
    shape[3] = m.dims >= 3 ? m.c : 1;
    return m.dims;
}

// Copy one output Mat into dst. Multi-channel Mats are cstep-strided
// per channel (NOT contiguous) — a flat memcpy across channels
// corrupts data.
void copy_mat(const ncnn::Mat& m, float* dst) {
    if (m.dims <= 2) {
        memcpy(dst, m.data, mat_elems(m) * sizeof(float));
        return;
    }
    const size_t plane = mat_elems(m) / (size_t)(m.c > 0 ? m.c : 1);
    for (int c = 0; c < m.c; c++) {
        memcpy(dst + (size_t)c * plane, m.channel(c).data,
               plane * sizeof(float));
    }
}

// Shared extract path over all discovered outputs.
int extract_all(hn_net* n, const ncnn::Mat& in, float* out, int out_cap,
                int32_t* shapes, int shape_cap, int32_t* required_out) {
    ncnn::Extractor ex = n->net.create_extractor();
    if (ex.input(n->input_blob.c_str(), in) != 0) return HN_ERR_NO_INPUT_BLOB;
    std::vector<ncnn::Mat> outs(n->output_blobs.size());
    size_t total = 0;
    for (size_t i = 0; i < n->output_blobs.size(); i++) {
        if (ex.extract(n->output_blobs[i], outs[i]) != 0) return HN_ERR_EXTRACT;
        const size_t elems = mat_elems(outs[i]);
        if (elems == 0) return HN_ERR_SHAPE;
        total += elems;
        if (shapes != nullptr && (int)(i + 1) * 4 <= shape_cap) {
            fill_shape(outs[i], shapes + i * 4);
        }
    }
    if ((size_t)out_cap < total) {
        if (required_out != nullptr) *required_out = (int32_t)total;
        return HN_ERR_CAPACITY;
    }
    size_t off = 0;
    for (const auto& m : outs) {
        copy_mat(m, out + off);
        off += mat_elems(m);
    }
    return (int)total;
}

} // namespace

hn_net_t hn_create(const hn_options_t* opts) {
    if (opts != nullptr &&
        (opts->struct_size < HN_OPTIONS_V1_SIZE ||
         opts->struct_size > (uint32_t)sizeof(hn_options_t))) {
        return nullptr;
    }
    auto* n = new (std::nothrow) hn_net();
    if (n == nullptr) return nullptr;
    try {
        if (opts != nullptr) {
            n->pixel_format = opts->pixel_format;
            for (int i = 0; i < 3; i++) {
                n->mean[i] = opts->mean[i];
                n->norm[i] = opts->norm[i];
            }
            if (opts->input_blob != nullptr) n->input_blob = opts->input_blob;
        }
        set_options(n->net, opts ? opts->use_vulkan : 0,
                    opts ? opts->device_index : -1);
    } catch (...) {
        delete n;
        return nullptr;
    }
    return n;
}

int hn_load(hn_net_t net, const char* param_path, const char* bin_path) {
    if (net == nullptr || param_path == nullptr || bin_path == nullptr) {
        return HN_ERR_INVALID;
    }
    auto* n = as_net(net);
    try {
        if (!parse_output_blobs(param_path, &n->output_blobs)) {
            return HN_ERR_PARAM;
        }
        if (n->net.load_param(param_path) != 0) return HN_ERR_PARAM;
        if (n->net.load_model(bin_path) != 0) return HN_ERR_BIN;
        n->loaded = true;
        // Warm-up for shape hints: 64x64 zeros through the real input
        // path. Failure is non-fatal — hints stay zero and callers fall
        // back to the capacity-retry on the first real extract.
        const int ch = channel_count(n->pixel_format);
        std::vector<uint8_t> zeros(
            (size_t)kWarmupSize * kWarmupSize * ch, 0);
        ncnn::Mat in = ncnn::Mat::from_pixels(
            zeros.data(), pixel_type(n->pixel_format),
            kWarmupSize, kWarmupSize);
        // substract_mean_normalize reads norm[channels] entries — one per
        // channel, NOT a single shared scalar (a 1-element array reads OOB
        // and corrupts G/B planes).
        in.substract_mean_normalize(n->mean, n->norm);
        n->output_shape_hints.assign(n->output_blobs.size() * 5, 0);
        ncnn::Extractor ex = n->net.create_extractor();
        if (ex.input(n->input_blob.c_str(), in) == 0) {
            for (size_t i = 0; i < n->output_blobs.size(); i++) {
                ncnn::Mat om;
                if (ex.extract(n->output_blobs[i], om) == 0 &&
                    mat_elems(om) > 0) {
                    n->output_shape_hints[i * 5] = om.dims;
                    fill_shape(om, &n->output_shape_hints[i * 5 + 1]);
                }
            }
        }
        return HN_OK;
    } catch (...) {
        n->loaded = false;
        return HN_ERR_EXCEPTION;
    }
}

int hn_output_count(hn_net_t net) {
    if (net == nullptr) return 0;
    return (int)as_net(net)->output_blobs.size();
}

int hn_output_shape(hn_net_t net, int out_index, int32_t shape[4]) {
    if (net == nullptr || shape == nullptr) return 0;
    auto* n = as_net(net);
    if (out_index < 0 || (size_t)out_index >= n->output_blobs.size()) {
        return 0;
    }
    const int32_t* hint = &n->output_shape_hints[out_index * 5];
    if (hint[0] == 0) return 0; // no warm-up hint
    memcpy(shape, hint + 1, 4 * sizeof(int32_t));
    return hint[0];
}

int hn_extract(hn_net_t net, const uint8_t* pixels, int w, int h,
               float* out, int out_cap, int32_t* shapes, int shape_cap,
               int32_t* required_out) {
    if (net == nullptr || pixels == nullptr || w <= 0 || h <= 0) {
        return HN_ERR_INVALID;
    }
    auto* n = as_net(net);
    if (!n->loaded) return HN_ERR_INVALID;
    try {
        ncnn::Mat in = ncnn::Mat::from_pixels(
            pixels, pixel_type(n->pixel_format), w, h);
        in.substract_mean_normalize(n->mean, n->norm);
        return extract_all(n, in, out, out_cap, shapes, shape_cap,
                           required_out);
    } catch (...) {
        return HN_ERR_EXCEPTION;
    }
}

int hn_extract_f32(hn_net_t net, const float* data, const int32_t* shape,
                   int dims, float* out, int out_cap, int32_t* shapes,
                   int shape_cap, int32_t* required_out) {
    if (net == nullptr || data == nullptr || shape == nullptr ||
        dims < 1 || dims > 4) {
        return HN_ERR_INVALID;
    }
    auto* n = as_net(net);
    if (!n->loaded) return HN_ERR_INVALID;
    try {
        // Wraps the caller's buffer (no copy); safe because `data`
        // outlives this call. The ncnn 20260526 release headers have no
        // Mat(dims, sizes, data, elemsize) constructor, so dispatch on
        // the 1/2/3/4-D external-data overloads (ncnn order w/h[/d]/c).
        ncnn::Mat in;
        switch (dims) {
            case 1:
                in = ncnn::Mat(shape[0], (void*)data, (size_t)4);
                break;
            case 2:
                in = ncnn::Mat(shape[0], shape[1], (void*)data, (size_t)4);
                break;
            case 3:
                in = ncnn::Mat(shape[0], shape[1], shape[2], (void*)data,
                               (size_t)4);
                break;
            default:
                in = ncnn::Mat(shape[0], shape[1], shape[2], shape[3],
                               (void*)data, (size_t)4);
                break;
        }
        return extract_all(n, in, out, out_cap, shapes, shape_cap,
                           required_out);
    } catch (...) {
        return HN_ERR_EXCEPTION;
    }
}

void hn_destroy(hn_net_t net) {
    delete static_cast<hn_net*>(net);
}

int hn_gpu_count(void) {
#if NCNN_VULKAN
#if defined(_WIN32)
    if (!windows_vulkan_allowed()) return 0;
#endif
    try {
        return ncnn::get_gpu_count();
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}

int hn_gpu_devices(hn_gpu_device_t* devices, char (*names)[HN_NAME_MAX],
                   int n) {
#if NCNN_VULKAN
#if defined(_WIN32)
    if (!windows_vulkan_allowed()) return 0;
#endif
    try {
        const int count = ncnn::get_gpu_count();
        if (count <= 0 || devices == nullptr || names == nullptr || n <= 0) {
            return 0;
        }
        int written = 0;
        for (int i = 0; i < count && written < n; i++) {
            const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
            devices[written].index = i;
            devices[written].type = info.type();
            devices[written].score = (int)info.rough_score();
            devices[written].vendor_id = info.vendor_id();
            const char* name = info.device_name();
            if (name != nullptr) {
                strncpy(names[written], name, HN_NAME_MAX - 1);
                names[written][HN_NAME_MAX - 1] = '\0';
            } else {
                names[written][0] = '\0';
            }
            written++;
        }
        return written;
    } catch (...) {
        return 0;
    }
#else
    (void)devices;
    (void)names;
    (void)n;
    return 0;
#endif
}
