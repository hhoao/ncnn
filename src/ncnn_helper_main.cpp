// Standalone ncnn inference helper for Windows.
//
// Rationale: ncnn's Vulkan device init crashes inside Flutter engine
// processes (NVIDIA nvoglv64 access violation — the engine's GL/D3D
// rendering stack conflicts with the driver's Vulkan path). In a plain
// process the same shim runs flawlessly on GPU. This helper is spawned
// by the app as a child process and speaks a binary protocol over
// stdin/stdout (all integers little-endian, full-buffer reads):
//
//   load:  u32 cmd=1 | u32 paramLen | param | u32 binLen | bin |
//          u32 optsLen | opts
//          opts: i32 useVulkan, i32 deviceIndex, i32 pixelFormat,
//                u32 blobLen | blob, f32 mean[3], f32 norm[3],
//                i32 warmup_w, i32 warmup_h   (fixed part = 48 bytes,
//                optsLen = 48 + blobLen; future field appends may add
//                trailing bytes — they are skipped)
//          -> i32 status | u32 outCount |
//             outCount × (u32 dims, u32 s0, u32 s1, u32 s2, u32 s3)
//   predict: u32 cmd=2 | u32 w | u32 h | u32 frameLen | pixels
//          -> i32 status | u32 total | total × f32 (all outputs concat)
//   extractF32: u32 cmd=4 | u32 dims | dims × u32 shape |
//          u32 dataLen | f32 data
//          -> i32 status | u32 total | total × f32
//   gpu:   u32 cmd=3
//          -> u32 status | u32 count | count × (u32 idx, u32 type,
//             u32 score, u32 vendor, u32 nameLen, name bytes)
//   quit:  u32 cmd=0 -> exits
#include "ncnn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#endif

static int read_exact(uint8_t* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        size_t n = fread(buf + got, 1, len - got, stdin);
        if (n == 0) return -1;
        got += n;
    }
    return 0;
}

static int write_exact(const uint8_t* buf, size_t len) {
    size_t put = 0;
    while (put < len) {
        size_t n = fwrite(buf + put, 1, len - put, stdout);
        if (n == 0) return -1;
        put += n;
    }
    return 0;
}

static void put_u32(uint32_t v) {
    uint8_t b[4] = {(uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
                    (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF)};
    write_exact(b, 4);
}

static void put_i32(int32_t v) { put_u32((uint32_t)v); }

static uint32_t get_u32(const uint8_t* b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static int32_t get_i32(const uint8_t* b) { return (int32_t)get_u32(b); }

static float get_f32(const uint8_t* b) {
    uint32_t bits = get_u32(b);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

// Reads a u32-length-prefixed string. Returns 0 on success, -1 on
// EOF/IO error (pipe desync — caller exits), -2 on malloc failure.
static int read_str(char** out) {
    uint8_t lenb[4];
    if (read_exact(lenb, 4) != 0) return -1;
    const uint32_t len = get_u32(lenb);
    char* s = (char*)malloc((size_t)len + 1);
    if (s == NULL) return -2;
    if (read_exact((uint8_t*)s, len) != 0) {
        free(s);
        return -1;
    }
    s[len] = '\0';
    *out = s;
    return 0;
}

// (Re)grows *buf to >= count items (frees and reallocates). Returns 0
// on success or when count already fits, -1 on malloc failure (in
// which case *buf is freed and *cap reset to 0).
static int grow(void** buf, size_t* cap, size_t count, size_t item) {
    if (count <= *cap) return 0;
    free(*buf);
    *buf = malloc(count * item);
    *cap = *buf != NULL ? count : 0;
    return *buf != NULL ? 0 : -1;
}

// Shared tail of predict (cmd=2) and extractF32 (cmd=4): runs fn with
// the shared output buffer, retrying once on HN_ERR_CAPACITY with the
// required size, then writes status | total | floats.
typedef int (*extract_fn)(hn_net_t, void* args, float* out, int out_cap,
                          int32_t* shapes, int shape_cap, int32_t* required);

static void run_extract(hn_net_t net, extract_fn fn, void* args,
                        float** out_buf, size_t* out_cap, int32_t* shapes,
                        int shape_cap) {
    int32_t required = 0;
    int n = HN_ERR_INVALID;
    if (net != NULL) {
        for (int attempt = 0; attempt < 2; attempt++) {
            n = fn(net, args, *out_buf, (int)*out_cap, shapes, shape_cap,
                   &required);
            if (n != HN_ERR_CAPACITY) break;
            // Buffer too small: grow to the required size, retry once.
            if (required <= 0 ||
                grow((void**)out_buf, out_cap, (size_t)required,
                     sizeof(float)) != 0) {
                n = HN_ERR_CAPACITY;
                break;
            }
        }
    }
    if (n < 0) {
        put_i32(n);
        put_u32(0); // Dart always reads an 8-byte head
    } else {
        put_i32(0);
        put_u32((uint32_t)n);
        write_exact((const uint8_t*)*out_buf, (size_t)n * sizeof(float));
    }
    fflush(stdout);
}

typedef struct {
    const uint8_t* pixels;
    int w, h;
} pixel_args;

static int extract_pixels(hn_net_t net, void* p, float* out, int out_cap,
                          int32_t* shapes, int shape_cap, int32_t* required) {
    const pixel_args* a = (const pixel_args*)p;
    return hn_extract(net, a->pixels, a->w, a->h, out, out_cap, shapes,
                      shape_cap, required);
}

typedef struct {
    const float* data;
    const int32_t* shape;
    int dims;
} f32_args;

static int extract_f32(hn_net_t net, void* p, float* out, int out_cap,
                       int32_t* shapes, int shape_cap, int32_t* required) {
    const f32_args* a = (const f32_args*)p;
    return hn_extract_f32(net, a->data, a->shape, a->dims, out, out_cap,
                          shapes, shape_cap, required);
}

int main(void) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    // This is a standalone process (NOT the Flutter app) — ncnn's
    // Vulkan stack is safe here. The shim's in-app guard must not apply.
    _putenv_s("NCNN_DART_ENABLE_VK", "1");
    // Multi-vendor device enumeration crashes on some driver combos
    // (Intel + NVIDIA dual-GPU laptops, observed 2026-09). If the user
    // hasn't pinned an ICD list, restrict to the NVIDIA ICD only — the
    // app only wants the discrete GPU for inference anyway.
    if (getenv("VK_ICD_FILENAMES") == NULL &&
        getenv("NCNN_DART_ALL_ICDS") == NULL) {
        const char* nvCandidates[] = {
            "C:\\Windows\\System32\\DriverStore\\FileRepository\\nvam.inf_amd64_19fcbbfeca9a5b09\\nv-vk64.json",
            NULL,
        };
        // Scan the driver store's INF directories for the first
        // nv-vk64.json (its directory name is driver-version-dependent).
        char keep[512] = "";
        WIN32_FIND_DATAA fdInf;
        HANDLE hInf = FindFirstFileA(
            "C:\\Windows\\System32\\DriverStore\\FileRepository\\*", &fdInf);
        if (hInf != INVALID_HANDLE_VALUE) {
            do {
                if (!(fdInf.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    continue;
                }
                char path[512];
                snprintf(path, sizeof(path),
                         "C:\\Windows\\System32\\DriverStore\\FileRepository\\%s\\nv-vk64.json",
                         fdInf.cFileName);
                if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) {
                    snprintf(keep, sizeof(keep), "%s", path);
                    break;
                }
            } while (FindNextFileA(hInf, &fdInf));
            FindClose(hInf);
        }
        if (keep[0] == '\0') {
            for (int i = 0; nvCandidates[i] != NULL; i++) {
                if (GetFileAttributesA(nvCandidates[i]) !=
                    INVALID_FILE_ATTRIBUTES) {
                    snprintf(keep, sizeof(keep), "%s", nvCandidates[i]);
                    break;
                }
            }
        }
        if (keep[0] != '\0') {
            _putenv_s("VK_ICD_FILENAMES", keep);
        }
    }
#endif

    hn_net_t net = NULL;
    float* out_buf = NULL;
    size_t out_cap = 0;   // float count
    int32_t* shapes = NULL; // 4 entries per output, from the last load
    size_t shape_cap = 0;  // int32_t count
    int out_count = 0;
    uint8_t hdr[4];

    for (;;) {
        if (read_exact(hdr, 4) != 0) break;
        const uint32_t cmd = get_u32(hdr);

        if (cmd == 0) break; // quit

        if (cmd == 1) { // load
            // Paths arrive as UTF-8 bytes (Dart utf8-encodes them) and are
            // passed verbatim to hn_load/ncnn (fopen). On Windows, non-ASCII
            // paths therefore depend on the process's active code page being
            // UTF-8 — a known upstream ncnn limitation, documented in the
            // package README as a caveat.
            char *param = NULL, *bin = NULL, *blob = NULL;
            int32_t use_vulkan = 0, device_index = -1, pixel_format = 0;
            int32_t warmup_w = 0, warmup_h = 0;
            float mean[3] = {0, 0, 0}, norm[3] = {1, 1, 1};
            // 0 = ok, -1 = EOF/pipe desync (fatal), -2 = malformed frame
            // or malloc failure (answer INVALID, keep the pipe alive).
            int bad = read_str(&param);
            if (bad == 0) bad = read_str(&bin);
            if (bad == 0) {
                // opts: i32 useVulkan, i32 deviceIndex, i32 pixelFormat,
                // u32 blobLen | blob, f32 mean[3], f32 norm[3],
                // i32 warmup_w, i32 warmup_h. Fixed part 48 bytes,
                // optsLen = 48 + blobLen (+ appended fields, skipped).
                uint8_t lenb[4];
                if (read_exact(lenb, 4) != 0) {
                    bad = -1;
                } else {
                    const uint32_t optsLen = get_u32(lenb);
                    uint8_t head[16];
                    if (optsLen < 48 || read_exact(head, 16) != 0) {
                        bad = optsLen < 48 ? -2 : -1;
                    } else {
                        use_vulkan = get_i32(head);
                        device_index = get_i32(head + 4);
                        pixel_format = get_i32(head + 8);
                        const uint32_t blobLen = get_u32(head + 12);
                        if (optsLen < 48 + blobLen) {
                            bad = -2;
                        } else {
                            blob = (char*)malloc((size_t)blobLen + 1);
                            if (blob == NULL) {
                                bad = -2;
                            } else if (read_exact((uint8_t*)blob,
                                                  blobLen) != 0) {
                                bad = -1;
                            } else {
                                blob[blobLen] = '\0';
                                uint8_t tail[32]; // mean[3] norm[3] w h
                                if (read_exact(tail, 32) != 0) {
                                    bad = -1;
                                } else {
                                    for (int i = 0; i < 3; i++) {
                                        mean[i] = get_f32(tail + i * 4);
                                        norm[i] = get_f32(tail + 12 + i * 4);
                                    }
                                    warmup_w = get_i32(tail + 24);
                                    warmup_h = get_i32(tail + 28);
                                    // Skip appended (unknown) fields.
                                    const uint32_t extra =
                                        optsLen - 48 - blobLen;
                                    if (extra > 0) {
                                        uint8_t* skip = (uint8_t*)malloc(extra);
                                        if (skip == NULL) {
                                            bad = -2;
                                        } else {
                                            if (read_exact(skip, extra) != 0) {
                                                bad = -1;
                                            }
                                            free(skip);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (bad == -1) {
                // Pipe desync — nothing sane to answer, exit.
                free(param);
                free(bin);
                free(blob);
                break;
            }
            if (bad != 0 || param == NULL || bin == NULL) {
                // Malformed frame or malloc failure — the request body
                // was not fully consumed, so the pipe is desynced.
                // Answer the current request (Dart reads an 8-byte
                // head) and exit; the next Dart read fails with a
                // pipe-closed error instead of a hang.
                put_i32(HN_ERR_INVALID);
                put_u32(0);
                fflush(stdout);
                free(param);
                free(bin);
                free(blob);
                break;
            }

            hn_options_t opts;
            memset(&opts, 0, sizeof(opts));
            opts.struct_size = (uint32_t)sizeof(hn_options_t);
            opts.use_vulkan = use_vulkan;
            opts.device_index = device_index;
            opts.pixel_format = pixel_format;
            memcpy(opts.mean, mean, sizeof(mean));
            memcpy(opts.norm, norm, sizeof(norm));
            opts.input_blob = blob;
            opts.warmup_w = warmup_w;
            opts.warmup_h = warmup_h;

            // A second load replaces the previous net.
            if (net != NULL) hn_destroy(net);
            net = hn_create(&opts);
            free(blob); // hn_create copies input_blob into a std::string
            const int rc =
                net != NULL ? hn_load(net, param, bin) : HN_ERR_INVALID;
            free(param);
            free(bin);

            // Rows are only sent on success — after a failed reload the
            // net's shape hints may be stale, and Dart ignores them on
            // a non-zero status anyway.
            out_count = (rc == HN_OK && net != NULL) ? hn_output_count(net)
                                                     : 0;
            if (out_count > 0 &&
                grow((void**)&shapes, &shape_cap, (size_t)out_count * 4,
                     sizeof(int32_t)) != 0) {
                out_count = 0; // hn_extract accepts NULL shapes
            }
            put_i32(rc);
            put_u32((uint32_t)out_count);
            if (out_count > 0) {
                size_t need = 0;
                for (int i = 0; i < out_count; i++) {
                    int32_t s[4] = {0, 0, 0, 0};
                    // Unwritten (all 0) when no warm-up hint exists.
                    const int dims = hn_output_shape(net, i, s);
                    if (shapes != NULL) {
                        memcpy(shapes + i * 4, s, sizeof(s));
                    }
                    size_t elems = 1;
                    for (int k = 0; k < 4; k++) {
                        elems *= s[k] > 0 ? (size_t)s[k] : 1;
                    }
                    need += elems;
                    put_u32(dims > 0 ? (uint32_t)dims : 0);
                    put_u32((uint32_t)s[0]);
                    put_u32((uint32_t)s[1]);
                    put_u32((uint32_t)s[2]);
                    put_u32((uint32_t)s[3]);
                }
                if (need > out_cap) {
                    grow((void**)&out_buf, &out_cap, need, sizeof(float));
                }
            }
            fflush(stdout);
            continue;
        }

        if (cmd == 2) { // predict (pixels)
            uint8_t meta[12];
            if (read_exact(meta, 12) != 0) break;
            const uint32_t w = get_u32(meta);
            const uint32_t h = get_u32(meta + 4);
            const uint32_t frameLen = get_u32(meta + 8);
            uint8_t* frame =
                (uint8_t*)malloc(frameLen > 0 ? frameLen : 1);
            if (frame == NULL) {
                put_i32(HN_ERR_INVALID);
                put_u32(0);
                fflush(stdout);
                break; // body not consumed — pipe desync
            }
            if (read_exact(frame, frameLen) != 0) {
                free(frame);
                break;
            }
            pixel_args args = {frame, (int)w, (int)h};
            run_extract(net, extract_pixels, &args, &out_buf, &out_cap,
                        shapes, (int)shape_cap);
            free(frame);
            continue;
        }

        if (cmd == 4) { // extractF32 (raw float input)
            uint8_t dimsb[4];
            if (read_exact(dimsb, 4) != 0) break;
            const uint32_t dims = get_u32(dimsb);
            if (dims < 1 || dims > 4) {
                put_i32(HN_ERR_INVALID);
                put_u32(0);
                fflush(stdout);
                break; // rest of the frame not consumed — pipe desync
            }
            int32_t shape[4] = {1, 1, 1, 1};
            uint8_t sb[16];
            if (read_exact(sb, dims * 4) != 0) break;
            for (uint32_t i = 0; i < dims; i++) {
                shape[i] = get_i32(sb + i * 4);
            }
            uint8_t lenb[4];
            if (read_exact(lenb, 4) != 0) break;
            const uint32_t dataLen = get_u32(lenb); // float count
            float* data = (float*)malloc(
                dataLen > 0 ? (size_t)dataLen * sizeof(float) : sizeof(float));
            if (data == NULL) {
                put_i32(HN_ERR_INVALID);
                put_u32(0);
                fflush(stdout);
                break; // body not consumed — pipe desync
            }
            if (read_exact((uint8_t*)data, (size_t)dataLen * 4) != 0) {
                free(data);
                break;
            }
            f32_args args = {data, shape, (int)dims};
            run_extract(net, extract_f32, &args, &out_buf, &out_cap, shapes,
                        (int)shape_cap);
            free(data);
            continue;
        }

        if (cmd == 3) { // gpu enumeration (unchanged from v1)
            hn_gpu_device_t devices[HN_MAX_GPU];
            char names[HN_MAX_GPU][HN_NAME_MAX];
            const int count = hn_gpu_devices(devices, names, HN_MAX_GPU);
            put_u32(0);
            put_u32((uint32_t)count);
            for (int i = 0; i < count; i++) {
                put_u32((uint32_t)devices[i].index);
                put_u32((uint32_t)devices[i].type);
                put_u32((uint32_t)devices[i].score);
                put_u32(devices[i].vendor_id);
                const uint32_t nameLen =
                    (uint32_t)strnlen(names[i], HN_NAME_MAX);
                put_u32(nameLen);
                write_exact((const uint8_t*)names[i], nameLen);
            }
            fflush(stdout);
            continue;
        }
    }

    if (net != NULL) hn_destroy(net);
    free(out_buf);
    free(shapes);
    return 0;
}
