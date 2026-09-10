// Public C surface of the ncnn Dart/Flutter plugin.
//
// Thin extern "C" shim over ncnn::Net. The surface is small and versioned:
// hn_options_t carries struct_size so fields can be appended without
// breaking binary compatibility. Symbols keep the hn_ prefix so they
// cannot collide with ncnn's own C++ symbols.
#ifndef NCNN_DART_API_H
#define NCNN_DART_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define HN_API __declspec(dllexport)
#else
#define HN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a loaded ncnn network.
typedef void* hn_net_t;

/// Vulkan device info returned by hn_gpu_devices.
typedef struct {
    int index;          /// ncnn device index
    int type;           /// 0=discrete, 1=integrated, 2=virtual, 3=cpu
    int score;          /// ncnn rough_score heuristic (higher = faster)
    uint32_t vendor_id; /// PCI vendor id
} hn_gpu_device_t;

#define HN_MAX_GPU 8
#define HN_NAME_MAX 128

/// Input pixel layout (source buffer layout; from_pixels produces a
/// 3-channel Mat, or 1-channel for GRAY).
enum hn_pixel_format {
    HN_PIX_RGB = 0,
    HN_PIX_BGR = 1,
    HN_PIX_RGBA = 2,
    HN_PIX_BGRA = 3,
    HN_PIX_GRAY = 4,
};

/// Error codes (returned as negative ints).
enum hn_error {
    HN_OK = 0,
    HN_ERR_INVALID = -1,       /// null args / bad state / bad options
    HN_ERR_PARAM = -2,         /// load_param failed (or no output blobs)
    HN_ERR_BIN = -3,           /// load_model failed
    HN_ERR_NO_INPUT_BLOB = -4, /// configured input blob not in the graph
    HN_ERR_EXTRACT = -5,       /// extractor input/extract failed
    HN_ERR_SHAPE = -6,         /// output blob has no usable elements
    HN_ERR_CAPACITY = -7,      /// out too small (*required_out set)
    HN_ERR_EXCEPTION = -8,     /// C++ exception caught at the boundary
};

/// Network options. struct_size MUST be sizeof(hn_options_t) as compiled
/// by the caller; fields are only ever APPENDED, and hn_create accepts
/// any struct_size in [HN_OPTIONS_V1_SIZE, sizeof(hn_options_t)].
/// mean/norm are per-channel; the first N entries are used for the actual
/// channel count (GRAY=1, others=3).
typedef struct {
    uint32_t struct_size;
    int use_vulkan;
    int device_index;       /// -1 = ncnn default device
    float mean[3];
    float norm[3];          /// all-1 = no scaling; YOLO convention = 1/255
    int pixel_format;       /// enum hn_pixel_format
    const char* input_blob; /// NULL = "in0" (ultralytics convention)
    int warmup_w;           /// warm-up width for shape hints; 0 = auto
    int warmup_h;           /// warm-up height for shape hints; 0 = auto
} hn_options_t;

#define HN_OPTIONS_V1_SIZE \
    ((uint32_t)(offsetof(hn_options_t, input_blob) + sizeof(const char*)))

/// Create a network. opts may be NULL for all defaults. Returns NULL on
/// OOM or a struct_size outside the accepted range.
HN_API hn_net_t hn_create(const hn_options_t* opts);

/// Load param+bin. Discovers graph output blobs by parsing the param
/// file (blobs produced but never consumed) and runs a warm-up (64x64
/// zeros by default, or warmup_w x warmup_h from the options) to record
/// output shapes as hints for hn_output_shape. Auto warm-up is SKIPPED
/// when the param contains a Reshape layer: such graphs are usually
/// locked to the trained input size (fixed reshape dims), and ncnn's
/// Reshape does not validate element totals — warming up at the wrong
/// size reads/writes out of bounds and corrupts the heap. Callers that
/// know the input size (e.g. from model metadata) pass warmup_w/h
/// explicitly to get hints; otherwise the first hn_extract's
/// capacity-retry provides the sizes.
HN_API int hn_load(hn_net_t net, const char* param_path, const char* bin_path);

/// Number of graph output blobs (0 before a successful hn_load).
HN_API int hn_output_count(hn_net_t net);

/// Fill shape[0..3] = (w, h, d|1, c) for output `out_index`, padded with
/// 1s. shape[] is only written when a warm-up hint exists, i.e. the
/// return is non-zero; a return of 0 leaves shape untouched (get the
/// size via the hn_extract capacity-retry). Returns the blob's ncnn
/// dims, or 0 when the hint is unavailable.
HN_API int hn_output_shape(hn_net_t net, int out_index, int32_t shape[4]);

/// Run inference on a pixel buffer (layout per options.pixel_format,
/// w*h*channels bytes) and write ALL output blobs, concatenated, into
/// `out`. shapes (may be NULL) gets 4 entries per output, same
/// convention as hn_output_shape but for THIS run. On HN_ERR_CAPACITY,
/// *required_out (if non-NULL) receives the needed float count.
/// Returns total floats written, or a negative enum hn_error.
HN_API int hn_extract(hn_net_t net, const uint8_t* pixels, int w, int h,
                      float* out, int out_cap, int32_t* shapes,
                      int shape_cap, int32_t* required_out);

/// Same, with a raw float input tensor (no pixel decode, no mean/norm —
/// the caller preprocesses). shape holds `dims` entries (1..4, ncnn
/// order w/h/d/c), data is contiguous in that order.
HN_API int hn_extract_f32(hn_net_t net, const float* data,
                          const int32_t* shape, int dims, float* out,
                          int out_cap, int32_t* shapes, int shape_cap,
                          int32_t* required_out);

/// Destroy a network (NULL is a no-op).
HN_API void hn_destroy(hn_net_t net);

/// Count available Vulkan devices (0 if Vulkan is unavailable/disabled).
HN_API int hn_gpu_count(void);

/// Fill devices[0..n-1] and names[i][HN_NAME_MAX] for up to n Vulkan
/// devices. Returns the number written.
HN_API int hn_gpu_devices(hn_gpu_device_t* devices,
                          char (*names)[HN_NAME_MAX], int n);

#ifdef __cplusplus
}
#endif

#endif // NCNN_DART_API_H
