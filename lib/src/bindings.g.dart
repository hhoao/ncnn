// Hand-maintained bindings for src/ncnn_api.h.
//
// ffigen's generated shape cannot plug into NcnnRuntime's custom
// DynamicLibrary resolution chain (test-VM marker machinery), so the
// bindings are hand-written against the header. Symbol coverage is
// guaranteed by test/symbol_coverage_test.dart, which opens the library
// and looks up every function below.
// ignore_for_file: always_specify_types, constant_identifier_names

library;

import 'dart:ffi';

/// Opaque handle to a loaded ncnn network.
typedef HnNet = Pointer<Void>;

/// enum hn_pixel_format values.
const int hnPixRgb = 0;
const int hnPixBgr = 1;
const int hnPixRgba = 2;
const int hnPixBgra = 3;
const int hnPixGray = 4;

/// enum hn_error values.
const int hnOk = 0;
const int hnErrInvalid = -1;
const int hnErrParam = -2;
const int hnErrBin = -3;
const int hnErrNoInputBlob = -4;
const int hnErrExtract = -5;
const int hnErrShape = -6;
const int hnErrCapacity = -7;
const int hnErrException = -8;

const int hnMaxGpu = 8;
const int hnNameMax = 128;

/// int32 fields — C struct is 4*int + uint32 = 20 bytes, aligned.
final class HnGpuDevice extends Struct {
  @Int32()
  external int index;
  @Int32()
  external int type;
  @Int32()
  external int score;
  @Uint32()
  external int vendorId;
}

/// C mirror of hn_options_t — field order and types must match exactly
/// (struct_size is validated by hn_create).
final class HnOptions extends Struct {
  @Uint32()
  external int structSize;
  @Int32()
  external int useVulkan;
  @Int32()
  external int deviceIndex;
  @Array.multi([3])
  external Array<Float> mean;
  @Array.multi([3])
  external Array<Float> norm;
  @Int32()
  external int pixelFormat;
  external Pointer<Char> inputBlob;
  @Int32()
  external int warmupW;
  @Int32()
  external int warmupH;
}

typedef HnCreateNative = Pointer<Void> Function(Pointer<HnOptions>);
typedef HnCreateDart = Pointer<Void> Function(Pointer<HnOptions>);
typedef HnLoadNative = Int32 Function(
    Pointer<Void>, Pointer<Char>, Pointer<Char>);
typedef HnLoadDart = int Function(Pointer<Void>, Pointer<Char>, Pointer<Char>);
typedef HnOutputCountNative = Int32 Function(Pointer<Void>);
typedef HnOutputCountDart = int Function(Pointer<Void>);
typedef HnOutputShapeNative = Int32 Function(
    Pointer<Void>, Int32, Pointer<Int32>);
typedef HnOutputShapeDart = int Function(Pointer<Void>, int, Pointer<Int32>);
typedef HnExtractNative = Int32 Function(Pointer<Void>, Pointer<Uint8>, Int32,
    Int32, Pointer<Float>, Int32, Pointer<Int32>, Int32, Pointer<Int32>);
typedef HnExtractDart = int Function(Pointer<Void>, Pointer<Uint8>, int, int,
    Pointer<Float>, int, Pointer<Int32>, int, Pointer<Int32>);
typedef HnExtractF32Native = Int32 Function(
    Pointer<Void>,
    Pointer<Float>,
    Pointer<Int32>,
    Int32,
    Pointer<Float>,
    Int32,
    Pointer<Int32>,
    Int32,
    Pointer<Int32>);
typedef HnExtractF32Dart = int Function(
    Pointer<Void>,
    Pointer<Float>,
    Pointer<Int32>,
    int,
    Pointer<Float>,
    int,
    Pointer<Int32>,
    int,
    Pointer<Int32>);
typedef HnDestroyNative = Void Function(Pointer<Void>);
typedef HnDestroyDart = void Function(Pointer<Void>);
typedef HnGpuCountNative = Int32 Function();
typedef HnGpuCountDart = int Function();
// names is char (*)[hnNameMax] — inline fixed-size rows, not pointers.
typedef HnGpuDevicesNative = Int32 Function(
    Pointer<HnGpuDevice>, Pointer<Uint8>, Int32);
typedef HnGpuDevicesDart = int Function(
    Pointer<HnGpuDevice>, Pointer<Uint8>, int);

/// Looked-up C entry points of the ncnn shim.
class NcnnNative {
  NcnnNative._();

  late final HnCreateDart hnCreate;
  late final HnLoadDart hnLoad;
  late final HnOutputCountDart hnOutputCount;
  late final HnOutputShapeDart hnOutputShape;
  late final HnExtractDart hnExtract;
  late final HnExtractF32Dart hnExtractF32;
  late final HnDestroyDart hnDestroy;
  late final HnGpuCountDart hnGpuCount;
  late final HnGpuDevicesDart hnGpuDevices;

  /// Resolves all shim symbols from [lib].
  factory NcnnNative.fromLibrary(DynamicLibrary lib) {
    final n = NcnnNative._();
    n.hnCreate = lib.lookupFunction<HnCreateNative, HnCreateDart>('hn_create');
    n.hnLoad = lib.lookupFunction<HnLoadNative, HnLoadDart>('hn_load');
    n.hnOutputCount =
        lib.lookupFunction<HnOutputCountNative, HnOutputCountDart>(
            'hn_output_count');
    n.hnOutputShape =
        lib.lookupFunction<HnOutputShapeNative, HnOutputShapeDart>(
            'hn_output_shape');
    n.hnExtract =
        lib.lookupFunction<HnExtractNative, HnExtractDart>('hn_extract');
    n.hnExtractF32 = lib
        .lookupFunction<HnExtractF32Native, HnExtractF32Dart>('hn_extract_f32');
    n.hnDestroy =
        lib.lookupFunction<HnDestroyNative, HnDestroyDart>('hn_destroy');
    n.hnGpuCount =
        lib.lookupFunction<HnGpuCountNative, HnGpuCountDart>('hn_gpu_count');
    n.hnGpuDevices = lib
        .lookupFunction<HnGpuDevicesNative, HnGpuDevicesDart>('hn_gpu_devices');
    return n;
  }
}
