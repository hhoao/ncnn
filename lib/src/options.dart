import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings.g.dart' as native;

/// Input pixel layout (source buffer layout; the shim produces a
/// 3-channel tensor, or 1-channel for gray).
enum NcnnPixelFormat {
  rgb(native.hnPixRgb),
  bgr(native.hnPixBgr),
  rgba(native.hnPixRgba),
  bgra(native.hnPixBgra),
  gray(native.hnPixGray);

  const NcnnPixelFormat(this.value);

  final int value;
}

/// Preprocessing / backend options for [NcnnNet.load].
///
/// Defaults follow the ultralytics ncnn export convention so YOLO models
/// work with `const NcnnOptions()`; everything is overridable for other
/// models.
class NcnnOptions {
  const NcnnOptions({
    this.useVulkan = false,
    this.deviceIndex = -1,
    this.mean = const [0, 0, 0],
    this.norm = const [1, 1, 1],
    this.pixelFormat = NcnnPixelFormat.rgb,
    this.inputBlob,
    this.warmupWidth = 0,
    this.warmupHeight = 0,
  });

  /// ultralytics YOLO convention: RGB, x/255, no mean subtraction,
  /// input blob "in0". [warmupWidth]/[warmupHeight] pass the exported
  /// imgsz through for shape hints (required for Reshape graphs — see
  /// [NcnnOptions] and the helper protocol).
  const NcnnOptions.yolo({
    this.useVulkan = false,
    this.deviceIndex = -1,
    this.warmupWidth = 0,
    this.warmupHeight = 0,
  })  : mean = const [0, 0, 0],
        norm = const [_oneOver255, _oneOver255, _oneOver255],
        pixelFormat = NcnnPixelFormat.rgb,
        inputBlob = null;

  static const double _oneOver255 = 1.0 / 255.0;

  /// Enable Vulkan compute. CPU fallback happens at a higher level
  /// ([NcnnInferenceEngine]); a failed Vulkan load throws here.
  final bool useVulkan;

  /// Vulkan device index (-1 = ncnn default). Ignored unless
  /// [useVulkan].
  final int deviceIndex;

  /// Per-channel mean subtraction (RGB order).
  final List<double> mean;

  /// Per-channel norm division (all-1 = no scaling).
  final List<double> norm;

  final NcnnPixelFormat pixelFormat;

  /// Input blob name; null = "in0" (ultralytics convention).
  final String? inputBlob;

  /// Warm-up input width for shape hints; 0 = auto (64x64, skipped when
  /// the graph has a Reshape layer). Must be paired with
  /// [warmupHeight].
  final int warmupWidth;

  /// Warm-up input height; must be >0 together with [warmupWidth] to
  /// take effect.
  final int warmupHeight;

  /// Allocates the native mirror (caller must [freeNative]).
  Pointer<native.HnOptions> toNative() {
    assert(mean.length == 3, 'mean must have 3 entries');
    assert(norm.length == 3, 'norm must have 3 entries');
    final p = calloc<native.HnOptions>();
    final s = p.ref;
    s.structSize = sizeOf<native.HnOptions>();
    s.useVulkan = useVulkan ? 1 : 0;
    s.deviceIndex = deviceIndex;
    for (var i = 0; i < 3; i++) {
      s.mean[i] = mean[i];
      s.norm[i] = norm[i];
    }
    s.pixelFormat = pixelFormat.value;
    s.inputBlob = (inputBlob ?? 'in0').toNativeUtf8().cast<Char>();
    s.warmupW = warmupWidth;
    s.warmupH = warmupHeight;
    return p;
  }

  /// Frees a pointer returned by [toNative] (including the blob string).
  static void freeNative(Pointer<native.HnOptions> p) {
    if (p == nullptr) return;
    calloc.free(p.ref.inputBlob);
    calloc.free(p);
  }
}
