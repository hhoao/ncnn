import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.g.dart' as native;
import 'options.dart';
import 'runtime.dart';

/// One output blob of an inference run.
class NcnnOutput {
  const NcnnOutput({required this.shape, required this.data});

  /// 4 entries: (w, h, d|1, c) — padded with 1s, ncnn convention.
  final List<int> shape;

  /// Flattened values in ncnn order.
  final Float32List data;
}

/// A loaded ncnn network.
///
/// NOT thread-safe: one extract at a time per instance (ncnn::Net is
/// not reentrant across concurrent extractors on the same net). Use
/// [extractInIsolate] to keep the UI isolate responsive — but never two
/// isolates on the same instance concurrently.
class NcnnNet {
  NcnnNet._(this._handle, this._outputCount, this._shapeHints, int capHint)
      : _capHint = capHint;

  Pointer<Void>? _handle;
  final int _outputCount;
  final List<List<int>?> _shapeHints;
  int _capHint;

  /// Loads a model. [options] selects Vulkan/device and preprocessing.
  /// Throws [StateError] on any failure — caller decides whether to
  /// retry on CPU (see NcnnInferenceEngine).
  static Future<NcnnNet> load({
    required String paramPath,
    required String binPath,
    NcnnOptions options = const NcnnOptions(),
  }) async {
    final rt = NcnnRuntime.instance;
    final optsPtr = options.toNative();
    final handle = rt.lib.hnCreate(optsPtr);
    NcnnOptions.freeNative(optsPtr);
    if (handle == nullptr) {
      throw StateError('hn_create failed (OOM or bad options struct)');
    }
    final paramPtr = paramPath.toNativeUtf8();
    final binPtr = binPath.toNativeUtf8();
    try {
      final status = rt.lib.hnLoad(handle, paramPtr.cast(), binPtr.cast());
      if (status != native.hnOk) {
        rt.lib.hnDestroy(handle);
        throw StateError('hn_load failed with status $status '
            '(-2 param, -3 bin, -8 exception)');
      }
    } finally {
      calloc.free(paramPtr);
      calloc.free(binPtr);
    }

    final count = rt.lib.hnOutputCount(handle);
    if (count <= 0) {
      rt.lib.hnDestroy(handle);
      throw StateError('model has no discoverable output blobs');
    }
    final hints = <List<int>?>[];
    var cap = 0;
    final shapeBuf = calloc<Int32>(4);
    try {
      for (var i = 0; i < count; i++) {
        final dims = rt.lib.hnOutputShape(handle, i, shapeBuf);
        if (dims == 0) {
          hints.add(null);
          continue;
        }
        final shape = List<int>.generate(4, (k) => shapeBuf[k]);
        hints.add(shape);
        cap += shape.fold(1, (a, b) => a * b);
      }
    } finally {
      calloc.free(shapeBuf);
    }
    return NcnnNet._(handle, count, hints, cap > 0 ? cap : 256);
  }

  int get outputCount => _outputCount;

  /// Warm-up shape hint for output [i] — 4 entries (w, h, d|1, c), or
  /// null when unavailable (shapes may vary with input size).
  List<int>? outputShape(int i) =>
      (i >= 0 && i < _outputCount) ? _shapeHints[i] : null;

  /// Runs inference on a pixel buffer (w*h*channels bytes, layout per
  /// options.pixelFormat). Returns all output blobs in param order.
  ///
  /// Callers must feed the model's expected input size (e.g. the `imgsz`
  /// width/height from ultralytics export metadata): upstream ncnn
  /// Reshape does not validate element totals, so a size-locked model
  /// run at a different size heap-corrupts instead of failing.
  List<NcnnOutput> extract(Uint8List pixels, int width, int height) {
    final rt = NcnnRuntime.instance;
    final handle = _handle;
    if (handle == null) {
      throw StateError('NcnnNet disposed or not loaded');
    }
    final pixPtr = calloc<Uint8>(pixels.length);
    pixPtr.asTypedList(pixels.length).setAll(0, pixels);
    try {
      return _run((out, shapes, required) => rt.lib.hnExtract(
            handle,
            pixPtr,
            width,
            height,
            out,
            _capHint,
            shapes,
            _outputCount * 4,
            required,
          ));
    } finally {
      calloc.free(pixPtr);
    }
  }

  /// Same, with a raw float input tensor (caller-preprocessed).
  /// [shape] has 1..4 entries in ncnn order (w, h, d, c).
  List<NcnnOutput> extractF32(Float32List data, List<int> shape) {
    final rt = NcnnRuntime.instance;
    final handle = _handle;
    if (handle == null) {
      throw StateError('NcnnNet disposed or not loaded');
    }
    if (shape.isEmpty || shape.length > 4) {
      throw ArgumentError.value(shape, 'shape', 'must have 1..4 entries');
    }
    final dataPtr = calloc<Float>(data.length);
    dataPtr.asTypedList(data.length).setAll(0, data);
    final shapePtr = calloc<Int32>(shape.length);
    for (var i = 0; i < shape.length; i++) {
      shapePtr[i] = shape[i];
    }
    try {
      return _run((out, shapes, required) => rt.lib.hnExtractF32(
            handle,
            dataPtr,
            shapePtr,
            shape.length,
            out,
            _capHint,
            shapes,
            _outputCount * 4,
            required,
          ));
    } finally {
      calloc.free(dataPtr);
      calloc.free(shapePtr);
    }
  }

  /// [extract] on a background isolate — keeps the UI isolate free of
  /// the (tens of ms on CPU) synchronous FFI call. Still serialized per
  /// instance: never call concurrently from two isolates.
  Future<List<NcnnOutput>> extractInIsolate(
      Uint8List pixels, int width, int height) {
    return Isolate.run(() => extract(pixels, width, height));
  }

  /// Convenience: first output blob's flat data (classify models).
  Float32List predict(Uint8List pixels, int width, int height) {
    final outputs = extract(pixels, width, height);
    if (outputs.isEmpty) {
      throw StateError('no outputs produced');
    }
    return outputs.first.data;
  }

  /// Shared capacity-retry loop: calls [fn] with an out buffer; on
  /// HN_ERR_CAPACITY retries with the required size (until success or
  /// a hard error).
  List<NcnnOutput> _run(
      int Function(Pointer<Float>, Pointer<Int32>, Pointer<Int32>) fn) {
    var cap = _capHint;
    while (true) {
      final out = calloc<Float>(cap);
      final shapes = calloc<Int32>(_outputCount * 4);
      final required = calloc<Int32>();
      try {
        final n = fn(out, shapes, required);
        if (n == native.hnErrCapacity) {
          cap = required.value;
          if (cap <= 0) {
            throw StateError('hn_extract reported capacity 0 required');
          }
          continue; // finally frees; retry with the right size
        }
        if (n <= 0) {
          throw StateError('hn_extract failed with status $n');
        }
        _capHint = cap;
        final shapeList = <List<int>>[];
        for (var i = 0; i < _outputCount; i++) {
          shapeList.add(List<int>.generate(4, (k) => shapes[i * 4 + k]));
        }
        // Copy first, then hand out sublistView slices of the copy —
        // views of `out` would dangle after the free in finally.
        final flat = Float32List.fromList(out.asTypedList(n));
        final parts = splitOutputs(flat, shapeList);
        return List.generate(_outputCount,
            (i) => NcnnOutput(shape: shapeList[i], data: parts[i]));
      } finally {
        calloc.free(out);
        calloc.free(shapes);
        calloc.free(required);
      }
    }
  }

  /// Slices a flat multi-output buffer by per-output shapes.
  /// Pure function (unit-tested directly).
  static List<Float32List> splitOutputs(
      Float32List flat, List<List<int>> shapes) {
    final parts = <Float32List>[];
    var offset = 0;
    for (final shape in shapes) {
      var elems = 1;
      for (final d in shape) {
        elems *= d;
      }
      if (offset + elems > flat.length) {
        throw ArgumentError(
            'shapes ($shapes) exceed buffer length ${flat.length}');
      }
      parts.add(Float32List.sublistView(flat, offset, offset + elems));
      offset += elems;
    }
    return parts;
  }

  void dispose() {
    final handle = _handle;
    _handle = null;
    if (handle != null) {
      NcnnRuntime.instance.lib.hnDestroy(handle);
    }
  }
}
