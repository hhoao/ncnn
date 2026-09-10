import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'options.dart';
import 'runtime.dart' show NcnnGpuDevice;

/// Wire protocol v2 of the `ncnn_helper` child process (all little-endian,
/// see src/ncnn_helper_main.cpp):
///
/// ```
/// load:       u32 cmd=1 | u32 paramLen | param | u32 binLen | bin |
///             u32 optsLen | opts
///   opts:     i32 useVulkan, i32 deviceIndex, i32 pixelFormat,
///             u32 blobLen | blob, f32 mean[3], f32 norm[3],
///             i32 warmup_w, i32 warmup_h          (fixed part = 48 bytes)
///   response: i32 status | u32 outCount |
///             outCount × (u32 dims, u32 s0..s3)
/// predict:    u32 cmd=2 | u32 w | u32 h | u32 frameLen | RGB bytes
///   response: i32 status | u32 total | total × f32 (all outputs concat)
/// extractF32: u32 cmd=4 | u32 dims | dims × u32 shape |
///             u32 dataLen | f32 data
///   response: i32 status | u32 total | total × f32
/// gpu:        u32 cmd=3
///   response: u32 status | u32 count | count × (u32 idx, u32 type,
///             u32 score, u32 vendor, u32 nameLen, name bytes)
/// quit:       u32 cmd=0
/// ```
///
/// Pure functions — unit-testable without spawning a process.
class NcnnHelperFrames {
  NcnnHelperFrames._();

  static void _u32(BytesBuilder b, int v) {
    b.add(Uint8List(4)..buffer.asByteData().setUint32(0, v, Endian.little));
  }

  static void _i32(BytesBuilder b, int v) {
    b.add(Uint8List(4)..buffer.asByteData().setInt32(0, v, Endian.little));
  }

  static void _f32(BytesBuilder b, double v) {
    b.add(Uint8List(4)..buffer.asByteData().setFloat32(0, v, Endian.little));
  }

  static void _bytes(BytesBuilder b, List<int> v) => b.add(v);

  /// Bare cmd frame (gpu / quit style — no payload).
  static Uint8List cmdRequest(int cmd) {
    final b = BytesBuilder();
    _u32(b, cmd);
    return b.takeBytes();
  }

  /// cmd=3: enumerate Vulkan devices.
  static Uint8List gpuRequest() => cmdRequest(3);

  /// cmd=1: load with full options.
  static Uint8List loadRequest({
    required String paramPath,
    required String binPath,
    required NcnnOptions options,
  }) {
    final b = BytesBuilder();
    _u32(b, 1);
    final param = paramPath.codeUnits;
    final bin = binPath.codeUnits;
    _u32(b, param.length);
    _bytes(b, param);
    _u32(b, bin.length);
    _bytes(b, bin);
    final blob = (options.inputBlob ?? 'in0').codeUnits;
    _u32(b, 48 + blob.length); // fixed part 48 + blob bytes
    _i32(b, options.useVulkan ? 1 : 0);
    _i32(b, options.deviceIndex);
    _i32(b, options.pixelFormat.value);
    _u32(b, blob.length);
    _bytes(b, blob);
    for (var i = 0; i < 3; i++) {
      _f32(b, options.mean[i]);
    }
    for (var i = 0; i < 3; i++) {
      _f32(b, options.norm[i]);
    }
    _i32(b, options.warmupWidth);
    _i32(b, options.warmupHeight);
    return b.takeBytes();
  }

  /// cmd=1 response → (status, per-output shape hints).
  static (int, List<List<int>>) parseLoadResponse(Uint8List bytes) {
    final d = ByteData.sublistView(bytes);
    var off = 0;
    final status = d.getInt32(off, Endian.little);
    off += 4;
    final count = d.getUint32(off, Endian.little);
    off += 4;
    final shapes = <List<int>>[];
    for (var i = 0; i < count; i++) {
      off += 4; // dims (redundant with shape entries)
      shapes.add([
        d.getUint32(off, Endian.little),
        d.getUint32(off + 4, Endian.little),
        d.getUint32(off + 8, Endian.little),
        d.getUint32(off + 12, Endian.little),
      ]);
      off += 16;
    }
    return (status, shapes);
  }

  /// cmd=2 request body.
  static Uint8List predictRequest(Uint8List rgb, int w, int h) {
    final b = BytesBuilder();
    _u32(b, 2);
    _u32(b, w);
    _u32(b, h);
    _u32(b, rgb.length);
    _bytes(b, rgb);
    return b.takeBytes();
  }

  /// cmd=4 request body.
  static Uint8List extractF32Request(Float32List data, List<int> shape) {
    final b = BytesBuilder();
    _u32(b, 4);
    _u32(b, shape.length);
    for (final d in shape) {
      _u32(b, d);
    }
    _u32(b, data.length);
    final raw = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    _bytes(b, raw);
    return b.takeBytes();
  }

  /// cmd=2/cmd=4 response → (status, concatenated outputs).
  static (int, Float32List) parseDataResponse(Uint8List bytes) {
    final d = ByteData.sublistView(bytes);
    final status = d.getInt32(0, Endian.little);
    final total = d.getUint32(4, Endian.little);
    return (
      status,
      Float32List.view(bytes.buffer, bytes.offsetInBytes + 8, total),
    );
  }
}

/// Runs ncnn inference in the standalone `ncnn_helper` child process.
///
/// Windows: ncnn's Vulkan device init crashes inside Flutter engine
/// processes (NVIDIA nvoglv64 access violation caused by the engine's
/// GL/D3D rendering stack — reproduced and dump-verified 2026-09). The
/// same shim runs flawlessly in a plain process, so on Windows GPU
/// inference goes through this helper; requests are framed
/// little-endian over stdin/stdout (see ncnn_helper_main.cpp).
class NcnnHelperProcess {
  NcnnHelperProcess._(this._process);

  final Process _process;
  final List<Uint8List> _chunks = <Uint8List>[];
  int _chunkOffset = 0;
  Completer<void>? _drained;
  bool _disposed = false;

  /// Shapes of the loaded model's outputs (filled by [load]).
  List<List<int>> outputShapes = const [];

  /// Spawn the helper exe. Lookup order: the directory of the running
  /// executable (bundled apps), then the pinned lib dir (test VM /
  /// NCNN_DART_LIB_DIR — the plugin dll and helper sit together).
  static Future<NcnnHelperProcess> start() async {
    if (!Platform.isWindows) {
      throw StateError('ncnn_helper is Windows-only');
    }
    final candidates = <String>[
      File(Platform.resolvedExecutable).parent.path,
      Platform.environment['NCNN_DART_LIB_DIR'] ?? '',
    ];
    for (final dir in candidates) {
      if (dir.isEmpty) continue;
      final p = '$dir${Platform.pathSeparator}ncnn_helper.exe';
      if (File(p).existsSync()) return _spawn(p);
    }
    throw StateError('ncnn_helper.exe not found (looked in '
        '${candidates.where((d) => d.isNotEmpty).join(', ')})');
  }

  static Future<NcnnHelperProcess> _spawn(String path) async {
    final process =
        await Process.start(path, const [], environment: Platform.environment);
    final helper = NcnnHelperProcess._(process);
    // Single long-lived subscription (a stream allows one listener).
    // dart:io delivers Uint8List events even though the declared element
    // type is List<int> on older SDKs.
    process.stdout.listen(
      (List<int> data) {
        helper._chunks.add(data as Uint8List);
        helper._drained?.complete();
      },
      onDone: () =>
          helper._drained?.completeError(StateError('helper closed the pipe')),
      onError: (Object e) => helper._drained?.completeError(e),
    );
    return helper;
  }

  Future<Uint8List> _readExact(int len) async {
    final result = Uint8List(len);
    var copied = 0;
    while (copied < len) {
      while (_chunks.isEmpty) {
        final done = Completer<void>();
        _drained = done;
        await done.future;
        _drained = null;
      }
      final chunk = _chunks.first;
      final take = (chunk.length - _chunkOffset).clamp(0, len - copied);
      result.setRange(copied, copied + take, chunk, _chunkOffset);
      copied += take;
      _chunkOffset += take;
      if (_chunkOffset >= chunk.length) {
        _chunks.removeAt(0);
        _chunkOffset = 0;
      }
    }
    return result;
  }

  /// Enumerate Vulkan devices (safe in the child process).
  Future<List<NcnnGpuDevice>> gpuDevices() async {
    _process.stdin.add(NcnnHelperFrames.gpuRequest());
    final head = await _readExact(8);
    final d = ByteData.sublistView(head);
    final status = d.getUint32(0, Endian.little);
    final count = d.getUint32(4, Endian.little);
    if (status != 0 || count == 0) return const [];
    final result = <NcnnGpuDevice>[];
    for (var i = 0; i < count; i++) {
      final meta = await _readExact(20);
      final md = ByteData.sublistView(meta);
      final nameLen = md.getUint32(16, Endian.little);
      final nameBytes = await _readExact(nameLen);
      result.add(NcnnGpuDevice(
        index: md.getUint32(0, Endian.little),
        type: md.getUint32(4, Endian.little),
        score: md.getUint32(8, Endian.little),
        vendorId: md.getUint32(12, Endian.little),
        name: String.fromCharCodes(nameBytes),
      ));
    }
    return result;
  }

  /// Loads a model; returns per-output shape hints (also cached in
  /// [outputShapes]). Throws [StateError] on a non-zero status.
  Future<List<List<int>>> load({
    required String paramPath,
    required String binPath,
    NcnnOptions options = const NcnnOptions(),
  }) async {
    _process.stdin.add(NcnnHelperFrames.loadRequest(
      paramPath: paramPath,
      binPath: binPath,
      options: options,
    ));
    final head = await _readExact(8);
    final hd = ByteData.sublistView(head);
    final status = hd.getInt32(0, Endian.little);
    final count = hd.getUint32(4, Endian.little);
    if (status != 0) {
      throw StateError('helper load failed with status $status');
    }
    final body = count > 0 ? await _readExact(count * 20) : Uint8List(0);
    final (_, shapes) = NcnnHelperFrames.parseLoadResponse(
      Uint8List.fromList(head + body),
    );
    outputShapes = shapes;
    return shapes;
  }

  /// Runs one prediction on a pixel buffer; returns ALL outputs
  /// concatenated (split via [outputShapes] / NcnnNet.splitOutputs).
  Future<Float32List> predict(Uint8List rgb, int width, int height) async {
    _process.stdin.add(NcnnHelperFrames.predictRequest(rgb, width, height));
    return _readDataResponse('predict');
  }

  /// Raw float tensor input (caller-preprocessed).
  Future<Float32List> extractF32(Float32List data, List<int> shape) async {
    _process.stdin.add(NcnnHelperFrames.extractF32Request(data, shape));
    return _readDataResponse('extractF32');
  }

  Future<Float32List> _readDataResponse(String op) async {
    final head = await _readExact(8);
    final hd = ByteData.sublistView(head);
    final status = hd.getInt32(0, Endian.little);
    final total = hd.getUint32(4, Endian.little);
    if (status != 0) {
      throw StateError('helper $op failed with status $status');
    }
    final body = await _readExact(total * 4);
    final (_, data) = NcnnHelperFrames.parseDataResponse(
      Uint8List.fromList(head + body),
    );
    return data;
  }

  /// Kill the helper. The helper's ncnn atexit cleanup crashes on exit
  /// (known NVIDIA driver issue), so terminate hard — the OS reclaims
  /// everything.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _process.stdin.close();
    _process.kill();
  }
}
