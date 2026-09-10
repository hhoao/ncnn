import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show visibleForTesting;

import 'helper_process.dart';
import 'options.dart';
import 'runtime.dart';

typedef NcnnNetLoader = Future<NcnnNet> Function({
  required String paramPath,
  required String binPath,
  required NcnnOptions options,
});

typedef NcnnHelperStarter = Future<NcnnHelperProcess> Function();

/// High-level inference engine: picks the backend (Windows: ncnn_helper
/// child process for GPU; other platforms: in-process FFI), selects the
/// best Vulkan device and falls back to CPU on any failure.
///
/// [NcnnHelperProcess] is strict request/response (single-flight): the
/// engine does not queue, so concurrent [extract]/[predict] calls must be
/// serialized by the caller (the app-level predictor does this).
///
/// [NcnnNet.extract] is a synchronous FFI call (tens of ms on CPU) — do
/// not call it on the UI isolate; use [NcnnNet.extractInIsolate] or run
/// the engine inside a worker isolate.
class NcnnInferenceEngine {
  NcnnInferenceEngine({
    NcnnNetLoader? netLoader,
    NcnnHelperStarter? helperStarter,
    Future<List<NcnnGpuDevice>> Function()? gpuDeviceProbe,
    bool forceInProcess = false,
    void Function(String message)? onLog,
  })  : _netLoader = netLoader ?? _defaultNetLoader,
        _helperStarter = helperStarter ?? NcnnHelperProcess.start,
        _gpuDeviceProbe =
            gpuDeviceProbe ?? (() async => NcnnRuntime.instance.gpuDevices),
        _forceInProcess = forceInProcess,
        _log = onLog ?? ((_) {});

  /// Test hook overriding the platform check that routes loads through
  /// the Windows-only helper process — lets non-Windows CI exercise
  /// [_loadViaHelper]. Production leaves it null (real [Platform]).
  @visibleForTesting
  static bool? debugIsWindowsOverride;

  final NcnnNetLoader _netLoader;
  final NcnnHelperStarter _helperStarter;
  final Future<List<NcnnGpuDevice>> Function() _gpuDeviceProbe;
  final bool _forceInProcess;
  final void Function(String) _log;

  NcnnNet? _net;
  NcnnHelperProcess? _helper;
  bool _loaded = false;
  List<String>? _classNames;
  bool _usingGpu = false;

  static Future<NcnnNet> _defaultNetLoader({
    required String paramPath,
    required String binPath,
    required NcnnOptions options,
  }) {
    return NcnnNet.load(
        paramPath: paramPath, binPath: binPath, options: options);
  }

  /// Class names (caller-provided; the engine itself never parses
  /// model metadata).
  List<String> get classNames {
    final names = _classNames;
    if (names == null || names.isEmpty) {
      throw StateError('Class names not set. Call loadModel() first.');
    }
    return names;
  }

  bool get isLoaded => _loaded;
  bool get usingGpu => _usingGpu;

  /// Discrete first, then ncnn rough_score (higher = faster), then index.
  ///
  /// Type-3 (software Vulkan, e.g. llvmpipe) devices are excluded before
  /// ranking: their rough_score can outscore a real discrete GPU (observed
  /// with llvmpipe beating an Intel Arc), which would select a software
  /// device that runs slower than plain CPU. If only type-3 devices exist,
  /// returns null so callers fall back to CPU.
  static NcnnGpuDevice? bestDevice(List<NcnnGpuDevice> devices) {
    final candidates = devices.where((d) => d.type != 3).toList();
    if (candidates.isEmpty) return null;
    final sorted = candidates
      ..sort((a, b) {
        final aDiscrete = a.type == 0 ? 1 : 0;
        final bDiscrete = b.type == 0 ? 1 : 0;
        final cmp = bDiscrete.compareTo(aDiscrete);
        if (cmp != 0) return cmp;
        final scoreCmp = b.score.compareTo(a.score);
        if (scoreCmp != 0) return scoreCmp;
        return b.index.compareTo(a.index);
      });
    return sorted.first;
  }

  Future<void> loadModel({
    required String paramPath,
    required String binPath,
    NcnnOptions options = const NcnnOptions.yolo(),
    List<String>? fallbackClassNames,
  }) async {
    final bool isWindows = debugIsWindowsOverride ?? Platform.isWindows;
    if (isWindows && !_forceInProcess) {
      await _loadViaHelper(
          paramPath: paramPath, binPath: binPath, options: options);
    } else {
      await _loadInProcess(
          paramPath: paramPath,
          binPath: binPath,
          options: options,
          isWindows: isWindows);
    }
    _loaded = true;
    _classNames = fallbackClassNames;
    _log('ncnn session ready '
        '(${_classNames?.length ?? 0} classes, gpu=$_usingGpu)');
  }

  Future<void> _loadInProcess({
    required String paramPath,
    required String binPath,
    required NcnnOptions options,
    required bool isWindows,
  }) async {
    // On Windows the in-process GPU probe itself is the crash path
    // (NVIDIA nvoglv64 access violation inside Flutter engine processes —
    // see helper_process.dart). forceInProcess on Windows therefore means
    // forced CPU: never touch in-process Vulkan at all.
    final NcnnGpuDevice? device;
    if (_forceInProcess && isWindows) {
      device = null;
      _log('ncnn using CPU (in-process forced)');
    } else {
      device = await _bestProbeDevice();
    }
    _usingGpu = device != null;
    if (device != null) {
      _log('ncnn using Vulkan device ${device.index}: ${device.name} '
          '(type=${device.type}, score=${device.score})');
    } else if (!(_forceInProcess && isWindows)) {
      _log('ncnn using CPU (no Vulkan device)');
    }
    try {
      _net = await _netLoader(
        paramPath: paramPath,
        binPath: binPath,
        options: _withDevice(options, device?.index ?? -1),
      );
    } catch (e) {
      if (device == null) rethrow;
      _log('ncnn Vulkan load failed, falling back to CPU: $e');
      _usingGpu = false;
      _net = await _netLoader(
        paramPath: paramPath,
        binPath: binPath,
        options: _withDevice(options, -1),
      );
    }
  }

  Future<void> _loadViaHelper({
    required String paramPath,
    required String binPath,
    required NcnnOptions options,
  }) async {
    try {
      final helper = await _helperStarter();
      _helper = helper;
      var gpuIndex = -1;
      final devices = await helper.gpuDevices();
      final best = bestDevice(devices);
      if (best != null) {
        gpuIndex = best.index;
        _usingGpu = true;
        _log('ncnn (helper) using Vulkan device ${best.index}: '
            '${best.name} (score=${best.score})');
      } else {
        _log('ncnn (helper) using CPU (no Vulkan device)');
      }
      await helper.load(
          paramPath: paramPath,
          binPath: binPath,
          options: _withDevice(options, gpuIndex));
    } catch (e) {
      _log('ncnn helper path failed, falling back to in-process CPU: $e');
      _helper?.dispose();
      _helper = null;
      _usingGpu = false;
      _net = await _netLoader(
        paramPath: paramPath,
        binPath: binPath,
        options: _withDevice(options, -1),
      );
    }
  }

  NcnnOptions _withDevice(NcnnOptions o, int deviceIndex) => NcnnOptions(
        useVulkan: deviceIndex >= 0,
        deviceIndex: deviceIndex,
        mean: o.mean,
        norm: o.norm,
        pixelFormat: o.pixelFormat,
        inputBlob: o.inputBlob,
        warmupWidth: o.warmupWidth,
        warmupHeight: o.warmupHeight,
      );

  Future<NcnnGpuDevice?> _bestProbeDevice() async {
    try {
      return bestDevice(await _gpuDeviceProbe());
    } catch (e) {
      _log('gpu probe failed, using CPU: $e');
      return null;
    }
  }

  /// All output blobs of one run (pixel input).
  Future<List<NcnnOutput>> extract(
      Uint8List pixels, int width, int height) async {
    final helper = _helper;
    if (helper != null) {
      final flat = await helper.predict(pixels, width, height);
      return NcnnNet.splitOutputs(flat, helper.outputShapes)
          .asMap()
          .entries
          .map((e) =>
              NcnnOutput(shape: helper.outputShapes[e.key], data: e.value))
          .toList();
    }
    final net = _net;
    if (net == null) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }
    return net.extract(pixels, width, height);
  }

  /// Convenience: first output (classify models).
  Future<Float32List> predict(Uint8List pixels, int width, int height) async {
    final outputs = await extract(pixels, width, height);
    if (outputs.isEmpty) throw StateError('no outputs produced');
    return outputs.first.data;
  }

  Future<void> dispose() async {
    _helper?.dispose();
    _helper = null;
    _net?.dispose();
    _net = null;
    _loaded = false;
    _classNames = null;
    _usingGpu = false;
  }
}
