import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/ncnn.dart';

NcnnGpuDevice _dev(int type, int score, int index) => NcnnGpuDevice(
    index: index, type: type, score: score, vendorId: 0, name: 'd$index');

class _FakeNet implements NcnnNet {
  _FakeNet(this.options);
  final NcnnOptions options;
  bool disposed = false;

  @override
  List<NcnnOutput> extract(Uint8List pixels, int width, int height) => [
        NcnnOutput(
            shape: const [2, 1, 1, 1], data: Float32List.fromList([1, 2]))
      ];

  @override
  dynamic noSuchMethod(Invocation invocation) {
    if (invocation.memberName == #dispose) {
      disposed = true;
      return null;
    }
    throw UnimplementedError('${invocation.memberName}');
  }
}

class _FakeHelper implements NcnnHelperProcess {
  final List<NcnnGpuDevice> devices;
  final bool failLoad;

  _FakeHelper(this.devices, {this.failLoad = false});

  bool disposed = false;
  final List<NcnnOptions> loads = [];

  @override
  List<List<int>> outputShapes = const [];

  @override
  Future<List<NcnnGpuDevice>> gpuDevices() async => devices;

  @override
  Future<List<List<int>>> load({
    required String paramPath,
    required String binPath,
    NcnnOptions options = const NcnnOptions(),
  }) async {
    loads.add(options);
    if (failLoad) throw StateError('helper load boom');
    outputShapes = const [
      [2, 1, 1, 1],
      [1, 1, 1, 1],
    ];
    return outputShapes;
  }

  @override
  Future<Float32List> predict(Uint8List rgb, int width, int height) async =>
      Float32List.fromList([3, 4, 5]);

  @override
  void dispose() => disposed = true;

  @override
  dynamic noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('${invocation.memberName}');
}

void main() {
  tearDown(() => NcnnInferenceEngine.debugIsWindowsOverride = null);

  test('bestDevice prefers discrete, then score', () {
    final best = NcnnInferenceEngine.bestDevice([
      _dev(1, 90, 0), // integrated, high score
      _dev(0, 50, 1), // discrete, lower score
      _dev(0, 80, 2), // discrete, higher score
    ]);
    expect(best!.index, 2);
    expect(NcnnInferenceEngine.bestDevice(const []), isNull);
  });

  test('non-Windows: GPU load failure falls back to CPU', () async {
    final calls = <NcnnOptions>[];
    final engine = NcnnInferenceEngine(
      gpuDeviceProbe: () async => [_dev(0, 80, 1)],
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        calls.add(options);
        if (options.useVulkan) {
          throw StateError('vulkan boom');
        }
        return _FakeNet(options);
      },
    );
    await engine.loadModel(
      paramPath: '/a.param',
      binPath: '/b.bin',
      options: const NcnnOptions.yolo(warmupWidth: 640, warmupHeight: 640),
      fallbackClassNames: const ['x', 'y'],
    );
    expect(calls.length, 2);
    expect(calls[0].useVulkan, isTrue);
    expect(calls[0].deviceIndex, 1);
    expect(calls[1].useVulkan, isFalse);
    expect(calls[1].deviceIndex, -1);
    // _withDevice must preserve warmup size hints across the rebuild.
    expect(calls[1].warmupWidth, 640);
    expect(calls[1].warmupHeight, 640);
    expect(engine.usingGpu, isFalse);
    expect(engine.classNames, ['x', 'y']);
    final logits = await engine.predict(Uint8List(12), 2, 2);
    expect(logits, [1, 2]);
  });

  test('non-Windows: no GPU device -> direct CPU load', () async {
    final calls = <NcnnOptions>[];
    final engine = NcnnInferenceEngine(
      gpuDeviceProbe: () async => const [],
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        calls.add(options);
        return _FakeNet(options);
      },
    );
    await engine.loadModel(paramPath: '/a.param', binPath: '/b.bin');
    expect(calls.length, 1);
    expect(calls[0].useVulkan, isFalse);
    expect(engine.usingGpu, isFalse);
  });

  test('Windows: helper path loads via helper and splits outputs', () async {
    NcnnInferenceEngine.debugIsWindowsOverride = true;
    final helper = _FakeHelper([_dev(1, 90, 0), _dev(0, 50, 1)]);
    var netLoaded = 0;
    final engine = NcnnInferenceEngine(
      helperStarter: () async => helper,
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        netLoaded++;
        return _FakeNet(options);
      },
    );
    await engine.loadModel(
      paramPath: '/a.param',
      binPath: '/b.bin',
      options: const NcnnOptions.yolo(warmupWidth: 640, warmupHeight: 640),
      fallbackClassNames: const ['x', 'y'],
    );
    expect(netLoaded, 0, reason: 'helper path must not load an in-process net');
    expect(helper.loads.length, 1);
    expect(helper.loads[0].useVulkan, isTrue);
    // Best device is the discrete one (index 1), not the higher-score iGPU.
    expect(helper.loads[0].deviceIndex, 1);
    expect(helper.loads[0].warmupWidth, 640);
    expect(helper.loads[0].warmupHeight, 640);
    expect(engine.usingGpu, isTrue);

    final outputs = await engine.extract(Uint8List(12), 2, 2);
    expect(outputs.length, 2);
    expect(outputs[0].shape, [2, 1, 1, 1]);
    expect(outputs[0].data, [3, 4]);
    expect(outputs[1].shape, [1, 1, 1, 1]);
    expect(outputs[1].data, [5]);
    expect(await engine.predict(Uint8List(12), 2, 2), [3, 4]);

    await engine.dispose();
    expect(helper.disposed, isTrue);
    expect(engine.isLoaded, isFalse);
  });

  test('Windows: helper start failure falls back to in-process CPU', () async {
    NcnnInferenceEngine.debugIsWindowsOverride = true;
    final calls = <NcnnOptions>[];
    final engine = NcnnInferenceEngine(
      helperStarter: () async => throw StateError('helper spawn boom'),
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        calls.add(options);
        return _FakeNet(options);
      },
    );
    await engine.loadModel(paramPath: '/a.param', binPath: '/b.bin');
    expect(calls.length, 1);
    expect(calls[0].useVulkan, isFalse);
    expect(calls[0].deviceIndex, -1);
    expect(engine.usingGpu, isFalse);
    expect(await engine.predict(Uint8List(12), 2, 2), [1, 2]);
  });

  test('Windows: helper load failure falls back to in-process CPU', () async {
    NcnnInferenceEngine.debugIsWindowsOverride = true;
    final helper = _FakeHelper([_dev(0, 80, 1)], failLoad: true);
    final calls = <NcnnOptions>[];
    final engine = NcnnInferenceEngine(
      helperStarter: () async => helper,
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        calls.add(options);
        return _FakeNet(options);
      },
    );
    await engine.loadModel(paramPath: '/a.param', binPath: '/b.bin');
    expect(helper.loads.length, 1);
    expect(calls.length, 1);
    expect(calls[0].useVulkan, isFalse);
    expect(engine.usingGpu, isFalse);
    expect(await engine.predict(Uint8List(12), 2, 2), [1, 2]);
    // The failed helper must be disposed and detached.
    expect(helper.disposed, isTrue);
    final outputs = await engine.extract(Uint8List(12), 2, 2);
    expect(outputs.single.data, [1, 2]);
  });

  test('dispose cleans the in-process net', () async {
    _FakeNet? net;
    final engine = NcnnInferenceEngine(
      gpuDeviceProbe: () async => const [],
      netLoader: (
          {required paramPath, required binPath, required options}) async {
        net = _FakeNet(options);
        return net!;
      },
    );
    await engine.loadModel(paramPath: '/a.param', binPath: '/b.bin');
    await engine.dispose();
    expect(net!.disposed, isTrue);
    expect(engine.isLoaded, isFalse);
    expect(
      () => engine.classNames,
      throwsStateError,
      reason: 'classNames must be cleared after dispose',
    );
  });
}
