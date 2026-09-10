import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:ncnn/ncnn.dart';

void main() => runApp(const _App());

class _App extends StatelessWidget {
  const _App();

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'ncnn example',
        theme: ThemeData(useMaterial3: true),
        home: const _Home(),
      );
}

/// What the example needs to know about the model before loading it.
class _ModelInfo {
  const _ModelInfo({required this.classNames, required this.imgsz});

  final List<String> classNames;
  final int imgsz;
}

class _Home extends StatefulWidget {
  const _Home();

  @override
  State<_Home> createState() => _HomeState();
}

class _HomeState extends State<_Home> {
  final _param = TextEditingController();
  final _bin = TextEditingController();
  String _status = 'Enter model paths, then Load.';
  List<String> _top = const [];
  NcnnInferenceEngine? _engine;
  int _imgsz = 640;

  /// Probed once — a real app caches this for a device picker.
  late final Future<List<NcnnGpuDevice>> _gpuDevices = _probeGpuDevices();

  static Future<List<NcnnGpuDevice>> _probeGpuDevices() async {
    try {
      return NcnnRuntime.instance.gpuDevices;
    } catch (_) {
      return const <NcnnGpuDevice>[]; // no Vulkan here — CPU is fine.
    }
  }

  @override
  void dispose() {
    _engine?.dispose();
    _param.dispose();
    _bin.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _status = 'Loading…');
    final engine = NcnnInferenceEngine(
        onLog: (m) => setState(() => _status = m));
    try {
      // metadata.yaml (ultralytics export) sits next to the model files.
      final info = _readModelInfo();
      await engine.loadModel(
        paramPath: _param.text,
        binPath: _bin.text,
        // Size-locked models must be warmed up and run at the exported
        // imgsz — upstream ncnn Reshape does not validate element totals.
        options: NcnnOptions.yolo(
            warmupWidth: info.imgsz, warmupHeight: info.imgsz),
        fallbackClassNames: info.classNames,
      );
      setState(() {
        _status = 'Ready (gpu=${engine.usingGpu})';
        _engine = engine;
        _imgsz = info.imgsz;
        _top = const [];
      });
    } catch (e) {
      await engine.dispose();
      setState(() => _status = 'Load failed: $e');
    }
  }

  _ModelInfo _readModelInfo() {
    var names = const ['class0'];
    var imgsz = 640;
    final meta = File('${File(_param.text).parent.path}/metadata.yaml');
    if (meta.existsSync()) {
      final yaml = meta.readAsStringSync();
      names =
          NcnnMetadata.tryParseClassNames(yaml) ?? const ['class0'];
      final match = RegExp(r'^imgsz:\s*(\d+)').firstMatch(yaml);
      if (match != null) {
        final parsed = int.tryParse(match.group(1)!);
        if (parsed != null && parsed > 0) imgsz = parsed;
      }
    }
    return _ModelInfo(classNames: names, imgsz: imgsz);
  }

  Future<void> _run() async {
    final engine = _engine;
    if (engine == null) return;
    // Zeros frame at the exported imgsz — a real app feeds decoded,
    // letterboxed frames at exactly this size.
    final size = _imgsz;
    final rgb = Uint8List(size * size * 3);
    setState(() => _status = 'Running…');
    try {
      final logits = await engine.predict(rgb, size, size);
      final top = topK(logits, 5, limit: engine.classNames.length);
      setState(() {
        _top = top
            .map((t) =>
                '${engine.classNames[t.$1]}: ${t.$2.toStringAsFixed(3)}')
            .toList();
      });
    } catch (e) {
      setState(() => _status = 'Run failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('ncnn example')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                  controller: _param,
                  decoration: const InputDecoration(
                      labelText: 'model.ncnn.param path')),
              TextField(
                  controller: _bin,
                  decoration: const InputDecoration(
                      labelText: 'model.ncnn.bin path')),
              const SizedBox(height: 8),
              FilledButton(onPressed: _load, child: const Text('Load')),
              FilledButton(
                  onPressed: _engine != null ? _run : null,
                  child: Text('Run ($_imgsz x $_imgsz zeros)')),
              const SizedBox(height: 16),
              Text(_status),
              ..._top.map((t) => Text(t)),
              const Spacer(),
              FutureBuilder<List<NcnnGpuDevice>>(
                future: _gpuDevices,
                builder: (context, snapshot) => Text(
                  'Vulkan devices: ${snapshot.data?.map((d) => d.name).join(', ') ?? '…'}',
                ),
              ),
            ],
          ),
        ),
      );
}
