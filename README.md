# ncnn

[ncnn](https://github.com/Tencent/ncnn) inference bindings for Dart and
Flutter — a thin C shim over `ncnn::Net` exposed via `dart:ffi`.

Runs image inference on CPU or **any Vulkan-capable GPU** (NVIDIA /
AMD / Intel / Apple via MoltenVK / Android), with automatic CPU
fallback. Five platforms: Android, iOS, Linux, macOS, Windows.

## Quick start

```dart
import 'package:ncnn/ncnn.dart';

final engine = NcnnInferenceEngine();
await engine.loadModel(
  paramPath: 'model.ncnn.param',
  binPath: 'model.ncnn.bin',
  fallbackClassNames: ['fireball', 'pickball'], // or parse metadata.yaml
);

// RGB24 bytes (w*h*3), e.g. from your own decode/letterbox step.
final logits = await engine.predict(rgb, width, height);
final top = topK(logits, 5);
```

Defaults follow the **ultralytics ncnn export** convention
(`YOLO('best.pt').export(format='ncnn')`): input blob `in0`,
preprocessing `x/255` (no mean), RGB input. Override anything:

```dart
final net = await NcnnNet.load(
  paramPath: '...param', binPath: '...bin',
  options: NcnnOptions(
    useVulkan: true, deviceIndex: 0,
    mean: [0.485, 0.456, 0.406], norm: [0.229, 0.224, 0.225],
    inputBlob: 'data',
    warmupWidth: 640, warmupHeight: 640, // exported imgsz (see below)
  ),
);
final outputs = net.extract(rgb, w, h); // all output blobs + shapes
```

**Feed the model its exported input size.** Ultralytics exports are
size-locked (`imgsz` in `metadata.yaml`): upstream ncnn `Reshape` does
not validate element totals, so running a size-locked model at a
different size corrupts memory instead of failing. Pass the exported
size as `warmupWidth`/`warmupHeight` and extract at exactly that size.

Non-image models: `net.extractF32(data, [w, h, d, c])` takes a raw
float tensor (caller-preprocessed).

## YOLO utilities

```dart
final outputs = await engine.extract(rgb, w, h);
final dets = decodeYoloDetect(
  outputs.first.data, outputs.first.shape, numClasses: 80);
```

Detect outputs of ultralytics exports are already decoded
(DFL/box decode live in the graph); this performs per-class NMS.
Classify: `argmax` / `topK` / `softmax`. `NcnnMetadata.tryParseClassNames`
parses class names out of ultralytics `metadata.yaml`.

## Platform notes

| Platform | GPU | Notes |
|---|---|---|
| Linux | Vulkan | needs host Vulkan loader (`libvulkan`) — every distro ships it |
| Windows | Vulkan via `ncnn_helper` | see below |
| macOS | Vulkan via MoltenVK | MoltenVK vendored automatically by the podspec |
| iOS | Vulkan via MoltenVK | same |
| Android | Vulkan | falls back to CPU on non-Vulkan devices |

### Windows: inference in a helper process

ncnn's Vulkan device init crashes inside Flutter engine processes
(NVIDIA `nvoglv64` access violation, reproduced and dump-verified; the
engine's GL/D3D stack conflicts with the driver's Vulkan path). This
package runs GPU inference in a tiny `ncnn_helper.exe` child process
over stdin/stdout — full GPU acceleration, transparent to your code.
Constructing `NcnnInferenceEngine(forceInProcess: true)` runs
in-process on Windows too, but that forces CPU: the in-process GPU
probe itself is the crash path.

Known limitation: model paths are sent to the helper as UTF-8 bytes
and passed verbatim to ncnn (`fopen`). Non-ASCII model paths on
Windows therefore depend on the helper process's active code page
being UTF-8 — a known upstream ncnn limitation.

### Thread/isolate safety

`NcnnNet.extract` is a synchronous FFI call (tens of ms on CPU). Never
call it on the UI isolate — use `NcnnNet.extractInIsolate`, or run the
engine in a worker isolate. One extract at a time per `NcnnNet`.

## API stability

C surface versioning: `hn_options_t` carries `struct_size`; fields are
only ever appended, so bindings compiled against older structs keep
working. Dart API follows semantic versioning.

## Development

- `src/ncnn_api.{h,cpp}` — the C shim (`extern "C"` over `ncnn::Net`).
  `ios/src/` and `macos/src/` hold committed copies (CocoaPods globs
  cannot escape the podspec dir; `dart pub publish` dereferences
  symlinks) — run `tool/sync_apple_sources.sh` after editing the shim.
- `lib/src/bindings.g.dart` — hand-maintained `dart:ffi` bindings
  (no ffigen step); `test/symbol_coverage_test.dart` asserts every
  bound symbol exists in the built plugin library.
- The `example/` app loads a `.param`/`.bin` pair from disk and runs a
  single classify pass.

## License

BSD 3-Clause. ncnn (BSD 3) and MoltenVK (Apache 2.0) are downloaded at
build time — see THIRD_PARTY_NOTICES.md.
