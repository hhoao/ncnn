## 0.1.0

Initial release.

- ncnn (Vulkan) inference bindings for Dart/Flutter via a thin C shim
  and dart:ffi — CPU + any Vulkan GPU (NVIDIA / AMD / Intel / Apple via
  MoltenVK / Android), automatic CPU fallback.
- Five-platform FFI plugin (Android / iOS / Linux / macOS / Windows);
  official ncnn prebuilts downloaded at build time.
- `NcnnNet`: shape-driven multi-output inference with capacity retry,
  configurable preprocessing (mean/norm), pixel format and input blob
  (ultralytics YOLO conventions as defaults).
- `hn_extract_f32`: raw float tensor input for non-image models.
- `NcnnInferenceEngine`: backend selection, best-device heuristic,
  GPU→CPU fallback, injectable backends for testing.
- Windows `ncnn_helper` child process: full GPU acceleration outside
  the Flutter engine process (ncnn's Vulkan init crashes inside it —
  dump-verified; see README).
- YOLO post-processing utilities: classify (argmax/topK/softmax),
  detect (grid decode + per-class NMS).
