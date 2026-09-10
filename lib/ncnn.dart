/// ncnn (Vulkan) inference bindings for Dart/Flutter.
///
/// Loads ncnn models (.param/.bin) and runs image inference on CPU or
/// any Vulkan-capable GPU. See README for the ultralytics YOLO
/// conventions and platform notes.
library;

export 'src/bindings.g.dart'
    show HnGpuDevice, hnMaxGpu, hnNameMax, hnOk, hnErrCapacity;
export 'src/options.dart';
export 'src/runtime.dart';
export 'src/net.dart';
