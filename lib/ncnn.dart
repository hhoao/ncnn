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
export 'src/inference_engine.dart';
export 'src/metadata.dart';
export 'src/helper_process.dart' show NcnnHelperProcess;
export 'src/yolo/classify.dart';
export 'src/yolo/detect.dart';
