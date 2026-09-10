import 'dart:ffi';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Guard against header/bindings drift: every `hn_*` entry point the
/// Dart bindings look up must exist in the built plugin library.
///
/// Runs for real on CI after a Linux/Windows build by pointing
/// NCNN_DART_LIB_DIR at the built bundle's lib dir, e.g.:
///
/// ```sh
/// NCNN_DART_LIB_DIR=build/linux/x64/debug/bundle/lib \
///   flutter test test/symbol_coverage_test.dart
/// ```
///
/// Skipped when NCNN_DART_LIB_DIR is unset or the platform resolves the
/// plugin library by other means (Android: APK-internal name;
/// iOS/macOS: statically linked into the app binary).
void main() {
  final libDir = Platform.environment['NCNN_DART_LIB_DIR'];
  const libNames = <String, String>{
    'linux': 'libncnn_plugin.so',
    'windows': 'ncnn_plugin.dll',
  };
  final libName = libNames[Platform.operatingSystem];
  final skip = (libDir == null || libDir.isEmpty || libName == null)
      ? 'NCNN_DART_LIB_DIR unset or ${Platform.operatingSystem} '
          'resolves the plugin library by other means — skipping.'
      : false;

  test('all hn_* symbols resolve in the native library', () {
    final path = '$libDir${Platform.pathSeparator}$libName';
    if (!File(path).existsSync()) {
      fail('NCNN_DART_LIB_DIR is set but $path is missing — '
          'run a build first.');
    }
    final lib = DynamicLibrary.open(path);
    const symbols = <String>[
      'hn_create',
      'hn_load',
      'hn_output_count',
      'hn_output_shape',
      'hn_extract',
      'hn_extract_f32',
      'hn_destroy',
      'hn_gpu_count',
      'hn_gpu_devices',
    ];
    for (final symbol in symbols) {
      expect(() => lib.lookup(symbol), returnsNormally,
          reason: 'missing symbol $symbol in $libName');
    }
  }, skip: skip);
}
