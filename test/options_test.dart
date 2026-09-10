import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/bindings.g.dart' as native;
import 'package:ncnn/src/options.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('HnOptions struct size is accepted by C (>= V1 size)', () {
    // The C side rejects struct_size outside
    // [HN_OPTIONS_V1_SIZE, sizeof(hn_options_t)] — compiled-together
    // means equality here.
    expect(sizeOf<native.HnOptions>(),
        greaterThanOrEqualTo(4 + 4 + 4 + 12 + 12 + 4));
  });

  test('NcnnOptions defaults match ultralytics convention', () {
    const o = NcnnOptions();
    expect(o.useVulkan, isFalse);
    expect(o.deviceIndex, -1);
    expect(o.mean, const [0, 0, 0]);
    expect(o.norm, const [1, 1, 1]);
    expect(o.pixelFormat, NcnnPixelFormat.rgb);
    expect(o.inputBlob, isNull);
    expect(o.warmupWidth, 0);
    expect(o.warmupHeight, 0);
  });

  test('NcnnOptions.yolo uses x/255 normalization', () {
    const o = NcnnOptions.yolo();
    expect(o.norm, everyElement(closeTo(1 / 255, 1e-9)));
    expect(o.mean, const [0, 0, 0]);
  });

  test('toNative roundtrips fields and frees cleanly', () {
    const o = NcnnOptions.yolo(useVulkan: true, deviceIndex: 2);
    final ptr = o.toNative();
    try {
      final s = ptr.ref;
      expect(s.structSize, sizeOf<native.HnOptions>());
      expect(s.useVulkan, 1);
      expect(s.deviceIndex, 2);
      expect(s.norm[0], closeTo(1 / 255, 1e-9));
      expect(s.inputBlob.cast<Utf8>().toDartString(), 'in0');
    } finally {
      NcnnOptions.freeNative(ptr);
    }
  });

  test('warmup fields roundtrip through toNative', () {
    const o = NcnnOptions(warmupWidth: 640, warmupHeight: 640);
    final ptr = o.toNative();
    try {
      expect(ptr.ref.warmupW, 640);
      expect(ptr.ref.warmupH, 640);
    } finally {
      NcnnOptions.freeNative(ptr);
    }
  });
}
