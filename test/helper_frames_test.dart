import 'dart:convert' show utf8;
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/helper_process.dart';
import 'package:ncnn/src/options.dart';

void main() {
  test('loadRequest encodes paths + full options frame', () {
    final bytes = NcnnHelperFrames.loadRequest(
      paramPath: '/a.param',
      binPath: '/b.bin',
      options: const NcnnOptions.yolo(
        useVulkan: true,
        deviceIndex: 1,
        warmupWidth: 640,
        warmupHeight: 640,
      ),
    );
    final d = ByteData.sublistView(bytes);
    var off = 0;
    expect(d.getUint32(off, Endian.little), 1); // cmd
    off += 4;
    final paramLen = d.getUint32(off, Endian.little);
    off += 4;
    expect(paramLen, 8);
    expect(bytes.sublist(off, off + paramLen), utf8.encode('/a.param'));
    off += paramLen;
    final binLen = d.getUint32(off, Endian.little);
    off += 4;
    expect(bytes.sublist(off, off + binLen), utf8.encode('/b.bin'));
    off += binLen;
    // opts fixed part = 3×i32 + u32 blobLen + f32 mean[3] + f32 norm[3] +
    // i32 warmupW + i32 warmupH = 48 bytes, then the blob bytes.
    const blobLen = 3; // 'in0'
    expect(d.getUint32(off, Endian.little), 48 + blobLen); // optsLen
    off += 4;
    expect(d.getInt32(off, Endian.little), 1); // useVulkan
    expect(d.getInt32(off + 4, Endian.little), 1); // deviceIndex
    expect(d.getInt32(off + 8, Endian.little), 0); // pixelFormat = rgb
    expect(d.getUint32(off + 12, Endian.little), blobLen);
    expect(bytes.sublist(off + 16, off + 16 + blobLen), utf8.encode('in0'));
    final f = off + 16 + blobLen; // mean[3] | norm[3] | warmup pair
    for (var i = 0; i < 3; i++) {
      expect(d.getFloat32(f + i * 4, Endian.little), 0); // mean
      expect(
        d.getFloat32(f + 12 + i * 4, Endian.little),
        closeTo(1 / 255, 1e-9), // norm
      );
    }
    expect(d.getInt32(f + 24, Endian.little), 640); // warmup_w
    expect(d.getInt32(f + 28, Endian.little), 640); // warmup_h
    expect(bytes.length, f + 32);
  });

  test('loadRequest encodes non-ASCII paths as UTF-8 bytes', () {
    final bytes = NcnnHelperFrames.loadRequest(
      paramPath: '/模型/a.param',
      binPath: '/模型/b.bin',
      options: const NcnnOptions(),
    );
    final d = ByteData.sublistView(bytes);
    expect(d.getUint32(0, Endian.little), 1); // cmd
    final paramLen = d.getUint32(4, Endian.little);
    // '/' (1) + 模型 (2×3 UTF-8 bytes) + '/a.param' (8) = 15 bytes —
    // NOT the UTF-16 code-unit count (11).
    expect(paramLen, 15);
    // 模型 → E6 A8 A1 E5 9E 8B, followed by the ASCII suffix.
    expect(bytes.sublist(8, 8 + paramLen), [
      0x2F, 0xE6, 0xA8, 0xA1, 0xE5, 0x9E, 0x8B, // /模型
      0x2F, 0x61, 0x2E, 0x70, 0x61, 0x72, 0x61, 0x6D, // /a.param
    ]);
    final binLen = d.getUint32(8 + paramLen, Endian.little);
    // '/' (1) + 模型 (6) + '/b.bin' (6) = 13 bytes (code units: 9).
    expect(binLen, 13);
    expect(
      bytes.sublist(8 + paramLen + 4, 8 + paramLen + 4 + binLen),
      utf8.encode('/模型/b.bin'),
    );
  });

  test('parseLoadResponse reads status + shape rows', () {
    final b = BytesBuilder();
    void u32(int v) =>
        b.add(Uint8List(4)..buffer.asByteData().setUint32(0, v, Endian.little));
    u32(0); // status
    u32(2); // outCount
    for (final s in [
      [2, 84, 8400, 1, 1],
      [1, 4, 1, 1, 1],
    ]) {
      u32(s[0]);
      u32(s[1]);
      u32(s[2]);
      u32(s[3]);
      u32(s[4]);
    }
    final (status, shapes) = NcnnHelperFrames.parseLoadResponse(b.takeBytes());
    expect(status, 0);
    expect(shapes, [
      [84, 8400, 1, 1],
      [4, 1, 1, 1],
    ]);
  });

  test('parseDataResponse reads status + floats', () {
    final b = BytesBuilder();
    b.add(Uint8List(4)..buffer.asByteData().setUint32(0, 0, Endian.little));
    b.add(Uint8List(4)..buffer.asByteData().setUint32(0, 2, Endian.little));
    final f = ByteData(8)
      ..setFloat32(0, 1.5, Endian.little)
      ..setFloat32(4, -2.5, Endian.little);
    b.add(f.buffer.asUint8List());
    final (status, data) = NcnnHelperFrames.parseDataResponse(b.takeBytes());
    expect(status, 0);
    expect(data, Float32List.fromList([1.5, -2.5]));
  });

  test('predictRequest encodes cmd/w/h/pixels', () {
    final rgb = Uint8List.fromList([1, 2, 3, 4, 5, 6]);
    final bytes = NcnnHelperFrames.predictRequest(rgb, 2, 1);
    final d = ByteData.sublistView(bytes);
    expect(d.getUint32(0, Endian.little), 2); // cmd
    expect(d.getUint32(4, Endian.little), 2); // w
    expect(d.getUint32(8, Endian.little), 1); // h
    expect(d.getUint32(12, Endian.little), 6); // frameLen
    expect(bytes.sublist(16), rgb);
  });

  test('extractF32Request encodes shape + float payload', () {
    final data = Float32List.fromList([1.5, -2.5]);
    final bytes = NcnnHelperFrames.extractF32Request(data, const [2, 2]);
    final d = ByteData.sublistView(bytes);
    expect(d.getUint32(0, Endian.little), 4); // cmd
    expect(d.getUint32(4, Endian.little), 2); // dims
    expect(d.getUint32(8, Endian.little), 2); // shape[0]
    expect(d.getUint32(12, Endian.little), 2); // shape[1]
    expect(d.getUint32(16, Endian.little), 2); // dataLen (float count)
    expect(d.getFloat32(20, Endian.little), 1.5);
    expect(d.getFloat32(24, Endian.little), -2.5);
    expect(bytes.length, 28);
  });

  test('gpuRequest is a bare cmd=3 frame', () {
    expect(NcnnHelperFrames.gpuRequest(), [3, 0, 0, 0]);
  });

  test('start is Windows-only', () async {
    // The helper exists only for Windows; other platforms use the
    // in-process FFI plugin. Trivially green on Windows itself.
    if (Platform.isWindows) return;
    await expectLater(NcnnHelperProcess.start(), throwsStateError);
  });
}
