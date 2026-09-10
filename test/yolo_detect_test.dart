import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/yolo/detect.dart';

void main() {
  // 2 classes -> row width 4+2 = 6. Layout [6, 4] (column-major:
  // out[c*4 + i]).
  // anchor0: box(50,50,20,20) cls0=0.9   — kept
  // anchor1: box(52,52,20,20) cls0=0.8   — IoU vs anchor0 = 324/476
  //           ~= 0.68 > 0.45, suppressed by NMS
  // anchor2: box(200,200,30,30) cls1=0.7 — kept (other class)
  // anchor3: box(300,300,10,10) cls1=0.1 — below 0.25, filtered
  final Float32List out = _colMajor([
    50, 52, 200, 300, // cx
    50, 52, 200, 300, // cy
    20, 20, 30, 10, // w
    20, 20, 30, 10, // h
    0.9, 0.8, 0.05, 0.05, // cls0
    0.05, 0.05, 0.7, 0.1, // cls1
  ], 4);
  const shape = [6, 4, 1, 1];

  test('decodes boxes, filters low conf, NMS per class', () {
    final dets = decodeYoloDetect(out, shape, numClasses: 2);
    expect(dets.length, 2);
    expect(dets[0].classIndex, 0);
    expect(dets[0].score, closeTo(0.9, 1e-6));
    expect(dets[0].x, 50);
    expect(dets[0].w, 20);
    expect(dets[1].classIndex, 1);
    expect(dets[1].x, 200);
  });

  test('row-major [n, 6] layout is auto-detected', () {
    final rowMajor = Float32List.fromList([
      50, 50, 20, 20, 0.9, 0.05, //
      200, 200, 30, 30, 0.05, 0.7,
    ]);
    final dets = decodeYoloDetect(rowMajor, const [2, 6, 1, 1], numClasses: 2);
    expect(dets.length, 2);
    expect(dets[1].classIndex, 1);
  });

  test('iou threshold 1.0 keeps overlapping same-class boxes', () {
    final dets = decodeYoloDetect(out, shape, numClasses: 2, iouThreshold: 1.0);
    expect(dets.length, 3); // anchor0 + anchor1 + anchor2
  });

  test('throws on shape that matches neither layout', () {
    expect(
      () => decodeYoloDetect(Float32List(6), const [3, 2, 1, 1], numClasses: 2),
      throwsArgumentError,
    );
  });
}

Float32List _colMajor(List<double> rows, int n) {
  final out = Float32List(rows.length);
  final ch = rows.length ~/ n;
  for (var c = 0; c < ch; c++) {
    for (var i = 0; i < n; i++) {
      out[c * n + i] = rows[c * n + i];
    }
  }
  return out;
}
