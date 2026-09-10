import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/yolo/classify.dart';

void main() {
  test('argmax finds the highest score', () {
    expect(argmax(Float32List.fromList([0.1, 0.9, 0.3])), 1);
    expect(argmax(Float32List.fromList([0.5])), 0);
  });

  test('topK returns descending pairs', () {
    final top = topK(Float32List.fromList([0.1, 0.9, 0.3, 0.7]), 2);
    expect(top.length, 2);
    // Float32 storage: 0.9 is 0.89999997615... exactly, so compare with
    // tolerance instead of record equality.
    expect(top[0].$1, 1);
    expect(top[0].$2, closeTo(0.9, 1e-6));
    expect(top[1].$1, 3);
    expect(top[1].$2, closeTo(0.7, 1e-6));
  });

  test('topK respects limit (metadata class count vs logits)', () {
    final top = topK(Float32List.fromList([0.1, 0.9, 0.3]), 5, limit: 2);
    expect(top.length, 2);
    expect(top[0].$1, 1);
  });

  test('softmax sums to 1 and is stable for large logits', () {
    final p = softmax(Float32List.fromList([1000.0, 1000.0, 0.0]));
    expect(p.length, 3);
    expect(p[0], closeTo(0.5, 1e-6));
    expect(p.reduce((a, b) => a + b), closeTo(1.0, 1e-6));
  });
}
