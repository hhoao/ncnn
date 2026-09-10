import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/net.dart';

void main() {
  test('splitOutputs slices the flat buffer by shape products', () {
    final flat = Float32List.fromList(
      List<double>.generate(10, (i) => i.toDouble()),
    );
    final parts = NcnnNet.splitOutputs(flat, const [
      [3, 1, 1, 1], // 3 elems
      [2, 3, 1, 1], // 6 elems
    ]);
    expect(parts.length, 2);
    expect(parts[0], Float32List.fromList([0, 1, 2]));
    expect(parts[1], Float32List.fromList([3, 4, 5, 6, 7, 8]));
  });

  test('splitOutputs with single output returns one view', () {
    final flat = Float32List.fromList([1.5, -2.5]);
    final parts = NcnnNet.splitOutputs(flat, const [
      [2, 1, 1, 1],
    ]);
    expect(parts.length, 1);
    expect(parts[0].length, 2);
    expect(parts[0][1], -2.5);
  });

  test('splitOutputs throws when shapes exceed the buffer', () {
    final flat = Float32List(2);
    expect(
      () => NcnnNet.splitOutputs(flat, const [
        [4, 1, 1, 1],
      ]),
      throwsArgumentError,
    );
  });
}
