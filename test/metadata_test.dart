import 'package:flutter_test/flutter_test.dart';
import 'package:ncnn/src/metadata.dart';

void main() {
  test('parses plain ultralytics metadata.yaml names', () {
    const yaml = 'task: classify\nnames:\n  0: fireball\n  1: pickball\n';
    expect(NcnnMetadata.tryParseClassNames(yaml), ['fireball', 'pickball']);
  });

  test('parses quoted names and names with spaces', () {
    const yaml = 'names:\n  0: "fire ball"\n  1: \'pick ball\'\n  2: plain\n';
    expect(NcnnMetadata.tryParseClassNames(yaml),
        ['fire ball', 'pick ball', 'plain']);
  });

  test('stops at the end of the names block', () {
    const yaml = 'names:\n  0: a\n  1: b\nother_key: 1\n';
    expect(NcnnMetadata.tryParseClassNames(yaml), ['a', 'b']);
  });

  test('returns null when there is no names block', () {
    expect(NcnnMetadata.tryParseClassNames('task: classify\n'), isNull);
  });

  test('returns null for empty names block', () {
    expect(NcnnMetadata.tryParseClassNames('names:\n'), isNull);
  });
}
