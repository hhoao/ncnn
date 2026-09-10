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

  group('tryParseImgSize', () {
    // Real ultralytics export writes imgsz as a YAML list (verified
    // against huji-algorithm production metadata.yaml exports).
    test('parses the real ultralytics list form', () {
      const yaml = 'stride: 1\ntask: classify\nbatch: 1\n'
          'imgsz:\n- 640\n- 640\nnames:\n  0: fire_ball\n';
      expect(NcnnMetadata.tryParseImgSize(yaml), 640);
    });

    test('parses the scalar form', () {
      const yaml = 'task: classify\nimgsz: 640\nbatch: 1\n';
      expect(NcnnMetadata.tryParseImgSize(yaml), 640);
    });

    test('parses a non-640 list form', () {
      const yaml = 'imgsz:\n- 224\n- 224\n';
      expect(NcnnMetadata.tryParseImgSize(yaml), 224);
    });

    test('returns null when imgsz is absent', () {
      const yaml = 'task: classify\nnames:\n  0: fireball\n';
      expect(NcnnMetadata.tryParseImgSize(yaml), isNull);
    });

    test('returns null for non-numeric imgsz', () {
      expect(NcnnMetadata.tryParseImgSize('imgsz: auto\n'), isNull);
      expect(NcnnMetadata.tryParseImgSize('imgsz:\n- auto\n'), isNull);
    });
  });
}
