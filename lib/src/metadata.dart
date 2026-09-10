import 'dart:convert';

/// Parses YOLO class names out of ultralytics ncnn `metadata.yaml`.
///
/// Layout:
/// ```yaml
/// names:
///   0: fireball
///   1: "fire ball"
/// ```
class NcnnMetadata {
  NcnnMetadata._();

  static List<String>? tryParseClassNames(String yaml) {
    final lines = const LineSplitter().convert(yaml);
    var inNames = false;
    final entries = <int, String>{};
    for (final line in lines) {
      if (!inNames) {
        if (line.trim() == 'names:') inNames = true;
        continue;
      }
      final match = RegExp(r'^\s+(\d+):\s*(.*?)\s*$').firstMatch(line);
      if (match == null) break; // names block ended
      var name = match.group(2)!;
      if (name.length >= 2 &&
          ((name.startsWith('"') && name.endsWith('"')) ||
              (name.startsWith("'") && name.endsWith("'")))) {
        name = name.substring(1, name.length - 1);
      }
      if (name.isEmpty) return null;
      entries[int.parse(match.group(1)!)] = name;
    }
    if (entries.isEmpty) return null;
    final indices = entries.keys.toList()..sort();
    return indices.map((i) => entries[i]!).toList();
  }

  /// Returns the exported square input size (`imgsz`) from ultralytics
  /// ncnn `metadata.yaml`, or null when it is absent or non-numeric.
  ///
  /// ultralytics emits either a scalar or a YAML list (the default for
  /// its `yaml.dump`), so both layouts are accepted:
  /// ```yaml
  /// imgsz: 640
  /// ```
  /// ```yaml
  /// imgsz:
  /// - 640
  /// - 640
  /// ```
  static int? tryParseImgSize(String yaml) {
    final match = RegExp(
      r'^imgsz:\s*(?:(\d+)|-\s*(\d+))',
      multiLine: true,
    ).firstMatch(yaml);
    if (match == null) return null;
    final parsed = int.tryParse(match.group(1) ?? match.group(2) ?? '');
    return (parsed != null && parsed > 0) ? parsed : null;
  }
}
