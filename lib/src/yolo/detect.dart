import 'dart:math' as math;
import 'dart:typed_data';

/// One decoded detection. (x, y) is the box CENTER in input-pixel
/// units; (w, h) the box size.
class NcnnDetection {
  const NcnnDetection({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.classIndex,
    required this.score,
  });

  final double x;
  final double y;
  final double w;
  final double h;
  final int classIndex;
  final double score;
}

/// Decodes an ultralytics ncnn detect output and applies per-class NMS.
///
/// The exported graph already contains DFL + box decode — the output is
/// per-anchor [cx, cy, w, h, cls0..clsN] with cls probs in [0, 1].
/// Layouts: [4+numClasses, n] (column-major, common) or
/// [n, 4+numClasses] (row-major) — auto-detected from [shape]
/// (ncnn convention (w, h, d, c): shape[0]==4+numClasses means
/// column-major).
///
/// Coordinates are in the model INPUT resolution; scale by
/// (imageWidth / inputWidth) for original-image boxes.
List<NcnnDetection> decodeYoloDetect(
  Float32List out,
  List<int> shape, {
  required int numClasses,
  double confThreshold = 0.25,
  double iouThreshold = 0.45,
}) {
  final stride = 4 + numClasses;
  var n = 0;
  var colMajor = false;
  if (shape[0] == stride && shape.length >= 2) {
    n = shape[1];
    colMajor = true;
  } else if (shape.length >= 2 && shape[1] == stride) {
    n = shape[0];
    colMajor = false;
  } else {
    throw ArgumentError(
      'shape $shape matches neither [$stride, n] nor [n, $stride] '
      'for numClasses=$numClasses',
    );
  }
  if (out.length < n * stride) {
    throw ArgumentError('buffer ${out.length} < $n*$stride');
  }

  final candidates = <NcnnDetection>[];
  for (var i = 0; i < n; i++) {
    var bestCls = 0;
    var bestScore = -1.0;
    for (var c = 0; c < numClasses; c++) {
      final s = _at(out, i, 4 + c, colMajor, stride, n);
      if (s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }
    if (bestScore < confThreshold) continue;
    candidates.add(NcnnDetection(
      x: _at(out, i, 0, colMajor, stride, n),
      y: _at(out, i, 1, colMajor, stride, n),
      w: _at(out, i, 2, colMajor, stride, n),
      h: _at(out, i, 3, colMajor, stride, n),
      classIndex: bestCls,
      score: bestScore,
    ));
  }
  return _nmsPerClass(candidates, iouThreshold);
}

double _at(Float32List out, int i, int ch, bool colMajor, int stride, int n) {
  return colMajor ? out[ch * n + i] : out[i * stride + ch];
}

List<NcnnDetection> _nmsPerClass(
    List<NcnnDetection> dets, double iouThreshold) {
  final byClass = <int, List<NcnnDetection>>{};
  for (final d in dets) {
    byClass.putIfAbsent(d.classIndex, () => []).add(d);
  }
  final kept = <NcnnDetection>[];
  for (final list in byClass.values) {
    list.sort((a, b) => b.score.compareTo(a.score));
    final suppressed = List<bool>.filled(list.length, false);
    for (var i = 0; i < list.length; i++) {
      if (suppressed[i]) continue;
      kept.add(list[i]);
      for (var j = i + 1; j < list.length; j++) {
        if (!suppressed[j] && _iou(list[i], list[j]) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  kept.sort((a, b) => b.score.compareTo(a.score));
  return kept;
}

double _iou(NcnnDetection a, NcnnDetection b) {
  final x1 = math.max(a.x - a.w / 2, b.x - b.w / 2);
  final y1 = math.max(a.y - a.h / 2, b.y - b.h / 2);
  final x2 = math.min(a.x + a.w / 2, b.x + b.w / 2);
  final y2 = math.min(a.y + a.h / 2, b.y + b.h / 2);
  final inter = math.max(0.0, x2 - x1) * math.max(0.0, y2 - y1);
  if (inter == 0) return 0;
  final union_ = a.w * a.h + b.w * b.h - inter;
  return union_ <= 0 ? 0 : inter / union_;
}
