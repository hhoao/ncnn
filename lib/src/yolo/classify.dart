import 'dart:math' as math;
import 'dart:typed_data';

/// Pure-Dart post-processing for classify outputs.

/// Index of the highest-scoring element.
int argmax(Float32List scores) {
  var best = 0;
  for (var i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

/// Top-k (index, value) pairs, descending by value.
List<(int, double)> topK(Float32List scores, int k, {int? limit}) {
  final n = (limit ?? scores.length).clamp(0, scores.length);
  final indexed = List.generate(n, (i) => (i, scores[i].toDouble()));
  indexed.sort((a, b) => b.$2.compareTo(a.$2));
  return indexed.take(k.clamp(0, n)).toList();
}

/// Softmax over the first `limit` entries (max-subtracted for
/// numerical stability).
Float32List softmax(Float32List logits, {int? limit}) {
  final n = (limit ?? logits.length).clamp(0, logits.length);
  if (n == 0) return Float32List(0);
  var maxVal = logits[0];
  for (var i = 1; i < n; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  final out = Float32List(n);
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    final e = math.exp(logits[i] - maxVal);
    out[i] = e;
    sum += e;
  }
  for (var i = 0; i < n; i++) {
    out[i] /= sum;
  }
  return out;
}
