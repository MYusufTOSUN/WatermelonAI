import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// 11-D visual feature vector matching backend/module_a/visual_analyzer.py:
///   extract_visual_feature_vector().
///
/// Layout:
///   [0] field_spot_yellow_ratio
///   [1] field_spot_mean_hue (0-179, OpenCV scale)
///   [2] field_spot_mean_saturation (0-255)
///   [3] field_spot_mean_value (0-255)
///   [4] field_spot_ripeness_score = min(1, yellow*3 + sat/255 * 0.3)
///   [5] texture_variance (Laplacian variance of grayscale)
///   [6] mean_gradient (Sobel magnitude mean)
///   [7] std_gradient (Sobel magnitude std)
///   [8] brightness_std (grayscale std)
///   [9] matteness_score = min(1, texture_variance / 1000)
///   [10] visual_score (0.6*field_spot_ripeness + 0.4*matteness)
class VisualFeatureExtractor {
  static const int featureDim = 11;

  Float64List extract(img.Image image) {
    final fieldSpot = _analyzeFieldSpot(image);
    final texture = _analyzeTexture(image);

    final visualScore = fieldSpot[4] * 0.6 + texture[4] * 0.4;
    return Float64List.fromList([
      fieldSpot[0], fieldSpot[1], fieldSpot[2], fieldSpot[3], fieldSpot[4],
      texture[0], texture[1], texture[2], texture[3], texture[4],
      visualScore,
    ]);
  }

  /// Returns [yellow_ratio, mean_hue, mean_sat, mean_value, ripeness_score].
  /// HSV scale: hue 0-179 (OpenCV-style), sat/val 0-255.
  List<double> _analyzeFieldSpot(img.Image src) {
    final h = src.height;
    final w = src.width;
    final y0 = (h * 0.7).floor();

    int yellowCount = 0;
    int total = 0;
    double sumH = 0.0;
    double sumS = 0.0;
    double sumV = 0.0;

    for (int y = y0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = src.getPixel(x, y);
        final hsv = _rgbToHsvOpenCv(p.r.toInt(), p.g.toInt(), p.b.toInt());
        final hh = hsv[0];
        final ss = hsv[1];
        final vv = hsv[2];
        sumH += hh;
        sumS += ss;
        sumV += vv;
        total++;
        // Backend yellow mask: H in [15,45], S in [50,255], V in [50,255]
        if (hh >= 15 && hh <= 45 && ss >= 50 && vv >= 50) {
          yellowCount++;
        }
      }
    }
    if (total == 0) return [0.0, 0.0, 0.0, 0.0, 0.0];
    final yellowRatio = yellowCount / total;
    final meanH = sumH / total;
    final meanS = sumS / total;
    final meanV = sumV / total;
    final ripenessScore =
        math.min(1.0, yellowRatio * 3.0 + meanS / 255.0 * 0.3);
    return [yellowRatio, meanH, meanS, meanV, ripenessScore];
  }

  /// Returns [texture_variance, mean_gradient, std_gradient, brightness_std, matteness_score].
  List<double> _analyzeTexture(img.Image src) {
    final h = src.height;
    final w = src.width;
    final gray = Float64List(w * h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = src.getPixel(x, y);
        // Standard luma
        final g =
            0.114 * p.b + 0.587 * p.g + 0.299 * p.r;
        gray[y * w + x] = g;
      }
    }

    // Brightness mean & std
    double sum = 0.0;
    for (final v in gray) {
      sum += v;
    }
    final mean = sum / gray.length;
    double varSum = 0.0;
    for (final v in gray) {
      final d = v - mean;
      varSum += d * d;
    }
    final brightnessStd = math.sqrt(varSum / gray.length);

    // Laplacian (4-neighbor): L = I_{x-1,y} + I_{x+1,y} + I_{x,y-1} + I_{x,y+1} - 4*I
    double lapMean = 0.0;
    int lapCount = 0;
    final lapBuf = <double>[];
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final c = gray[y * w + x];
        final lap = gray[y * w + (x - 1)] +
            gray[y * w + (x + 1)] +
            gray[(y - 1) * w + x] +
            gray[(y + 1) * w + x] -
            4 * c;
        lapBuf.add(lap);
        lapMean += lap;
        lapCount++;
      }
    }
    lapMean = lapCount > 0 ? lapMean / lapCount : 0.0;
    double lapVar = 0.0;
    for (final v in lapBuf) {
      final d = v - lapMean;
      lapVar += d * d;
    }
    final textureVariance = lapCount > 0 ? lapVar / lapCount : 0.0;

    // Sobel gradient magnitude
    double gSum = 0.0;
    int gCount = 0;
    final gBuf = <double>[];
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final tl = gray[(y - 1) * w + (x - 1)];
        final tc = gray[(y - 1) * w + x];
        final tr = gray[(y - 1) * w + (x + 1)];
        final cl = gray[y * w + (x - 1)];
        final cr = gray[y * w + (x + 1)];
        final bl = gray[(y + 1) * w + (x - 1)];
        final bc = gray[(y + 1) * w + x];
        final br = gray[(y + 1) * w + (x + 1)];

        final sx = -tl - 2 * cl - bl + tr + 2 * cr + br;
        final sy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        final mag = math.sqrt(sx * sx + sy * sy);
        gBuf.add(mag);
        gSum += mag;
        gCount++;
      }
    }
    final meanGradient = gCount > 0 ? gSum / gCount : 0.0;
    double gVar = 0.0;
    for (final v in gBuf) {
      final d = v - meanGradient;
      gVar += d * d;
    }
    final stdGradient = gCount > 0 ? math.sqrt(gVar / gCount) : 0.0;

    final matteness = math.min(1.0, textureVariance / 1000.0);
    return [textureVariance, meanGradient, stdGradient, brightnessStd, matteness];
  }

  /// RGB (0-255) → HSV OpenCV style: H in [0,179], S/V in [0,255].
  List<double> _rgbToHsvOpenCv(int r, int g, int b) {
    final rN = r / 255.0;
    final gN = g / 255.0;
    final bN = b / 255.0;
    final maxC = math.max(rN, math.max(gN, bN));
    final minC = math.min(rN, math.min(gN, bN));
    final delta = maxC - minC;
    double h = 0.0;
    if (delta > 1e-6) {
      if (maxC == rN) {
        h = 60.0 * (((gN - bN) / delta) % 6);
      } else if (maxC == gN) {
        h = 60.0 * (((bN - rN) / delta) + 2);
      } else {
        h = 60.0 * (((rN - gN) / delta) + 4);
      }
    }
    if (h < 0) h += 360.0;
    // OpenCV scales H to [0, 179]
    final hCv = h / 2.0;
    final s = maxC > 1e-6 ? delta / maxC : 0.0;
    final v = maxC;
    return [hCv, s * 255.0, v * 255.0];
  }
}
