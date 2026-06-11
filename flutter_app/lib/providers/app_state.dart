import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AppState extends ChangeNotifier {
  static const int _maxHistory = 50;
  static const String _kFirstLaunch = "isFirstLaunch";
  static const String _kHistory = "fieldTestHistory";

  bool _isFirstLaunch = true;
  bool _isLoading = true;
  List<Map<String, dynamic>> _history = [];

  bool get isFirstLaunch => _isFirstLaunch;
  bool get isLoading => _isLoading;
  List<Map<String, dynamic>> get history => _history;

  /// Aggregate field-test accuracy from history entries that have ground truth.
  Map<String, dynamic> get fieldStats {
    int total = 0;
    int correct = 0;
    int withGroundTruth = 0;
    for (final e in _history) {
      total++;
      final gt = e["groundTruthClass"];
      if (gt is int) {
        withGroundTruth++;
        final pred = e["classId"];
        if (pred is int && _binaryFromClass(pred) == _binaryFromClass(gt)) {
          correct++;
        }
      }
    }
    return {
      "total": total,
      "withGroundTruth": withGroundTruth,
      "correct": correct,
      "accuracy": withGroundTruth > 0 ? correct / withGroundTruth : null,
    };
  }

  int _binaryFromClass(int c) => c == 1 ? 1 : 0; // ripe=1 else=0

  // ---- Initialization ----
  Future<void> checkFirstLaunch() async {
    final prefs = await SharedPreferences.getInstance();
    _isFirstLaunch = prefs.getBool(_kFirstLaunch) ?? true;
    final rawHistory = prefs.getString(_kHistory);
    if (rawHistory != null) {
      try {
        final list = jsonDecode(rawHistory) as List;
        _history = list
            .map((e) => Map<String, dynamic>.from(e as Map))
            .toList();
      } catch (_) {
        _history = [];
      }
    }
    _isLoading = false;
    notifyListeners();
  }

  Future<void> completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kFirstLaunch, false);
    _isFirstLaunch = false;
    notifyListeners();
  }

  Future<void> resetOnboarding() async {
    _isFirstLaunch = true;
    notifyListeners();
  }

  // ---- History management ----
  Future<void> addHistoryResult(Map<String, dynamic> result) async {
    if (!result.containsKey("id")) {
      result["id"] = DateTime.now().millisecondsSinceEpoch.toString();
    }
    _history.insert(0, result);
    if (_history.length > _maxHistory) {
      _history = _history.sublist(0, _maxHistory);
    }
    notifyListeners();
    await _persist();
  }

  /// Marks ground truth for a given history entry by id.
  /// groundTruthClass: 0=immature, 1=ripe, 2=overripe.
  Future<void> setGroundTruth(String id, int groundTruthClass) async {
    for (int i = 0; i < _history.length; i++) {
      if (_history[i]["id"] == id) {
        _history[i] = {
          ..._history[i],
          "groundTruthClass": groundTruthClass,
          "groundTruthTimestamp": DateTime.now().toIso8601String(),
        };
        notifyListeners();
        await _persist();
        return;
      }
    }
  }

  Future<void> clearGroundTruth(String id) async {
    for (int i = 0; i < _history.length; i++) {
      if (_history[i]["id"] == id) {
        final map = Map<String, dynamic>.from(_history[i]);
        map.remove("groundTruthClass");
        map.remove("groundTruthTimestamp");
        _history[i] = map;
        notifyListeners();
        await _persist();
        return;
      }
    }
  }

  Future<void> deleteEntry(String id) async {
    _history.removeWhere((e) => e["id"] == id);
    notifyListeners();
    await _persist();
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kHistory, jsonEncode(_history));
  }
}
