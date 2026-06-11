package com.example.watermelon_ai

import android.os.Build
import android.os.SystemClock
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * Native Kotlin köprüsü
 *
 * Amaç:
 *  - SRR / haptik senkronizasyonu için yüksek çözünürlüklü zaman damgası sağlamak.
 *  - Gerekirse cihaz bilgilerini Flutter tarafına aktarmak.
 *
 * Channel adı: "watermelon_ai/native"
 *
 * Metodlar:
 *  - getElapsedRealtimeNanos(): Long
 *      Android'in monotic saatinden (SystemClock.elapsedRealtimeNanos) ns cinsinden zaman döner.
 *  - getDeviceInfo(): Map<String, String>
 *      Marka, model ve SDK versiyonunu döner.
 */
class MainActivity : FlutterActivity() {
    private val CHANNEL = "watermelon_ai/native"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getElapsedRealtimeNanos" -> {
                        try {
                            val nanos = SystemClock.elapsedRealtimeNanos()
                            result.success(nanos)
                        } catch (e: Exception) {
                            result.error(
                                "TIME_ERROR",
                                "elapsedRealtimeNanos error: ${e.message}",
                                null
                            )
                        }
                    }
                    "getDeviceInfo" -> {
                        val info = mapOf(
                            "manufacturer" to Build.MANUFACTURER,
                            "model" to Build.MODEL,
                            "sdkInt" to Build.VERSION.SDK_INT.toString()
                        )
                        result.success(info)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}
