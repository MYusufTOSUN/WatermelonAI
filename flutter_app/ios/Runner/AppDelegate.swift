import Flutter
import UIKit
import os

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let channelName = "watermelon_ai/native"

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let result = super.application(application, didFinishLaunchingWithOptions: launchOptions)

    if let controller = window?.rootViewController as? FlutterViewController {
      let channel = FlutterMethodChannel(
        name: channelName,
        binaryMessenger: controller.binaryMessenger
      )

      channel.setMethodCallHandler { [weak self] (call: FlutterMethodCall, result: @escaping FlutterResult) in
        switch call.method {
        case "getElapsedRealtimeNanos":
          // iOS'ta Android'deki gibi elapsedRealtimeNanos yok; burada
          // monotonic kabul edilen bir zaman kaynağından ns cinsinden değer döndürüyoruz.
          // CACurrentMediaTime() saniye cinsinden, bunu ns'e çeviriyoruz.
          let seconds = CACurrentMediaTime()
          let nanos = Int64(seconds * 1_000_000_000)
          result(nanos)

        case "getDeviceInfo":
          let device = UIDevice.current
          let info: [String: String] = [
            "manufacturer": "Apple",
            "model": device.model,
            "systemName": device.systemName,
            "systemVersion": device.systemVersion
          ]
          result(info)

        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }

    GeneratedPluginRegistrant.register(with: self)
    return result
  }
}
