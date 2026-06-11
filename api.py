"""
BU DOSYA ARTIK KULLANILMIYOR.

Proje tamamen OFFLINE modda, Flutter içinden TFLite modelleriyle
çalışacak şekilde yeniden tasarlandı:

  - MRD-YOLO (INT8 TFLite)  : Görsel olgunluk ve hacim/kütle tahmini
  - Akustik MLP (FP16 TFLite): 120-boyutlu akustik/haptik fingerprint sınıflandırması
  - Fusion (Flutter)        : EI = f2^2 * m^(2/3) + P_visual ile geç füzyon

Herhangi bir FastAPI / HTTP API'ye ihtiyaç YOK.

Bu dosyayı çalıştırma, sadece tarihçe için tutuluyor.
"""

