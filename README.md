# gold-price-prediction-project

Bu proje, son 10 yıllık gram altın verilerini kullanarak geliştirilmiş, özel bir veri önişleme sürecinden geçen ve lineer regresyon algoritmasıyla çalışan bir fiyat tahminleme sistemidir.

## 📌 Proje Özeti
* **Veri Seti:** 10 yıllık gram altın piyasa verileri (`gram_gold_10yrs.csv`).
* **Algoritma:** Gradyan İnişi (Gradient Descent) ile optimize edilmiş Lineer Regresyon.
* **Hedef:** Bir önceki günün açılış ve kapanış verilerini kullanarak, bir sonraki günün tahmini gram altın fiyatını hesaplamak.

## 🛠 Kullanılan Araçlar ve Teknolojiler
* **Programlama Dili:** Python
* **Veri İşleme:** Pandas (Vektörel işlemler), NumPy (Matematiksel hesaplamalar ve model optimizasyonu)
* **Geliştirme Ortamı:** Modüler yapı (`predict.py` ve `functions.py` modülleri)

## ⚙️ Teknik Özellikler & Veri Önişleme
Projenin en güçlü yanı, ham verideki hataların temizlenmesidir. Projede kullanılan `functions.py` modülü şunları içerir:
* **Veri Temizleme:** Sayısal formatlardaki hatalı noktalamaların (binlik ayraçlar vb.) düzeltilmesi.
* **Tarih Parse:** Türkçe ay isimlerindeki karakter hatalarını (örn: 'Ţubat', 'Ađustos') düzelten özel bir tarih formatlayıcı.
* **Feature Engineering:** `shift(1)` metodu ile verilerin bir gün kaydırılarak "Prev_Acilis" ve "Prev_Kapanis" (Önceki günün verisi) özelliklerinin oluşturulması.
* **Normalizasyon:** Modelin daha kararlı çalışması için verilerin standart ölçekleme (z-score normalization) ile ölçeklenmesi.

## 🚀 Nasıl Çalışır?
Modeli çalıştırmak için ana dizinde `predict.py` dosyasını çalıştırmanız yeterlidir:

```bash
python predict.py
