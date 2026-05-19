# LLM-Inference-Bench-Mac — Proje Deep Dive

> Bu doküman, projenin sıfırdan nasıl tasarlandığını, her katmanın ne yaptığını, hangi teknik kararların neden alındığını ve kullanılan tüm teknolojilerin birbirine nasıl bağlandığını açıklar.

---

## 1. Problemin Tanımı

Apple Silicon (M1/M2/M3/M4) çiplerde yerel (local) LLM çalıştırma seçenekleri gittikçe çoğaldı: Ollama, llama.cpp, MLX. Her biri farklı bir yaklaşım sunuyor — ama **hangisi, hangi modelde, hangi quantization seviyesinde en iyi performansı verir?** Piyasada bu soruya gerçek ölçüm verisiyle cevap veren bir araç yoktu.

**Çözüm:** Aynı Türkçe prompt'ları, aynı modelin farklı quantization seviyeleriyle, 3 farklı runtime'da koşturan ve sonuçları karşılaştıran sistematik bir benchmark suite geliştirdim.

---

## 2. Mimari: Katmanlı ve Modüler

Proje 5 ana katmandan oluşur:

```
┌──────────────────────────────────────────────────────┐
│                 CLI (run_benchmark.py)                │
├──────────────────────────────────────────────────────┤
│            Orchestrator (bench/runner.py)             │
├────────────┬─────────────────┬───────────────────────┤
│  Backend   │    Telemetry    │      Evaluation        │
│  Layer     │    Layer        │      Layer             │
│            │                 │                        │
│ • Ollama   │ • MemoryTracker │ • Rule-based (5+1 dim)│
│ • Llama.cpp│ • SystemInfo    │ • LLM-as-a-Judge      │
│ • MLX      │ • AppleMetrics  │                        │
├────────────┴─────────────────┴───────────────────────┤
│              Dashboard (Vanilla JS + Chart.js)        │
└──────────────────────────────────────────────────────┘
```

---

## 3. Backend Katmanı — Her Runtime Nasıl Çalışır

### 3.1 Tasarım Kararı: Abstract Base Class
`bench/backends/base.py` dosyasında `InferenceBackend` soyut sınıfı tanımlanır. Bu sınıf 4 zorunlu metod dayatır:

| Metod | Ne Yapar |
|-------|----------|
| `is_available()` | Runtime sisteme kurulu mu? |
| `load_model(config)` | Modeli belleğe yükle, load time döndür |
| `generate(prompt)` | Token üret, `InferenceResult` döndür |
| `cleanup()` | Belleği serbest bırak |

**Neden bu yapı?** Yeni bir runtime (örn: vLLM) eklemek istediğimde ana kodu (Runner) hiç değiştirmem gerekmez. Sadece yeni bir sınıf yazıp bu 4 metodu implement ederim. Bu, SOLID'in "Open/Closed Principle" kuralıdır.

### 3.2 Ollama Backend (`ollama_backend.py`)
- **İletişim:** `localhost:11434` adresine HTTP REST API üzerinden bağlanır
- **Kütüphane:** `requests` (Session objesi ile connection pooling)
- **Streaming:** `stream=True` ile token token okur → TTFT'yi milisaniye hassasiyetinde yakalar
- **Özellik:** Ollama'nın kendi döndürdüğü `eval_count`, `eval_duration` (nanosaniye) metriklerini de toplar

### 3.3 Llama.cpp Backend (`llamacpp_backend.py`)
- **İletişim:** Python binding'leri (`llama-cpp-python`) üzerinden direkt
- **Derleme:** `CMAKE_ARGS="-DGGML_METAL=on"` ile Metal GPU desteği zorunlu
- **Model Formatı:** GGUF (Q2_K, Q4_K_M, Q8_0 seviyelerinde)
- **Özellik:** `n_gpu_layers=-1` ile tüm katmanları GPU'ya offload eder

### 3.4 MLX Backend (`mlx_backend.py`)
- **İletişim:** Apple'ın kendi MLX framework'ü (`mlx-lm`) üzerinden direkt
- **Avantaj:** Apple Unified Memory Architecture (UMA) sayesinde Zero-copy memory access — CPU↔GPU arası veri kopyalama yok
- **Streaming:** `stream_generate()` API'si ile token token üretir
- **Fallback:** `stream_generate` yoksa `generate()` ile batch mode çalışır

---

## 4. Telemetry Katmanı — Donanımı Gerçek Zamanlı İzleme

### 4.1 MemoryTracker (`bench/metrics.py`)
**Problem:** RAM'i sadece önce-sonra ölçmek, inference sırasında yaşanan Peak (zirve) tüketimi yakalaymaz.

**Çözüm:** Python'un `threading` modülü ile arka planda çalışan bir daemon thread:

```python
with MemoryTracker(interval=0.1) as tracker:
    result = backend.generate(prompt)
# tracker.peak_mb → inference sırasındaki en yüksek RSS değeri
```

- `interval=0.1` → Saniyede 10 ölçüm
- Context Manager pattern ile kullanım (`with ... as`)
- Ana thread'i bloke etmez (Non-blocking)
- `psutil.Process().memory_info().rss` ile RSS (Resident Set Size) ölçer

### 4.2 Apple Silicon Detayları (`bench/apple_metrics.py`)
- GPU çekirdek sayısı (16 core M1 Pro, 32 core M1 Max vb.)
- Neural Engine varlığı kontrolü
- Bellek bant genişliği tahmini (M1 Pro: 200 GB/s)
- Metal API desteği kontrolü
- Çip nesli tespiti (M1/M2/M3/M4 + Pro/Max/Ultra)

### 4.3 SystemInfo
Tüm donanım bilgisini bir dataclass'ta birleştirir:
- `sysctl` ile CPU brand string
- `psutil` ile RAM ve CPU core sayısı
- `system_profiler` ile GPU detayları
- JSON'a serialize olabilir (`to_dict()`)

---

## 5. Evaluation Katmanı — Çıktı Kalitesini Ölçme

### 5.1 Kural Tabanlı Evaluator (`bench/quality.py`)
6 boyutlu deterministik skorlama (API gerektirmez, tamamen lokal):

| Boyut | Ağırlık | Ne Ölçer |
|-------|---------|----------|
| **Length** | %10 | Çıktı beklenen uzunlukta mı? |
| **Keywords** | %25 | Beklenen teknik terimler var mı? |
| **Coherence** | %20 | Cümle yapısı düzgün mü? |
| **Turkish** | %15 | Türkçe karakter yoğunluğu (Ş, Ğ, Ç, Ö) |
| **Relevance** | %15 | Unigram kelime çeşitliliği |
| **Repetition** | %15 | N-gram (trigram) tekrar oranı — dejenere çıktı tespiti |

**Bilinen Sınırlama:** Kural tabanlı sistem halüsinasyonu tespit edemez. "Klorofil ile fotosentez yapılmaz" gibi olgusal olarak yanlış ama doğru kelimeleri içeren cevaplar yüksek puan alır. Bu, Quality Judge ile çözülür.

### 5.2 LLM-as-a-Judge (`bench/quality_judge.py`)
Kural tabanlı sistemin boşluğunu dolduran semantik değerlendirici:
- Ollama üzerinden çalışan küçük bir model (varsayılan: Qwen 2.5 1.5B) hakemlik yapar
- Türkçe rubric ile 4 boyutta skorlar: `factual_accuracy`, `relevance`, `coherence`, `language_quality`
- `temperature=0.0` ile deterministik çıktı
- JSON formatında yapılandırılmış çıktı parse eder
- Opsiyonel: `--judge` CLI flag'i ile aktifleştirilir

---

## 6. Prompt Seti — Türkçe Benchmark Kütüphanesi

`bench/prompts.py` içinde 5 kategoride 9 prompt:

| Kategori | Örnek | Beklenen Anahtar Kelimeler |
|----------|-------|---------------------------|
| **Bilgi** (3) | "Yapay zekanın tarihçesini özetle" | turing, sinir, ağ |
| **Analiz** (2) | "Apple Silicon'ın avantajları" | arm, enerji, unified |
| **Yaratıcı** (2) | "Boğaz hakkında şiir yaz" | istanbul, su, köprü |
| **Kod** (2) | "Binary search yaz" | def, mid, return |
| **Çeviri** (1) | EN→TR çeviri | makine, öğrenme, veri |

3 mod: `quick` (1 prompt), `standard` (3 prompt), `full` (9 prompt)

---

## 7. Orkestratör — Her Şeyi Birleştiren Motor

`bench/runner.py` → `BenchmarkRunner` sınıfı:

**İş akışı:**
1. Sistem bilgisi topla (donanım, OS, Python versiyonu)
2. Backend'leri başlat (sadece kurulu olanlar)
3. Her model için:
   - Modeli yükle + load time ölç
   - **Warmup çalıştır** (cold-start'ı ayıkla)
   - Her prompt × her iterasyon için:
     - MemoryTracker başlat
     - Token üret (stream mode)
     - RAM Peak yakala
     - Kalite skoru hesapla
     - Sonuçları topla
4. İstatistiksel aggregation (mean, median, std, min, max)
5. JSON'a kaydet

**Warmup neden kritik?** İlk çağrıda model diskten RAM'e yüklenir. Bu süreyi ölçüme katarsak "inference hızı"nı yanlış hesaplarız. Warmup ile sadece saf decode throughput ölçeriz.

---

## 8. İstatistiksel Altyapı

- **Mean / Median / Std:** `statistics` modülü
- **Confidence Interval (CI 95%):** `_confidence_interval_95()` → `1.96 * (std / √n)`
- **Coefficient of Variation (CV):** `_coefficient_of_variation()` → std/mean * 100
- **Outlier Filtreleme:** IQR tabanlı (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- **Reproducibility:** Her çıktıda git hash, paket versiyonları ve benchmark parametreleri kaydedilir

---

## 9. Dashboard — Veri Görselleştirme

### Tech Stack
| Teknoloji | Neden |
|-----------|-------|
| **HTML5/CSS3** | Zero dependency, tarayıcıda tıkla-aç |
| **Chart.js** (CDN) | 6 farklı grafik tipi |
| **Vanilla JS (ES6+)** | React/Vue gereksiz overhead |
| **CSS Variables** | Karanlık mod tema sistemi |
| **Glassmorphism** | Blur + transparency ile premium his |

### Grafik Tipleri
1. **Throughput Bar Chart:** tok/s karşılaştırma
2. **TTFT Bar Chart:** İlk token süresi
3. **RAM Horizontal Bar:** Bellek tüketimi sıralaması
4. **Quality vs Speed Scatter:** X=hız, Y=kalite, Bubble=RAM (Pareto Optimal noktayı gösterir)
5. **Radar Chart:** Top 5 modelin çok boyutlu karşılaştırması
6. **Load Time Bar:** Model yükleme süreleri

### Özellikler
- JSON dosyası yükleme (drag-drop veya file picker)
- Backend/Quantization filtreleme
- CSV/PNG dışa aktarma
- Sortable data table
- Responsive layout

---

## 10. Test Altyapısı — 59 Test

### Yapı
```
tests/
├── conftest.py        # Shared fixtures (8 fixture)
├── test_quality.py    # 22 test — kalite ölçüm fonksiyonları
├── test_backends.py   # 24 test — backend arayüz uyumu
└── test_runner.py     # 13 test — aggregation ve config
```

### Test Stratejisi
- **Mock-based:** Gerçek model yüklemeden backend davranışını test eder (`unittest.mock`)
- **Edge case coverage:** Boş string, None, tekrarlı metin, tamamen İngilizce çıktı
- **Known limitation documentation:** Halüsinasyon zayıflığını test olarak belgeler
- **CI/CD:** GitHub Actions ile her push'ta otomatik çalışır (macOS, Python 3.10/3.11/3.12)

---

## 11. Logging Sistemi

`bench/logger.py` → Dual-handler yapı:
- **Console:** Renkli, saat formatında (`HH:MM:SS`)
- **Dosya:** `logs/bench_YYYYMMDD_HHMMSS.log` — kalıcı kayıt
- Her benchmark çalışması kendi log dosyasını oluşturur
- Crash durumunda bile nerede kaldığı belli olur

---

## 12. Tam Teknoloji Yığını

| Katman | Teknoloji | Versiyon/Detay |
|--------|-----------|----------------|
| **Dil** | Python | 3.10+ |
| **Runtime 1** | mlx-lm | Apple Native, zero-copy |
| **Runtime 2** | llama-cpp-python | Metal GPU offload |
| **Runtime 3** | Ollama | REST API, streaming |
| **Bellek İzleme** | psutil | RSS sampling, background thread |
| **Donanım Tespiti** | sysctl, system_profiler | Apple Silicon detayları |
| **CLI Arayüzü** | argparse + rich | Tablolar, progress bar, renkli çıktı |
| **Dashboard** | HTML/CSS/JS + Chart.js | Glassmorphism, 6 grafik |
| **Test** | pytest | 59 test, mock-based |
| **CI/CD** | GitHub Actions | macOS runner, multi-Python |
| **Versiyon Kontrol** | Git + GitHub | Açık kaynak |

---

## 13. Geliştirme Yol Haritası (Gelecek)

- [ ] **Enerji metrikleri:** `powermetrics` ile Watt ölçümü
- [ ] **Concurrency testi:** `asyncio` ile eşzamanlı yük
- [ ] **Needle-in-a-Haystack:** Uzun context degradasyon testi
- [ ] **Cross-device comparison:** M1 vs M2 vs M3 dashboard karşılaştırması
- [ ] **vLLM entegrasyonu:** Apple Silicon desteği olgunlaştığında
