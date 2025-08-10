# BLIP Model Fine-tuning for Car Brand Classification

## Proje Genel Bakış

Bu proje, **BLIP (Bootstrapping Language-Image Pre-training)** modelini kullanarak araba markası sınıflandırma görevini **image captioning** yaklaşımıyla çözen kapsamlı bir machine learning pipeline'ıdır. Geleneksel sınıflandırma yaklaşımından farklı olarak, model görüntüleri için "A photo of a car from the brand [MARKA]" formatında altyazılar üretir ve bu altyazılardan marka bilgisini çıkararak sınıflandırma yapar.

### Teknoloji Stack
- **Base Model**: Salesforce/blip-image-captioning-base (Hugging Face)
- **Framework**: PyTorch, Transformers, Accelerate
- **Cihaz Desteği**: Apple M3 MPS, CUDA, CPU
- **Veri Kümesi**: Kaggle Car Brand Classification Dataset
- **Metrikler**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Projenin Özellikleri
- ✅ Sınıflandırma verisini captioning formatına dönüştürme
- ✅ Memory-efficient fine-tuning (gradient checkpointing, accumulation)
- ✅ Multi-platform device support (MPS/CUDA/CPU)
- ✅ Comprehensive evaluation pipeline
- ✅ Automated brand extraction from captions
- ✅ Detailed performance analysis and visualization

## 1. Sistem Gereksinimleri ve Kurulum

### Minimum Sistem Gereksinimleri
- **RAM**: 8GB+ (16GB önerilen)
- **GPU**: Apple M1/M2/M3 MPS veya NVIDIA GPU (opsiyonel, CPU'da da çalışır)
- **Disk Alanı**: ~5GB (model + veri kümesi)
- **Python**: 3.8-3.11

### Detaylı Kurulum Adımları

#### 1.1 Conda Ortamı Oluşturma
```bash
# Yeni conda ortamı oluştur
conda create -n huggingface python=3.11 -y

# Ortamı aktifleştir  
conda activate huggingface

# Accelerate kütüphanesini conda ile kur (Trainer için gerekli)
conda install -n huggingface -c conda-forge -y "accelerate>=0.26.0"
```

#### 1.2 Python Bağımlılıklarının Kurulumu
```bash
# requirements.txt'ten tüm bağımlılıkları kur
pip install -r requirements.txt
```

**requirements.txt içeriği:**
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Hugging Face transformers library
- `pillow` - Image processing
- `pandas` - Data manipulation
- `scikit-learn` - ML metrics and evaluation
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars

## 2. Veri Kümesi Yapısı ve Hazırlığı

### 2.1 Kaggle Veri Kümesi İndirme
```bash
# Kaggle CLI ile veri kümesini indir (kaggle hesabı gerekli)
kaggle datasets download -d "car-brand-classification-dataset"

# Manuel indirme alternatifi:
# https://www.kaggle.com/datasets/car-brand-classification-dataset sayfasından ZIP dosyasını indir
```

### 2.2 Proje Klasör Yapısı
```
fine-tuning-blip-for-cars/
├── Car Brand Classification Dataset/    # Ana veri kümesi klasörü
│   ├── train/                          # Eğitim split'i (11,517 görüntü)
│   │   ├── Acura/         (349 adet)
│   │   ├── Aston Martin/  (349 adet)
│   │   ├── Audi/          (349 adet)
│   │   ├── BMW/           (349 adet)
│   │   ├── Bentley/       (349 adet)
│   │   ├── Buick/         (349 adet)
│   │   ├── Cadillac/      (349 adet)
│   │   ├── Chevrolet/     (349 adet)
│   │   ├── Chrysler/      (349 adet)
│   │   ├── Dodge/         (349 adet)
│   │   ├── FIAT/          (349 adet)
│   │   ├── Ford/          (349 adet)
│   │   ├── GMC/           (349 adet)
│   │   ├── Honda/         (349 adet)
│   │   ├── Hyundai/       (349 adet)
│   │   ├── INFINITI/      (349 adet)
│   │   ├── Jaguar/        (349 adet)
│   │   ├── Jeep/          (349 adet)
│   │   ├── Kia/           (349 adet)
│   │   ├── Land Rover/    (349 adet)
│   │   ├── Lexus/         (349 adet)
│   │   ├── Lincoln/       (349 adet)
│   │   ├── MINI/          (349 adet)
│   │   ├── Mazda/         (349 adet)
│   │   ├── Mercedes-Benz/ (349 adet)
│   │   ├── Mitsubishi/    (349 adet)
│   │   ├── Nissan/        (349 adet)
│   │   ├── Porsche/       (349 adet)
│   │   ├── Ram/           (349 adet)
│   │   ├── Subaru/        (349 adet)
│   │   ├── Toyota/        (349 adet)
│   │   ├── Volkswagen/    (349 adet)
│   │   └── Volvo/         (349 adet)
│   ├── val/                            # Validation split'i (2,475 görüntü)
│   │   └── [Aynı 33 marka, her birinden 75 adet]
│   └── test/                           # Test split'i (2,475 görüntü)
│       └── [Aynı 33 marka, her birinden 75 adet]
├── Scripts/
│   ├── prepare_dataset.py              # Metadata oluşturma script'i
│   ├── finetune_cars.py               # Fine-tuning script'i
│   └── evaluate_model.py              # Değerlendirme script'i
├── Models/
│   └── car_brand_model_final/         # Eğitilmiş model dosyaları
├── Results/
│   ├── evaluation_predictions.csv     # Validation tahminleri
│   ├── evaluation_predictions_test_split.csv  # Test tahminleri
│   ├── confusion_matrix.png           # Validation confusion matrix
│   ├── confusion_matrix_test_split.png # Test confusion matrix
│   └── evaluation_report*.txt         # Detaylı metrik raporları
├── metadata.jsonl                     # Caption formatında eğitim metadata'sı
├── requirements.txt                   # Python bağımlılıkları
├── README.md                          # Bu dosya
└── RESULTS.md                         # Detaylı sonuç analizi
```

### 2.3 Veri Kümesi Özellikleri
- **Toplam Sınıf Sayısı**: 33 araba markası
- **Toplam Görüntü**: 16,467 adet
- **Görüntü Formatı**: JPG, çeşitli boyutlarda
- **Eğitim/Val/Test Dağılımı**: ~70%/15%/15%
- **Sınıf Dengelenmiş**: Her markadan eşit sayıda örnek

## 3. Metadata Hazırlığı (Sınıflandırmadan Caption Formatına)

### 3.1 Script Çalıştırma
```bash
python prepare_dataset.py
```

### 3.2 İşlem Detayları
Bu script, klasik sınıflandırma veri kümesini BLIP modelinin anlayabileceği captioning formatına dönüştürür:

**Giriş**: Klasör bazlı sınıflandırma yapısı
```
train/
├── Audi/
│   ├── audi_1.jpg
│   ├── audi_2.jpg
│   └── ...
├── BMW/
│   ├── bmw_1.jpg
│   └── ...
```

**Çıkış**: JSONL formatında metadata dosyası (`metadata.jsonl`)
```json
{"file_name": "Car Brand Classification Dataset/train/Audi/audi_1.jpg", "text": "A photo of a car from the brand Audi."}
{"file_name": "Car Brand Classification Dataset/train/BMW/bmw_1.jpg", "text": "A photo of a car from the brand BMW."}
```

### 3.3 Caption Template Stratejisi
- **Sabit format**: "A photo of a car from the brand [MARKA]."
- **Tutarlı söz dizimi**: Model öğrenmesini kolaylaştırır
- **Marka adı vurgulama**: "brand" kelimesi sonrası marka adı çıkarımını kolaylaştırır

## 4. Model Fine-tuning (İnce Ayar)

### 4.1 Script Çalıştırma
```bash
python finetune_cars.py
```

### 4.2 Eğitim Parametreleri ve Optimizasyonlar

#### Model Konfigürasyonu
- **Base Model**: `Salesforce/blip-image-captioning-base`
- **Architecture**: Vision Transformer + GPT-2 Text Decoder
- **Pre-trained**: MS COCO, Visual Genome, CC3M, CC12M, SBU

#### Memory Optimization (MPS/Low-Memory için)
```python
TrainingArguments(
    per_device_train_batch_size=1,        # Küçük batch size
    gradient_accumulation_steps=8,         # Efektif batch: 1×8=8
    gradient_checkpointing=True,           # Memory tasarrufu
    dataloader_num_workers=0,              # MPS uyumluluğu
    dataloader_pin_memory=False,           # MPS için gerekli
    use_mps_device=True,                   # Apple Silicon desteği
    max_length=64,                         # Kısa caption'lar
    truncation=True                        # Uzun girişleri kes
)
```

#### Eğitim Hiperparametreleri
- **Learning Rate**: 5e-5 (BLIP için optimal)
- **Epochs**: 1 (büyük dataset + overfitting önleme)
- **Optimizer**: AdamW (default)
- **Scheduler**: Linear warmup + decay
- **Loss Function**: Cross-entropy (captioning loss)

### 4.3 Eğitim Süreci Monitoring
Eğitim sırasında izlenen metrikler:
- **Loss**: Caption generation loss
- **Learning Rate**: Decay schedule
- **Gradient Norm**: Stability indicator
- **Epoch Progress**: Training completion

## 5. Model Değerlendirme (Evaluation)

### 5.1 Validation Split Değerlendirmesi
```bash
python evaluate_model.py --model_dir ./car_brand_model_final --split_dir "Car Brand Classification Dataset/val" --num_beams 3
```

### 5.2 Test Split Değerlendirmesi
```bash
python evaluate_model.py --model_dir ./car_brand_model_final --split_dir "Car Brand Classification Dataset/test" --num_beams 3
```

### 5.3 Değerlendirme Parametreleri
- **num_beams**: Beam search beam sayısı (1=greedy, 3=daha kararlı)
- **max_new_tokens**: Üretilecek maksimum token sayısı (16)
- **limit**: Test için örnek limiti (0=tümü)

### 5.4 Üretilen Çıktı Dosyaları

#### Validation Split Çıktıları:
- `evaluation_report.txt` - Sınıf bazlı precision, recall, F1-score
- `evaluation_predictions.csv` - Her örnek için detaylı tahmin
- `confusion_matrix.png` - Görsel karmaşıklık matrisi
- `evaluation_summary.md` - Özet rapor

#### Test Split Çıktıları:
- `evaluation_report_test_split.txt`
- `evaluation_predictions_test_split.csv` 
- `confusion_matrix_test_split.png`
- `evaluation_summary_test_split.md`

### 5.5 Brand Extraction (Marka Çıkarım) Algoritması

Model altyazı üretir, marka sınıflandırması için çok-kademeli çıkarım algoritması:

```python
def extract_brand_from_caption(caption, brands):
    # 1. "brand [NAME]" pattern matching
    # 2. Direct normalization match
    # 3. Substring containment (bidirectional)
    # 4. Similarity scoring with difflib
```

**Örnek işlem**:
- **Model Output**: "A photo of a car from the brand Audi."
- **Extract**: "Audi" 
- **Normalize**: "audi"
- **Match**: brands.index("Audi") → prediction

## 6. Performans Sonuçları

### 6.1 Ana Metrikler
- **Validation Accuracy**: 81.09% (2,007/2,475 doğru)
- **Test Accuracy**: 79.68% (1,971/2,475 doğru)
- **Model Size**: ~990MB (fine-tuned)
- **Training Time**: ~77 dakika (M3 MPS)

### 6.2 Eğitim Metrikleri
- **Final Training Loss**: 0.1451
- **Training Samples/sec**: 2.477
- **Total Steps**: 1,440

## 7. Sorun Giderme ve Optimizasyon

### 7.1 Yaygın Hatalar ve Çözümleri

#### Accelerate Hatası
```
Error: Using the Trainer with PyTorch requires accelerate>=0.26.0
```
**Çözüm**:
```bash
conda install -n huggingface -c conda-forge -y "accelerate>=0.26.0"
```

#### MPS Memory Overflow
```
Error: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
```
**Çözüm**:
```python
# finetune_cars.py içinde
per_device_train_batch_size=1  # Daha da küçük batch
gradient_accumulation_steps=16  # Daha fazla accumulation
```

#### Processor Config Bulunamadı
```
OSError: ./car_brand_model_final does not appear to have a file named preprocessor_config.json
```
**Çözüm**: Model tekrar eğitilmeli (processor.save_pretrained eklendi)

### 7.2 Performans Optimizasyonu

#### GPU Memory Kullanımı
- **8GB VRAM**: batch_size=1, accumulation=8
- **16GB VRAM**: batch_size=2, accumulation=4
- **24GB+ VRAM**: batch_size=4, accumulation=2

#### Inference Hızlandırma
```python
# Hızlı inference için
model.eval()
torch.set_grad_enabled(False)
num_beams=1  # Greedy decoding
```

## 8. Gelişmiş Kullanım

### 8.1 Custom Test Script
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("./car_brand_model_final")
model = BlipForConditionalGeneration.from_pretrained("./car_brand_model_final")

image = Image.open("test_car.jpg")
inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, num_beams=3)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Generated caption: {caption}")
```

### 8.2 Model Export ve Deployment
```python
# ONNX export
torch.onnx.export(model, inputs, "car_brand_model.onnx")

# TorchScript export  
traced_model = torch.jit.trace(model, inputs)
traced_model.save("car_brand_model.pt")
```

## 9. Lisans ve Referanslar

### 9.1 Model ve Veri Lisansları
- **BLIP Model**: Salesforce Research (Apache 2.0)
- **Dataset**: Kaggle Car Brand Classification (Public Domain)
- **Transformers Library**: Hugging Face (Apache 2.0)

### 9.2 Referanslar
- [BLIP Paper](https://arxiv.org/abs/2201.12086): "BLIP: Bootstrapping Language-Image Pre-training"
- [Hugging Face BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Kaggle Dataset](https://www.kaggle.com/datasets/car-brand-classification-dataset)

### 9.3 Citation
```bibtex
@article{blip2022,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  journal={ICML},
  year={2022}
}
```


## 10. Hugging Face Modeli

Model, Hugging Face Hub üzerinde yayımlanmıştır. Kullanım ve detaylar için model sayfasına bakabilirsiniz: 

- Model sayfası: [SIYAKSARES/fine-tuned-blip-for-car-brands](https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands)

Hızlı kullanım örneği:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("SIYAKSARES/fine-tuned-blip-for-car-brands")
model = BlipForConditionalGeneration.from_pretrained("SIYAKSARES/fine-tuned-blip-for-car-brands")

image = Image.open("your_test_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, num_beams=3, max_new_tokens=16)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated caption:", caption)
```


