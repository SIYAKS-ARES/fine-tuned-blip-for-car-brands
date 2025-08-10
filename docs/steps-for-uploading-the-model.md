## Hugging Face Hub’a Yerel (Fine‑tuned) Model Yükleme Rehberi

Bu rehber, `./car_brand_model_final` klasöründe kayıtlı BLIP tabanlı fine‑tuned Transformers modelinizi Hugging Face Hub üzerinde yeni bir depoya yüklemenize yardımcı olur. Örnek hedef depo: [SIYAKSARES/fine-tuned-blip-for-car-brands](https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands).

### 1) Prerequisites: Hesap ve Kimlik Doğrulama

- **Hesap oluşturma**: Henüz hesabınız yoksa buradan oluşturun: [huggingface.co/join](https://huggingface.co/join)
- **Access Token (write yetkili)**:
  - Ayarlar → Access Tokens: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - “New token” → bir ad verin → rol: **write** → oluşturun → token’ı kopyalayın
- **CLI ile giriş**:
  ```bash
  huggingface-cli login
  ```
  - İstendiğinde kopyaladığınız token’ı yapıştırın ve Enter’a basın

### 2) Ortam Kurulumu: Git LFS (Large File Storage)

- **Neden gerekli?** `pytorch_model.bin` gibi büyük model ağırlıkları Git LFS ile verimli şekilde yönetilir; normal Git ile sorun yaşanır.
- **macOS (Homebrew)**:
  ```bash
  brew install git-lfs
  ```
- **Debian/Ubuntu**:
  ```bash
  sudo apt-get update && sudo apt-get install -y git-lfs
  ```
- **Zorunlu ilk kurulum (tek seferlik)**:
  ```bash
  git lfs install
  ```

### 3) Python Yükleme Script’i

Aşağıdaki script’i proje kök dizininize `upload_model.py` adıyla kaydedin. Yerel dizinden hem processor’ı hem de modeli yükler ve `.push_to_hub()` ile belirttiğiniz repo’ya gönderir.

```python
# upload_model.py

from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    # 1) Yerel model klasörü ve hedef Hub repo kimliği
    # Kendi kullanıcı adınız ve model adınızla değiştirin:
    # Örn: "SIYAKSARES/fine-tuned-blip-for-car-brands"
    local_model_path = "./car_brand_model_final"
    repo_id = "your-username/your-model-name"

    print(f"Local model path: {local_model_path}")
    print(f"Target Hub repo: {repo_id}")

    # 2) Yerelden processor ve model yükleme
    print("Loading processor from local directory...")
    processor = BlipProcessor.from_pretrained(local_model_path)

    print("Loading model from local directory...")
    model = BlipForConditionalGeneration.from_pretrained(local_model_path)

    # 3) Hub'a yükleme (huggingface-cli login yapılmış olmalı)
    print("Uploading processor to the Hub...")
    processor.push_to_hub(repo_id)
    print("Processor upload complete.")

    print("Uploading model to the Hub...")
    model.push_to_hub(repo_id)
    print("Model upload complete!")

    print("All done! Your model and processor are now on the Hub.")

if __name__ == "__main__":
    main()
```

**Çalıştırma**:
```bash
python upload_model.py
```

**Kontrol**: Yükleme tamamlandığında depo sayfanızda model dosyaları görünecektir (örnek: [SIYAKSARES/fine-tuned-blip-for-car-brands](https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands)).

—

**Kaynaklar**
- Hugging Face profil (örnek depo): [SIYAKSARES/fine-tuned-blip-for-car-brands](https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands)
- Hesap oluşturma: [huggingface.co/join](https://huggingface.co/join)
- Access Tokens: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
