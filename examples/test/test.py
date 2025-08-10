from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import glob

# Fine-tune edilmiş model yolunu belirtelim
model_path = "./car_brand_model_final"

# Processor'ı orijinal modelden yükleyelim
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    use_fast=True
)

# Fine-tune edilmiş modeli yükleyelim
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Test klasöründeki tüm resimleri bul
def get_all_test_images():
    """Test klasöründeki tüm resim dosyalarını bulur"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    
    for extension in image_extensions:
        images = glob.glob(extension)
        all_images.extend(images)
    
    return all_images

def test_all_images():
    """Test klasöründeki tüm resimleri tek tek modele tanıtır"""
    image_files = get_all_test_images()
    
    if not image_files:
        print("Test klasöründe resim dosyası bulunamadı!")
        return
    
    print(f"Toplam {len(image_files)} resim bulundu. Test başlıyor...\n")
    
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Test ediliyor: {img_path}")
            
            # Resmi yükle
            raw_image = Image.open(img_path)
            
            # Model tahmini yap
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            result = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"Model tahmini: {result}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Hata oluştu ({img_path}): {e}")
            print("-" * 50)

# Tüm test resimlerini çalıştır
test_all_images()