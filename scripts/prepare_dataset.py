import os
import json


def create_metadata_file():
    """
    Sınıflandırma veri kümesinin klasör yapısını tarar ve görüntü altyazı (caption)
    ince ayarı için kullanılacak `metadata.jsonl` dosyasını oluşturur.

    Beklenen klasör yapısı:
    `Car Brand Classification Dataset/train/<MarkaAdı>/*.jpg`
    """

    # Projedeki gerçek klasör adına göre ayarlandı
    dataset_base_path = "Car Brand Classification Dataset/train"
    output_file = "metadata.jsonl"

    # Klasör mevcut mu kontrolü
    if not os.path.exists(dataset_base_path):
        print(f"Hata: Veri kümesi yolu bulunamadı: '{dataset_base_path}'")
        print("Lütfen Kaggle veri kümesini doğru klasöre çıkardığınızdan emin olun.")
        return

    print(f"Veri kümesi hazırlanıyor: {dataset_base_path}")

    # Marka klasörlerini al (örn. 'Audi', 'BMW', ...)
    brand_folders = [
        d for d in os.listdir(dataset_base_path)
        if os.path.isdir(os.path.join(dataset_base_path, d))
    ]
    brand_folders.sort()

    total_images = 0

    # Çıktı dosyasını oluştur ve her satıra tek bir JSON nesnesi yaz
    with open(output_file, "w", encoding="utf-8") as writer:
        for brand_name in brand_folders:
            brand_path = os.path.join(dataset_base_path, brand_name)
            print(f"  İşleniyor: {brand_name}")

            # Görsel dosyalarını filtrele
            image_files = [
                img for img in os.listdir(brand_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            image_files.sort()

            for image_name in image_files:
                image_path = os.path.join(brand_path, image_name)

                # İstenilen biçimde açıklama (caption)
                caption = f"A photo of a car from the brand {brand_name}."

                data_entry = {
                    "file_name": image_path,
                    "text": caption,
                }

                writer.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                total_images += 1

    print("-" * 40)
    print(f"Tamamlandı: '{output_file}' oluşturuldu (toplam {total_images} görsel).")


if __name__ == "__main__":
    create_metadata_file()


