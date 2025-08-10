import argparse
import os
import string
from dataclasses import dataclass
from typing import List, Tuple

import difflib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_brands_and_images(split_dir: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    split_dir altındaki marka klasörlerini ve görsel yollarını çıkarır.
    Dönenler:
      - brands: Marka adları (sıralı ve benzersiz)
      - samples: (image_path, true_brand)
    """
    brands = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    brands.sort()

    samples: List[Tuple[str, str]] = []
    for brand in brands:
        brand_dir = os.path.join(split_dir, brand)
        for fname in os.listdir(brand_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(brand_dir, fname), brand))

    return brands, samples


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # noktalama ve birden çok boşluk temizliği
    table = str.maketrans({c: " " for c in string.punctuation})
    text = text.translate(table)
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def extract_brand_from_caption(caption: str, brands: List[str]) -> str:
    """
    Üretilen caption'dan marka ismi çıkar.
    1) "brand <NAME>" kalıbını yakalamaya çalış
    2) Normalizasyon sonrası doğrudan eşleşme
    3) Alt dize içeren en uzun eşleşme
    4) difflib ile en yüksek benzerlikli marka
    """
    norm_caption = normalize_text(caption)

    # 1) "brand" sonrası
    if "brand " in norm_caption:
        after = norm_caption.split("brand ", 1)[1]
        # ilk cümle sonuna kadar al
        after = after.split(".")[0].strip()
        # bazı durumlarda cümle sonu olmayabilir, tamamını kullan
        candidate = after
        if candidate:
            # Doğrudan eşleşme/alt dize kontrolü
            norm_brands = [normalize_text(b) for b in brands]
            if candidate in norm_brands:
                return brands[norm_brands.index(candidate)]

    # 2) Doğrudan eşleşme
    norm_brands = [normalize_text(b) for b in brands]
    for b_norm, b in zip(norm_brands, brands):
        if b_norm == norm_caption:
            return b

    # 3) Alt dize içerme (iki yönlü)
    substring_hits = []
    for b_norm, b in zip(norm_brands, brands):
        if b_norm in norm_caption or norm_caption in b_norm:
            substring_hits.append((len(b_norm), b))
    if substring_hits:
        # en uzun eşleşmeyi al
        substring_hits.sort(reverse=True)
        return substring_hits[0][1]

    # 4) difflib benzerlik skoru
    best_brand = None
    best_score = -1.0
    for b_norm, b in zip(norm_brands, brands):
        score = difflib.SequenceMatcher(None, norm_caption, b_norm).ratio()
        if score > best_score:
            best_score = score
            best_brand = b
    return best_brand or brands[0]


@dataclass
class EvalResult:
    accuracy: float
    confusion_matrix: np.ndarray
    report_text: str


def evaluate(model_dir: str, split_dir: str, limit: int = 0, max_new_tokens: int = 16, num_beams: int = 1) -> EvalResult:
    device = detect_device()
    print(f"Cihaz: {device}")

    # İnfer: processor model klasöründe yoksa base modelden yükle
    try:
        processor = BlipProcessor.from_pretrained(model_dir)
    except Exception:
        print("Uyarı: Processor model klasöründe bulunamadı, base modelden yükleniyor.")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    brands, samples = list_brands_and_images(split_dir)
    print(f"Sınıf sayısı: {len(brands)}  |  Örnek sayısı: {len(samples)}")

    if limit > 0:
        samples = samples[:limit]

    y_true: List[int] = []
    y_pred: List[int] = []
    rows = []

    with torch.inference_mode():
        for image_path, true_brand in tqdm(samples, desc="Değerlendiriliyor"):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Atlanıyor (okuma hatası): {image_path} -> {e}")
                continue

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            pred_brand = extract_brand_from_caption(caption, brands)

            y_true.append(brands.index(true_brand))
            y_pred.append(brands.index(pred_brand))

            rows.append(
                {
                    "file_name": image_path,
                    "true_brand": true_brand,
                    "pred_caption": caption,
                    "pred_brand": pred_brand,
                    "correct": pred_brand == true_brand,
                }
            )

    # Skorlar ve rapor
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(brands))))
    report = classification_report(y_true, y_pred, target_names=brands, digits=4)

    # Kayıtlar
    pd.DataFrame(rows).to_csv("evaluation_predictions_test_split.csv", index=False)
    with open("evaluation_report_test_split.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # CM görselleştirme
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=brands,
            yticklabels=brands,
            cbar=True,
        )
        plt.title("Confusion Matrix Test Split")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("confusion_matrix_test_split.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"Karmaşıklık matrisi görselleştirmesi başarısız: {e}")

    print(f"Accuracy: {accuracy:.4f}")
    print("Detaylı rapor 'evaluation_report_test_split.txt' dosyasına yazıldı.")
    print("Tüm tahminler 'evaluation_predictions_test_split.csv' dosyasına kaydedildi.")
    print("Karmaşıklık matrisi 'confusion_matrix_test_split.png' olarak kaydedildi.")

    # Markdown özet dosyası
    try:
        with open("evaluation_summary_test_split.md", "w", encoding="utf-8") as f:
            f.write("### Evaluation Summary\n\n")
            f.write(f"- **Model Dir**: `{model_dir}`\n")
            f.write(f"- **Split Dir**: `{split_dir}`\n")
            f.write(f"- **Num Samples**: {len(samples)}\n")
            f.write(f"- **Accuracy**: {accuracy:.4f}\n\n")
            f.write("#### Classification Report\n\n")
            f.write("```\n")
            f.write(report)
            f.write("\n``""\n\n")
            f.write("Confusion matrix görseli: `confusion_matrix_test_split.png`\n")
    except Exception as e:
        print(f"Markdown rapor yazılamadı: {e}")

    return EvalResult(accuracy=accuracy, confusion_matrix=cm, report_text=report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLIP caption fine-tune model evaluation for car brand classification")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./car_brand_model_final",
        help="Eğitilen modelin klasörü",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="Car Brand Classification Dataset/val",
        help="Değerlendirilecek split klasörü (örn. val ya da test)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Hızlı test için örnek sayısı sınırı (0=hepsi)")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Generate için max_new_tokens")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search beam sayısı (1=greedy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_dir=args.model_dir,
        split_dir=args.split_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )


