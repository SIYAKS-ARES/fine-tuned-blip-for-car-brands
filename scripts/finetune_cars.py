import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Trainer,
    TrainingArguments,
)


class CarCaptionDataset(Dataset):
    """
    `metadata.jsonl` dosyasından kayıtları okur ve BLIP işlemcisinden (processor)
    geçerek Trainer için uygun tensörleri döndürür.
    Beklenen alanlar: `file_name`, `text`
    """

    def __init__(self, jsonl_path: str, processor: BlipProcessor) -> None:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Metadata dosyası bulunamadı: '{jsonl_path}'. Önce prepare_dataset.py çalıştırın."
            )

        self.data = pd.read_json(path_or_buf=jsonl_path, lines=True)
        expected_columns = {"file_name", "text"}
        missing = expected_columns.difference(self.data.columns)
        if missing:
            raise ValueError(f"Metadata dosyasında eksik sütun(lar): {missing}")

        self.processor = processor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:  # type: ignore[override]
        item = self.data.iloc[idx]
        image_path: str = item["file_name"]
        caption: str = item["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # Kayıp görselleri atla
            print(f"Uyarı: Görsel bulunamadı, atlanıyor → {image_path}")
            return None

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        # batch dimension'u kaldır
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}

        # Labels: input_ids'in bir kopyası, pad token'larını -100 yap (loss dışında tutmak için)
        labels = encoding["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        encoding["labels"] = labels

        return encoding


def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """None dönen örnekleri ayıkla ve varsayılan collate uygula."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main() -> None:
    # 1) Model ve işlemci
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 2) Veri kümesi
    train_dataset = CarCaptionDataset(jsonl_path="metadata.jsonl", processor=processor)

    # 3) Eğitim argümanları
    training_args = TrainingArguments(
        output_dir="./car_brand_model",
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # MPS için daha küçük batch
        gradient_accumulation_steps=8,   # Efektif batch'i artırmak için biriktirme
        save_total_limit=2,
        save_steps=1000,
        logging_steps=100,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=["none"],  # istenirse ["tensorboard"] olarak değiştirilebilir
        dataloader_pin_memory=False,  # MPS uyarısını önlemek için
        dataloader_num_workers=0,     # MPS ile daha stabil
        use_mps_device=True,
        gradient_checkpointing=True,
    )

    # Gradient checkpointing kullanırken cache kapatılması önerilir
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # 4) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # 5) Eğitim ve kaydetme
    trainer.train()
    out_dir = "./car_brand_model_final"
    trainer.save_model(out_dir)
    # Değerlendirme sırasında işlemci dosyaları da gerekli olabileceği için kaydediyoruz
    try:
        processor.save_pretrained(out_dir)
    except Exception as e:
        print(f"Uyarı: Processor kaydedilemedi: {e}")
    print(f"Fine-tuning tamamlandı ve model kaydedildi → {out_dir}")


if __name__ == "__main__":
    main()


