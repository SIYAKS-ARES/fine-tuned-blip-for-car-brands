# Fine-tuned BLIP for Car Brands

Bu repo, BLIP tabanlı bir görüntü altyazılama (image-to-text) modelini araba markası tanıma (caption üzerinden marka çıkarımı) amacıyla ince ayar (fine-tuning) edilmiş haliyle içerir.

- Hugging Face model sayfası: [SIYAKSARES/fine-tuned-blip-for-car-brands](https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands)
- Belgeler: `docs/README.md`

## Hızlı Kullanım (HF Hub)

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

## Sonuç Özeti

- Validation Accuracy: 81.09%
- Test Accuracy: 79.68%

Detaylar ve kurulum/kullanım adımları için `docs/` klasörüne bakın.


