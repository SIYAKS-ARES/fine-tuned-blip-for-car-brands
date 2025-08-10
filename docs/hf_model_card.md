---
license: apache-2.0
tags:
  - transformers
  - blip
  - image-to-text
  - vision-language
library_name: transformers
pipeline_tag: image-to-text
base_model: Salesforce/blip-image-captioning-base
metrics:
  - accuracy
language:
  - en
---

# Fine-tuned BLIP for Car Brands

This repository hosts a fine-tuned BLIP image captioning model specialized for car brand identification from images via captions. The model is trained from `Salesforce/blip-image-captioning-base` and produces captions like: "A photo of a car from the brand Audi." Then, a simple brand-extraction layer maps captions to the final brand prediction.

- Model page: `https://huggingface.co/SIYAKSARES/fine-tuned-blip-for-car-brands`
- Base model: `Salesforce/blip-image-captioning-base`
- Task: Image-to-Text (captioning), used for brand classification
- License: Apache-2.0

## Results (Summary)

- Validation Accuracy: 81.09%
- Test Accuracy: 79.68%

Dataset: Kaggle Car Brand Classification Dataset (33 brands; class-balanced splits)

## Usage

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

## Notes

- The model is optimized for generating brand-explicit captions. For classification, extract the brand name from the caption string.
- For faster inference, consider `num_beams=1` (greedy decoding).

## Citation

If you use this model, please also cite the BLIP paper:

```
@article{blip2022,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  journal={ICML},
  year={2022}
}
```



