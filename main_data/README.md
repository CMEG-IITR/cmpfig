---
license: cc-by-nc-4.0
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - en
tags:
  - materials-science
  - compound-figures
  - scientific-figures
  - panel-detection
  - microscopy
  - figure-captioning
pretty_name: MatSciFig
---

# MatSciFig

**MatSciFig** is a large-scale dataset of panel-level crops from scientific compound figures in materials science literature, paired with panel-specific subcaptions and extended summaries extracted from the source papers.

Compound figures — multi-panel images labeled A, B, C… — are the dominant figure type in scientific publications. MatSciFig breaks these figures into individual panels and links each panel to its corresponding textual description, enabling fine-grained vision-language research on scientific imagery.

---

## Dataset Statistics

| | Count |
|---|---|
| Source figures | ~170,000+ |
| Panels with full metadata | **391,606** |

---

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `image` | bytes | Panel crop in original format |
| `image_id` | string | Parent figure identifier |
| `panel_suffix` | string | Panel label (A, B, C … or single) |
| `visualization_category` | string | High-level category (e.g. Microscopy, Generic Plot, Photograph) |
| `visualization_subtype` | string | Fine-grained subtype (e.g. SEM, Contour Plot, XRD Pattern) |
| `subcaption` | string | Panel-level caption from the source paper |
| `summary` | string | Extended description of the panel content |

---

## Example

```python
from datasets import load_dataset

ds = load_dataset("subham2507/MatSciFig")
sample = ds["train"][0]

print(sample["panel_suffix"])            # "A"
print(sample["visualization_category"])  # "Microscopy"
print(sample["visualization_subtype"])   # "SEM"
print(sample["subcaption"])              # "SEM image of Fe2O3 nanorods..."
print(sample["summary"])                 # "The SEM image reveals..."
sample["image"].show()                   # displays the panel crop
```

---

## Data Collection

- Compound figures collected from open-access materials science publications
- Panels detected and segmented using a custom-trained YOLO-based compound figure detector
- Subcaptions and summaries extracted and aligned from paper text using automated parsing
- Only panels with complete metadata are included

---

## Intended Use

- Scientific figure understanding and captioning
- Vision-language model training and evaluation
- Materials science multimodal research
- Panel classification by visualization type

---

## License

This dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

- ✅ Free to use for research and non-commercial purposes
- ✅ Share and adapt with attribution
- ❌ Commercial use is not permitted

---

## Citation

If you use MatSciFig in your research, please cite:

```bibtex

```
