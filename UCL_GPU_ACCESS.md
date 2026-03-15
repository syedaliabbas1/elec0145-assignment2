# ELEC0145 Assignment 2 — Task 1 Plan
## Image Classification: Dataset, CNN, Training & Evaluation

---

## Overview

| Item | Detail |
|------|--------|
| Goal | Train a CNN to classify 10 rim types from images |
| Classes | 10 (one per car model + year) |
| Images | 150 total (15 per class) |
| Model | ResNet18 (primary) + EfficientNetB0 (comparison) |
| Framework | PyTorch + torchvision |
| Environment | VS Code + UCL SSH + RTX 3090 (lab 105) + `.ipynb` |
| Page limit | 8 pages in report |

---

## 1. Dataset — 10 Rim Classes

| Class ID | Label | Rim Style | Why chosen |
|----------|-------|-----------|------------|
| 01 | BMW_M3_2023 | 5-double-spoke | Iconic, highly distinctive |
| 02 | Porsche_911_2024 | Centre-lock turbine | Very different from others |
| 03 | Mercedes_AMG_GT_2023 | Multi-spoke turbine | Dense spoke pattern |
| 04 | Audi_RS6_2023 | 5-V-spoke | Bold V shape |
| 05 | Ferrari_SF90_2023 | Mesh / diamond-cut | Fine mesh, unique texture |
| 06 | Ford_Mustang_GT500_2020 | 10-spoke | High spoke count |
| 07 | Lamborghini_Urus_2022 | Y-spoke | Aggressive Y pattern |
| 08 | Honda_CivicTypeR_2023 | Sharp multi-spoke | Angular, distinctive finish |
| 09 | Toyota_GR_Yaris_2022 | Simple 5-spoke | Plain contrast to complex ones |
| 10 | RangeRover_Sport_2023 | Split-spoke SUV | Wide, split design |

> Classes were chosen to maximise visual distinctiveness across spoke count, pattern type, and brand — reducing inter-class similarity and improving classifier performance.

---

## 2. Image Collection

### Sources (in order of priority)

| Source | Why | Best for |
|--------|-----|---------|
| Manufacturer configurator websites | High quality, rim-focused, clean background | White background isolated shots |
| Wheel retailer sites (blackcircles.com, alloywheels.com) | Rim fills entire frame, no cropping needed | Clean isolated rim images |
| Car review sites (autocar.co.uk, evo.co.uk, topgear.com) | Real-world variation, different lighting/angles | Varied realistic shots |
| Enthusiast forums (r/BMW, r/Porsche etc.) | Owner photos, genuine variation, partial occlusion | Realistic industrial-style variation |
| Google Images (filtered: Large size) | Quick top-up for remaining slots | Mixed use |

### Per-class image checklist (15 images each)

For each class, aim for this mix:

| Image type | Count | Description |
|-----------|-------|-------------|
| Isolated rim, white/neutral background | 4–5 | From configurator or retailer site |
| Close-up on car, straight-on angle | 3–4 | Rim fills frame, car body minimal |
| Angled / 3/4 view | 3 | Shows depth and spoke structure |
| Different lighting (outdoor, shadow, bright) | 2–3 | Adds real-world variation |
| Partial occlusion or distance shot | 1–2 | Brief specifically asks for this |

### Folder structure

```
dataset/
├── train/
│   ├── 01_BMW_M3_2023/
│   ├── 02_Porsche_911_2024/
│   ├── ...
│   └── 10_RangeRover_Sport_2023/
├── val/
│   ├── 01_BMW_M3_2023/
│   └── ...
└── test/
    ├── 01_BMW_M3_2023/
    └── ...
```

> Use `torchvision.datasets.ImageFolder` — it reads this structure automatically, using folder names as class labels.

### Preprocessing applied (state this in the report)

- **Cropping strategy:** Images were selected so the rim occupies at least 70% of the frame. Images where the rim was too small were rejected at collection time rather than cropped programmatically — this avoids introducing artefacts.
- **Resize:** All images resized to 224×224 px (standard ResNet input size), applied via `transforms.Resize((224, 224))` in the PyTorch transform pipeline.
- **Normalisation:** Pixel values normalised using ImageNet mean and std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — required because ResNet18 was pretrained on ImageNet with these values.

---

## 3. Data Split Strategy

| Split | Proportion | Count | Purpose |
|-------|-----------|-------|---------|
| Train | 70% | 105 images | Model learning |
| Validation | 15% | 22 images | Hyperparameter tuning, early stopping |
| Test | 15% | 23 images | Final unbiased performance evaluation |

### Justification (write this in the report)

The dataset is small (150 images), so the split must balance having enough training data against having a meaningful test set. A 70/15/15 split gives 105 training images — sufficient for fine-tuning a pretrained model — while the held-out test set remains completely unseen until final evaluation, ensuring unbiased results. Cross-validation was considered but ruled out as the additional computational cost is unnecessary given that transfer learning significantly reduces overfitting risk on small datasets.

> **Important:** Perform the split **before** any training. The test set must never influence any training or tuning decision.

---

## 4. Data Augmentation

Applied **only to the training set** — never to validation or test.

```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Justification (write this in the report)

| Augmentation | Justification |
|-------------|---------------|
| Random horizontal flip | Rims are rotationally symmetric — a flipped rim is still the same class |
| Random rotation (±15°) | Reflects real-world variation in rim orientation in the storage area |
| Colour jitter | Reflects variation in lighting conditions in the factory environment |
| Random grayscale | Encourages model to learn shape/texture rather than colour |

---

## 5. Model Architecture

### Primary model: ResNet18 (fine-tuned)

**Why ResNet18:**
- 11M parameters — appropriate scale for a 150-image dataset
- Residual connections prevent vanishing gradients during fine-tuning
- Proven to outperform larger models (ResNet50, VGG) on small datasets
- Well documented — easy to describe reproducibly in the report

**Architecture modification:**

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer for 10-class output
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 10)
)
```

**Training strategy (two-phase fine-tuning):**

| Phase | Layers unfrozen | Epochs | Learning rate | Purpose |
|-------|----------------|--------|---------------|---------|
| Phase 1 | Final FC layer only | 5 | 1e-3 | Warm up the new head |
| Phase 2 | All layers | 15–20 | 1e-4 | Fine-tune full network |

### Comparison model: EfficientNetB0

Run in parallel to provide a comparison in the evaluation section.

```python
model_eff = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = model_eff.classifier[1].in_features
model_eff.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 10)
)
```

---

## 6. Training Setup

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

criterion = nn.CrossEntropyLoss()

# Phase 1 — head only
optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=1e-3)

# Phase 2 — full network
optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Learning rate scheduler (robustness technique — mention in report)
scheduler = CosineAnnealingLR(optimizer_phase2, T_max=20, eta_min=1e-6)
```

### Key hyperparameters

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Batch size | 16 | Small dataset — larger batches would mean very few updates per epoch |
| Phase 1 epochs | 5 | Just enough to initialise the head before unfreezing |
| Phase 2 epochs | 15–20 | Monitor val loss — stop early if no improvement for 5 epochs |
| Optimizer | Adam | Adaptive learning rate, converges faster on small datasets than SGD |
| Weight decay | 1e-4 | L2 regularisation to reduce overfitting |
| Dropout | 0.4 | Regularisation in the classifier head |
| LR scheduler | CosineAnnealingLR | Smoothly decays LR — robustness technique required by brief |

### Overfitting mitigation strategy (write this in the report)

| Technique | Where applied |
|-----------|--------------|
| Transfer learning | Pretrained ImageNet weights — main defence against overfitting |
| Data augmentation | Training set only |
| Dropout (p=0.4) | Classifier head |
| Weight decay (1e-4) | Optimizer |
| Early stopping | Stop if val loss doesn't improve for 5 consecutive epochs |
| LR scheduling | CosineAnnealingLR across Phase 2 |

---

## 7. Evaluation

### Metrics to report

| Metric | How to produce | Why required |
|--------|---------------|-------------|
| Overall test accuracy | `correct / total` on test set | Brief explicitly requires it |
| Confusion matrix | `sklearn.metrics.confusion_matrix` | Brief explicitly requires it |
| Training/val loss curves | Plot per epoch using matplotlib | Evidence of training behaviour |
| Training/val accuracy curves | Plot per epoch using matplotlib | Evidence of overfitting or lack thereof |
| Per-class accuracy | From confusion matrix diagonal | Useful for discussion |

### Confusion matrix code

```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# After evaluating on test set:
cm = confusion_matrix(all_labels, all_predictions)
class_names = ['BMW_M3', 'Porsche_911', 'Mercedes_AMG', 'Audi_RS6',
               'Ferrari_SF90', 'Ford_Mustang', 'Lambo_Urus',
               'Honda_CTR', 'Toyota_GRY', 'RangeRover']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix — ResNet18')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
```

> Save all plots as `.png` at 150 dpi minimum — good enough for the report.

### Model comparison table (put this in the report)

| Model | Test Accuracy | Training time | Parameters | Final choice |
|-------|-------------|--------------|------------|-------------|
| ResNet18 | TBD | TBD | 11M | ✅ Yes |
| EfficientNetB0 | TBD | TBD | 5.3M | ❌ No |

Fill in after running both models. Justify the final choice based on accuracy, training stability, and complexity.

---

## 8. Industrial Reflection (required for top marks)

The brief awards marks for a short industrial reflection. Include this at the end of Task 1 in the report:

**Points to cover:**

- **Deployment assumption:** In the factory, images would be captured by a fixed overhead camera at the deburring station. The image quality and lighting would be more controlled than the varied training images — this likely improves real-world accuracy.
- **Likely failure mode:** Rims that are heavily occluded by debris or oil residue from casting may be misclassified. A confidence threshold should be implemented — if the model outputs confidence below ~80%, the system flags for human review rather than selecting a programme automatically.
- **Domain gap:** The model was trained on studio/marketing images but will be deployed on industrial camera images. This domain gap could reduce accuracy — fine-tuning on a small set of real factory images would be recommended before deployment.
- **Class imbalance in deployment:** In a real factory, some rim types are produced more frequently than others. The uniform 15-images-per-class training set does not reflect this — production retraining should use a representative distribution.

---

## 9. Report Writing Guide (Task 1 — 8 pages)

| Subsection | Approx pages | Key content |
|-----------|-------------|------------|
| Dataset & preprocessing | 1.5 | Class definitions table, image sources, preprocessing steps, example images (show 1–2 per class) |
| Model architecture | 1.5 | Why transfer learning, why ResNet18, architecture diagram or description, two-phase fine-tuning strategy |
| Training procedure | 1.5 | Hyperparameters table, augmentation justification, overfitting mitigation techniques, LR scheduler |
| Results & evaluation | 2.5 | Accuracy, confusion matrix with discussion, training curves, model comparison table, which classes are confused and why |
| Industrial reflection | 0.5 | Deployment assumptions, failure modes, domain gap |

> Stay within 8 pages. Cut prose, keep tables and figures — they communicate more per page.

---

## 10. Checklist Before Submitting Task 1

- [ ] 150 images collected, 15 per class, in labelled folders
- [ ] Train/val/test split done before any training
- [ ] ResNet18 trained, Phase 1 + Phase 2 complete
- [ ] EfficientNetB0 trained for comparison
- [ ] Test accuracy reported for both models
- [ ] Confusion matrix generated and saved
- [ ] Training/val loss and accuracy curves generated and saved
- [ ] Overfitting mitigation discussed in report
- [ ] LR scheduler mentioned and justified
- [ ] Industrial reflection included
- [ ] All figures have captions, axis labels, legends
- [ ] Figures readable in black and white
- [ ] Report section is within 8 pages
- [ ] Written reproducibly — another engineer could follow it

---

*ELEC0145 Assignment 2 | Task 1 Plan | Last updated: 9 March 2026*