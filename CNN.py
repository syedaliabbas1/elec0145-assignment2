import os
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — no display needed (required for UCL SSH terminal)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import confusion_matrix, classification_report

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# =============================================================================
# 1. Data Loading
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

DATA_ROOT = 'dataset'
BATCH_SIZE = 16

train_dataset = ImageFolder(os.path.join(DATA_ROOT, 'train'), transform=train_transforms)
val_dataset   = ImageFolder(os.path.join(DATA_ROOT, 'val'),   transform=val_test_transforms)
test_dataset  = ImageFolder(os.path.join(DATA_ROOT, 'test'),  transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

CLASS_NAMES = train_dataset.classes
SHORT_NAMES = ['BMW_M3', 'Porsche_911', 'Mercedes_AMG_C63', 'Audi_RS6',
               'Ferrari_SF90', 'Maybach_S580', 'Lambo_Urus',
               'AlfaRomeo_Giulia', 'RangeRover', 'Tesla_ModelS']
NUM_CLASSES = len(CLASS_NAMES)

print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')
print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

# =============================================================================
# 2. Training Utilities
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def run_phase(model, train_loader, val_loader, optimizer, criterion,
              n_epochs, scheduler=None, patience=5, label='Phase'):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)

        if scheduler:
            scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        elapsed = time.time() - t0
        print(f'[{label}] Epoch {epoch:02d}/{n_epochs} '
              f'| train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'| val_loss={vl_loss:.4f} acc={vl_acc:.3f} '
              f'| {elapsed:.1f}s', flush=True)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'  Early stopping at epoch {epoch} (no val improvement for {patience} epochs)', flush=True)
                break

    model.load_state_dict(best_weights)
    return history

# =============================================================================
# 3. Plot Utilities
# =============================================================================

def plot_curves(hist_p1, hist_p2, model_name, save_path):
    train_loss = hist_p1['train_loss'] + hist_p2['train_loss']
    val_loss   = hist_p1['val_loss']   + hist_p2['val_loss']
    train_acc  = hist_p1['train_acc']  + hist_p2['train_acc']
    val_acc    = hist_p1['val_acc']    + hist_p2['val_acc']
    epochs = range(1, len(train_loss) + 1)
    p1_end = len(hist_p1['train_loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, 'k-',  label='Train loss')
    ax1.plot(epochs, val_loss,   'k--', label='Val loss')
    ax1.axvline(p1_end + 0.5, color='grey', linestyle=':', linewidth=1, label='Phase 2 start')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2.plot(epochs, [a * 100 for a in train_acc], 'k-',  label='Train acc')
    ax2.plot(epochs, [a * 100 for a in val_acc],   'k--', label='Val acc')
    ax2.axvline(p1_end + 0.5, color='grey', linestyle=':', linewidth=1, label='Phase 2 start')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f'{model_name} Training Curves (Phase 1 + Phase 2)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}', flush=True)


def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    acc = (labels == preds).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title(f'{title}\nTest Accuracy: {acc:.1f}%', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}', flush=True)
    return cm, acc

# =============================================================================
# 4. ResNet18 — Two-Phase Fine-Tuning
# =============================================================================
print('\n' + '='*60)
print('RESNET18')
print('='*60)

resnet = models.resnet18(weights='IMAGENET1K_V1')

for param in resnet.parameters():
    param.requires_grad = False

num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, NUM_CLASSES)
)

resnet = resnet.to(device)
criterion = nn.CrossEntropyLoss()

print('ResNet18 ready. Trainable params (Phase 1):',
      sum(p.numel() for p in resnet.parameters() if p.requires_grad))

# Phase 1
P1_EPOCHS = 5
optimizer_p1 = optim.Adam(resnet.fc.parameters(), lr=1e-3)
print('=== ResNet18 Phase 1 (FC head only) ===')
hist_p1 = run_phase(resnet, train_loader, val_loader,
                    optimizer_p1, criterion,
                    n_epochs=P1_EPOCHS, label='P1')

# Phase 2
for name, param in resnet.named_parameters():
    param.requires_grad = ('layer4' in name or 'fc' in name)

P2_EPOCHS = 20
optimizer_p2 = optim.Adam(
    filter(lambda p: p.requires_grad, resnet.parameters()),
    lr=1e-4, weight_decay=1e-4
)
scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=P2_EPOCHS, eta_min=1e-6)

print('Trainable params (Phase 2):',
      sum(p.numel() for p in resnet.parameters() if p.requires_grad))
print('=== ResNet18 Phase 2 (layer4 + FC) ===')
hist_p2 = run_phase(resnet, train_loader, val_loader,
                    optimizer_p2, criterion,
                    n_epochs=P2_EPOCHS, scheduler=scheduler_p2,
                    patience=5, label='P2')

torch.save(resnet.state_dict(), 'models/resnet18_best.pth')
print('Saved: models/resnet18_best.pth')

# Curves
plot_curves(hist_p1, hist_p2, 'ResNet18', 'results/resnet18_curves.png')

# Evaluation
rn_labels, rn_preds = get_predictions(resnet, test_loader)
rn_cm, rn_acc = plot_confusion_matrix(
    rn_labels, rn_preds, SHORT_NAMES,
    'Confusion Matrix — ResNet18',
    'results/resnet18_confusion_matrix.png'
)
print(f'\nResNet18 Test Accuracy: {rn_acc:.1f}%')

per_class_acc = rn_cm.diagonal() / rn_cm.sum(axis=1)
print('ResNet18 per-class accuracy:')
for name, acc in zip(SHORT_NAMES, per_class_acc):
    print(f'  {name:20s}: {acc*100:.1f}%')
print('\nClassification Report:')
print(classification_report(rn_labels, rn_preds, target_names=SHORT_NAMES))

# =============================================================================
# 5. EfficientNet-B0 — Two-Phase Fine-Tuning
# =============================================================================
print('\n' + '='*60)
print('EFFICIENTNET-B0')
print('='*60)

effnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

for param in effnet.parameters():
    param.requires_grad = False

num_features_eff = effnet.classifier[1].in_features
effnet.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features_eff, NUM_CLASSES)
)

effnet = effnet.to(device)
print('EfficientNet-B0 ready. Trainable params (Phase 1):',
      sum(p.numel() for p in effnet.parameters() if p.requires_grad))

# Phase 1
optimizer_eff_p1 = optim.Adam(effnet.classifier.parameters(), lr=1e-3)
print('=== EfficientNet-B0 Phase 1 (classifier head only) ===')
hist_eff_p1 = run_phase(effnet, train_loader, val_loader,
                        optimizer_eff_p1, criterion,
                        n_epochs=P1_EPOCHS, label='Eff-P1')

# Phase 2
for name, param in effnet.named_parameters():
    param.requires_grad = ('features.7' in name or 'features.8' in name or 'classifier' in name)

optimizer_eff_p2 = optim.Adam(
    filter(lambda p: p.requires_grad, effnet.parameters()),
    lr=1e-4, weight_decay=1e-4
)
scheduler_eff_p2 = CosineAnnealingLR(optimizer_eff_p2, T_max=P2_EPOCHS, eta_min=1e-6)

print('Trainable params (Phase 2):',
      sum(p.numel() for p in effnet.parameters() if p.requires_grad))
print('=== EfficientNet-B0 Phase 2 (features[7,8] + classifier) ===')
hist_eff_p2 = run_phase(effnet, train_loader, val_loader,
                        optimizer_eff_p2, criterion,
                        n_epochs=P2_EPOCHS, scheduler=scheduler_eff_p2,
                        patience=5, label='Eff-P2')

torch.save(effnet.state_dict(), 'models/efficientnet_b0_best.pth')
print('Saved: models/efficientnet_b0_best.pth')

# Curves
plot_curves(hist_eff_p1, hist_eff_p2, 'EfficientNet-B0', 'results/efficientnet_curves.png')

# Evaluation
eff_labels, eff_preds = get_predictions(effnet, test_loader)
eff_cm, eff_acc = plot_confusion_matrix(
    eff_labels, eff_preds, SHORT_NAMES,
    'Confusion Matrix — EfficientNet-B0',
    'results/efficientnet_confusion_matrix.png'
)
print(f'\nEfficientNet-B0 Test Accuracy: {eff_acc:.1f}%')

per_class_acc_eff = eff_cm.diagonal() / eff_cm.sum(axis=1)
print('EfficientNet-B0 per-class accuracy:')
for name, acc in zip(SHORT_NAMES, per_class_acc_eff):
    print(f'  {name:20s}: {acc*100:.1f}%')
print('\nClassification Report:')
print(classification_report(eff_labels, eff_preds, target_names=SHORT_NAMES))

# =============================================================================
# 6. Model Comparison
# =============================================================================
rn_params  = sum(p.numel() for p in resnet.parameters()) / 1e6
eff_params = sum(p.numel() for p in effnet.parameters()) / 1e6

print('\n' + '='*60)
print(f"{'Model':<20} {'Test Acc':>10} {'Params':>10}")
print('-'*60)
print(f"{'ResNet18':<20} {rn_acc:>9.1f}% {rn_params:>9.1f}M")
print(f"{'EfficientNet-B0':<20} {eff_acc:>9.1f}% {eff_params:>9.1f}M")
print('='*60)

winner = 'ResNet18' if rn_acc >= eff_acc else 'EfficientNet-B0'
print(f'\nHigher test accuracy: {winner}')

# =============================================================================
# 7. Sample Predictions
# =============================================================================
best_model = resnet if rn_acc >= eff_acc else effnet
best_model.eval()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for class_idx, ax in enumerate(axes.flatten()):
    class_indices = [i for i, (_, label) in enumerate(test_dataset.samples) if label == class_idx]
    img_path, true_label = test_dataset.samples[class_indices[0]]

    img = Image.open(img_path).convert('RGB')
    tensor = val_test_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = best_model(tensor)
        pred_label = out.argmax(1).item()
        confidence = torch.softmax(out, dim=1)[0, pred_label].item() * 100

    ax.imshow(img)
    ax.axis('off')
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'True: {SHORT_NAMES[true_label]}\nPred: {SHORT_NAMES[pred_label]} ({confidence:.0f}%)',
                 fontsize=7, color=color)

model_label = 'ResNet18' if rn_acc >= eff_acc else 'EfficientNet-B0'
plt.suptitle(f'Sample Test Predictions — {model_label} (green=correct, red=wrong)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150)
plt.close()
print('Saved: results/sample_predictions.png')

print('\nAll done. Results saved to results/')
