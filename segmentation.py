
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from PIL import Image


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_LIMIT = 300
VALIDATION_LIMIT = 60

DEVICE = torch.device("cpu")

random.seed(42)
torch.manual_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Device:", DEVICE)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

class PetSegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, mask = self.dataset[index]

        # Resize original image
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Resize segmentation mask using nearest neighbour
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        # Convert image into tensor
        image = transforms.ToTensor()(image)

        # Convert mask into integer labels
        mask = np.array(mask, dtype=np.int64)

        # Original labels: 1, 2, 3
        # New labels: 0, 1, 2
        mask = np.clip(mask, 1, 3) - 1

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


print("\nDownloading/loading Oxford-IIIT Pet dataset...")

original_dataset = OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="segmentation",
    download=True
)

dataset = PetSegmentationDataset(original_dataset)

# Select a small subset for CPU training
train_size = min(TRAIN_LIMIT, len(dataset) - VALIDATION_LIMIT)
validation_size = min(VALIDATION_LIMIT, len(dataset) - train_size)

remaining_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, _ = random_split(
    dataset,
    [train_size, validation_size, remaining_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# --------------------------------------------------
# 3. DOUBLE CONVOLUTION BLOCK
# --------------------------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# 4. U-NET ARCHITECTURE
# --------------------------------------------------

class UNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(3, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.decoder3 = DoubleConv(256 + 128, 128)
        self.decoder2 = DoubleConv(128 + 64, 64)
        self.decoder1 = DoubleConv(64 + 32, 32)

        self.pool = nn.MaxPool2d(2)

        # Final output layer
        self.output_layer = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder path with skip connections
        d3 = F.interpolate(
            b,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return self.output_layer(d1)


# --------------------------------------------------
# 5. CREATE MODEL
# --------------------------------------------------

model = UNet(num_classes=3).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nU-Net model created successfully!")


# --------------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    total_pixels = 0
    correct_pixels = 0

    intersection = torch.zeros(3)
    union = torch.zeros(3)

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct_pixels += (
                predictions == masks
            ).sum().item()

            total_pixels += masks.numel()

            for class_id in range(3):

                pred_class = predictions == class_id
                true_class = masks == class_id

                intersection[class_id] += (
                    pred_class & true_class
                ).sum().item()

                union[class_id] += (
                    pred_class | true_class
                ).sum().item()

    pixel_accuracy = correct_pixels / total_pixels

    iou = intersection / (union + 1e-8)

    mean_iou = iou.mean().item()

    return pixel_accuracy, mean_iou


# --------------------------------------------------
# 7. TRAINING
# --------------------------------------------------

train_losses = []
validation_accuracies = []
validation_ious = []

best_accuracy = 0.0

print("\nStarting U-Net training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    accuracy, mean_iou = evaluate(
        model,
        validation_loader
    )

    train_losses.append(epoch_loss)
    validation_accuracies.append(accuracy)
    validation_ious.append(mean_iou)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Pixel Accuracy: {accuracy:.4f} "
        f"Mean IoU: {mean_iou:.4f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "models/best_unet.pth"
        )

print("\nTraining completed!")


# --------------------------------------------------
# 8. SAVE TRAINING RESULTS
# --------------------------------------------------

with open(
    "results/training_metrics.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Training Loss",
        "Pixel Accuracy",
        "Mean IoU"
    ])

    for i in range(EPOCHS):

        writer.writerow([
            i + 1,
            train_losses[i],
            validation_accuracies[i],
            validation_ious[i]
        ])


# --------------------------------------------------
# 9. PLOT TRAINING GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U-Net Training Loss")

plt.legend()
plt.grid(True)

plt.savefig("results/training_graph.png")
plt.close()


# --------------------------------------------------
# 10. VISUALIZE SEGMENTATION
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "models/best_unet.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

image, true_mask = validation_dataset[0]

with torch.no_grad():

    prediction = model(
        image.unsqueeze(0).to(DEVICE)
    )

    predicted_mask = torch.argmax(
        prediction,
        dim=1
    ).squeeze(0).cpu().numpy()


# Display original image, actual mask and predicted mask

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(true_mask.numpy(), cmap="viridis", vmin=0, vmax=2)
plt.title("Actual Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap="viridis", vmin=0, vmax=2)
plt.title("U-Net Prediction")
plt.axis("off")

plt.tight_layout()

plt.savefig("results/segmentation_output.png")
plt.show()

print("\nAll results saved successfully!")

print("\nBest Validation Pixel Accuracy:", round(best_accuracy, 4))
print("Final Validation Mean IoU:", round(validation_ious[-1], 4))

print("\nCheck the results folder for output images.")
