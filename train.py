
# ==========================================================
# PROGRAM 2: RESNET vs DENSENET vs EFFICIENTNET
# Transfer Learning and Fine-Tuning
# Dataset: CIFAR-10
# Framework: PyTorch
# Device: CPU
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import pandas as pd

import time
import os
from pathlib import Path

from torch.utils.data import DataLoader, Subset


# ==========================================================
# 1. PROJECT SETTINGS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 32
IMAGE_SIZE = 96

TRAIN_LIMIT = 2000
TEST_LIMIT = 500

HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 1

HEAD_LR = 0.001
FINE_LR = 0.0001

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("=" * 60)
print("ADVANCED CNN ARCHITECTURE COMPARISON")
print("=" * 60)

print("Device:", DEVICE)


# ==========================================================
# 2. IMAGE PREPROCESSING
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 3. FIND OR DOWNLOAD CIFAR-10 DATASET
# ==========================================================

print("\nChecking CIFAR-10 dataset...")

# Search the project and its parent folders for
# an existing CIFAR-10 download.

search_roots = [BASE_DIR, BASE_DIR.parent]

if BASE_DIR.parent.parent.exists():
    search_roots.append(BASE_DIR.parent.parent)

data_root = None

for search_root in search_roots:

    matches = list(
        search_root.rglob("cifar-10-batches-py")
    )

    if matches:
        data_root = matches[0].parent
        break

if data_root is None:

    data_root = BASE_DIR / "dataset"

    print("Existing dataset not found.")
    print("Downloading CIFAR-10 into:", data_root)

else:

    print("Existing dataset found:", data_root)


# Load the training dataset

train_dataset = torchvision.datasets.CIFAR10(
    root=str(data_root),
    train=True,
    download=True,
    transform=train_transform
)

# Load the testing dataset

test_dataset = torchvision.datasets.CIFAR10(
    root=str(data_root),
    train=False,
    download=True,
    transform=test_transform
)


# Select smaller subsets for CPU training

torch.manual_seed(42)

train_indices = torch.randperm(
    len(train_dataset)
)[:TRAIN_LIMIT].tolist()

test_indices = torch.randperm(
    len(test_dataset)
)[:TEST_LIMIT].tolist()

train_dataset = Subset(
    train_dataset, train_indices
)

test_dataset = Subset(
    test_dataset, test_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# ==========================================================
# 4. LOAD PRETRAINED CNN ARCHITECTURES
# ==========================================================

def load_model(model_name):

    print("\nLoading pretrained model:", model_name)

    if model_name == "ResNet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        model.fc = nn.Linear(
            model.fc.in_features, 10
        )

        final_block = model.layer4


    elif model_name == "DenseNet121":

        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        model.classifier = nn.Linear(
            model.classifier.in_features, 10
        )

        final_block = nn.Sequential(
            model.features.denseblock4,
            model.features.norm5
        )


    elif model_name == "EfficientNetB0":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 10
        )

        final_block = model.features[-1]


    else:

        raise ValueError("Unknown model name")


    model = model.to(DEVICE)

    return model, final_block


# ==========================================================
# 5. FREEZE PRETRAINED LAYERS
# ==========================================================

def freeze_backbone(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    # Keep the new classification layer trainable

    if isinstance(model, models.ResNet):

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif isinstance(model, models.DenseNet):

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    else:

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


# ==========================================================
# 6. UNFREEZE SELECTED LAYERS FOR FINE-TUNING
# ==========================================================

def unfreeze_final_block(model_name, model):

    if model_name == "ResNet18":

        block = model.layer4

    elif model_name == "DenseNet121":

        block = model.features.denseblock4

    else:

        block = model.features[-1]


    for parameter in block.parameters():

        parameter.requires_grad = True

    print("Final feature block unfrozen.")


# ==========================================================
# 7. MODEL EVALUATION
# ==========================================================

def evaluate_model(model, criterion):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    accuracy = 100 * correct / total

    average_loss = total_loss / len(test_loader)

    return accuracy, average_loss


# ==========================================================
# 8. TRAINING FUNCTION
# ==========================================================

def train_phase(
    model,
    optimizer,
    criterion,
    epochs,
    phase_name
):

    model.train()

    for epoch in range(epochs):

        start = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        train_accuracy = 100 * correct / total

        train_loss = running_loss / len(train_loader)

        test_accuracy, test_loss = evaluate_model(
            model, criterion
        )

        elapsed = time.time() - start

        print(
            f"{phase_name} | "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}% | "
            f"Train Loss: {train_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )


# ==========================================================
# 9. RUN ONE COMPLETE MODEL EXPERIMENT
# ==========================================================

def run_experiment(model_name):

    print("\n" + "=" * 60)

    print("STARTING:", model_name)

    print("=" * 60)

    total_start = time.time()

    model, final_block = load_model(model_name)

    criterion = nn.CrossEntropyLoss()


    # ----------------------------------------------
    # STAGE A: FEATURE EXTRACTION
    # ----------------------------------------------

    print("\nSTAGE A: FREEZING PRETRAINED LAYERS")

    freeze_backbone(model)

    optimizer = optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=HEAD_LR
    )

    train_phase(
        model,
        optimizer,
        criterion,
        HEAD_EPOCHS,
        "Feature Extraction"
    )


    # ----------------------------------------------
    # STAGE B: FINE-TUNING
    # ----------------------------------------------

    print("\nSTAGE B: FINE-TUNING")

    unfreeze_final_block(model_name, model)

    optimizer = optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=FINE_LR
    )

    train_phase(
        model,
        optimizer,
        criterion,
        FINETUNE_EPOCHS,
        "Fine-Tuning"
    )


    # ----------------------------------------------
    # FINAL EVALUATION
    # ----------------------------------------------

    final_accuracy, final_loss = evaluate_model(
        model, criterion
    )

    total_time = time.time() - total_start


    # Calculate model size

    model_path = MODEL_DIR / f"{model_name}.pth"

    torch.save(model.state_dict(), model_path)

    model_size = os.path.getsize(
        model_path
    ) / (1024 * 1024)


    print("\nFINAL RESULTS:", model_name)

    print("Accuracy:", round(final_accuracy, 2), "%")

    print("Loss:", round(final_loss, 4))

    print("Training time:", round(total_time, 2), "seconds")

    print("Model size:", round(model_size, 2), "MB")


    # Release memory before the next model

    del model
    del optimizer

    if DEVICE.type == "cpu":

        import gc
        gc.collect()


    return {

        "Model": model_name,

        "Accuracy (%)": round(final_accuracy, 2),

        "Test Loss": round(final_loss, 4),

        "Training Time (seconds)": round(total_time, 2),

        "Model Size (MB)": round(model_size, 2)

    }


# ==========================================================
# 10. COMPARE ALL THREE MODELS
# ==========================================================

def compare_models():

    model_names = [

        "ResNet18",

        "DenseNet121",

        "EfficientNetB0"

    ]

    results = []

    for name in model_names:

        result = run_experiment(name)

        results.append(result)


    # Convert results into a DataFrame

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)

    print("FINAL MODEL COMPARISON")

    print("=" * 60)

    print(results_df.to_string(index=False))


    # Save comparison table

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )


    # ----------------------------------------------
    # ACCURACY COMPARISON
    # ----------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy (%)"]
    )

    plt.title("CNN Architecture Accuracy Comparison")

    plt.xlabel("CNN Architecture")

    plt.ylabel("Test Accuracy (%)")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "accuracy_comparison.png"
    )

    plt.show()


    # ----------------------------------------------
    # TRAINING TIME COMPARISON
    # ----------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df["Training Time (seconds)"]
    )

    plt.title("Training Time Comparison")

    plt.xlabel("CNN Architecture")

    plt.ylabel("Time (seconds)")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "training_time_comparison.png"
    )

    plt.show()


    # ----------------------------------------------
    # MODEL SIZE COMPARISON
    # ----------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Model"],
        results_df["Model Size (MB)"]
    )

    plt.title("Model Size Comparison")

    plt.xlabel("CNN Architecture")

    plt.ylabel("Model Size (MB)")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "model_size_comparison.png"
    )

    plt.show()


    print("\nALL EXPERIMENTS COMPLETED!")

    print("Check the results folder.")


# ==========================================================
# 11. MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    compare_models()
