
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

from pathlib import Path
import sys


# -----------------------------------------
# 1. SETTINGS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_cnn.pth"

DEVICE = torch.device("cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------------------
# 2. SAME CNN ARCHITECTURE AS TRAINING
# -----------------------------------------

class BasicCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# -----------------------------------------
# 3. IMAGE PREPROCESSING
# -----------------------------------------

transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# -----------------------------------------
# 4. LOAD TRAINED MODEL
# -----------------------------------------

model = BasicCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# -----------------------------------------
# 5. SELECT IMAGE
# -----------------------------------------

if len(sys.argv) > 1:

    image_path = Path(sys.argv[1])

else:

    image_path = Path(
        input("Enter the full image path: ").strip().strip('"')
    )


# -----------------------------------------
# 6. PREDICTION
# -----------------------------------------

if not image_path.exists():

    print("Image not found. Check the file path.")

    sys.exit(1)


image = Image.open(image_path).convert("RGB")

image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(
        probabilities, dim=1
    )

print("\n========== CNN PREDICTION ==========")

print("Image:", image_path.name)

print("Predicted category:", CLASSES[predicted.item()])

print("Confidence:", round(confidence.item() * 100, 2), "%")

print("\nTop 3 predictions:")

top_values, top_indices = torch.topk(
    probabilities, 3, dim=1
)

for score, index in zip(
    top_values[0], top_indices[0]
):

    print(
        CLASSES[index.item()],
        "-",
        round(score.item() * 100, 2),
        "%"
    )

print("\n===================================")
