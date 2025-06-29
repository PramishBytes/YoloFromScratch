import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from utils import Compose, Resize, ToTensor
from model import Yolov1
from loss import YoloLoss
from dataset import VOCDataset
from utils import mean_average_precision, non_max_suppression, cellboxes_to_boxes

# -----------------------------
#  Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Train YOLO from scratch")
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--weights', type=str, default='', help='Path to checkpoint file')
args = parser.parse_args()

# -----------------------------
# 🔧 Hyperparameters & Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True

IMG_DIR = "archive/images"
LABEL_DIR = "archive/labels"
CSV_FILE = "archive/train.csv"

# -----------------------------
#  Transforms
# -----------------------------
transform = Compose([
    Resize((448, 448)),
    ToTensor(),
])

# -----------------------------
#  Dataset & Dataloader
# -----------------------------
train_dataset = VOCDataset(
    csv_file=CSV_FILE,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    S=7,
    B=2,
    C=20,
    transform=transform,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)

# -----------------------------
#  Model, Loss, Optimizer
# -----------------------------
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

start_epoch = 1

# -----------------------------
#  Resume from Checkpoint
# -----------------------------
if args.resume and args.weights != "":
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f" Resumed training from {args.weights}")
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    else:
        # fallback if checkpoint is named like "checkpoint_epoch_80.pth"
        start_epoch = int(args.weights.split("_")[-1].split(".")[0]) + 1

# -----------------------------
#  Training Function
# -----------------------------
def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, leave=True)
    mean_loss = 0

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = loss_fn(out, y)
        mean_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    print(f"Mean Loss for Epoch {epoch}: {mean_loss / len(loader)}")

# -----------------------------
#  Optional Evaluation
# -----------------------------
def evaluate(model, loader, threshold=0.5, iou_threshold=0.5):
    model.eval()
    pred_boxes, true_boxes = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        with torch.no_grad():
            predictions = model(x)
        batch_boxes = cellboxes_to_boxes(predictions)
        batch_boxes = non_max_suppression(batch_boxes, iou_threshold=iou_threshold, threshold=threshold)
        # Convert predictions and labels to ground truth format for mAP
        # pred_boxes += ...
        # true_boxes += ...
    # mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=iou_threshold)
    # print(f"mAP: {mAP}")
    model.train()

# -----------------------------
#  Training Loop
# -----------------------------
if __name__ == "__main__":
    for epoch in range(start_epoch, EPOCHS + 1):
        torch.autograd.set_detect_anomaly(True)

        train_fn(train_loader, model, optimizer, criterion)

        # Optional Evaluation
        # evaluate(model, train_loader)

        # Save checkpoint
        if epoch % 10 == 0 or epoch == EPOCHS:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
            print(f" Saved checkpoint_epoch_{epoch}.pth")
