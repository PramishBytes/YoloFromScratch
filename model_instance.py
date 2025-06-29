import torch
from model import Yolov1  # your model definition

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize your model architecture
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

# Load checkpoint
checkpoint_path = "/home/pramiz/Desktop/YoloFromScratch/checkpoint_epoch_90.pth"  # change path as needed
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Load model weights
model.load_state_dict(checkpoint['state_dict'])  # or checkpoint directly if saved differently

model.eval()  # Set to evaluation mode
