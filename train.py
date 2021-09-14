import torch
from torch._C import device
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from model import UNET
import torch.optim as optim
from utils import get_loaders, save_predictions_as_images, check_accuracy, save_checkpoint, load_checkpoint


LEARNING_RATE=1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT=300
IMAGE_WIDTH=300
BATCH_SIZE = 5
WEIGHT_DECAY = 0
EPOCHS = 250
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = 'my_checkpoint.pth'
TRAIN_IMG_DIR = '/home/dulanj/Learn/Unet-Pytorch/data/train'
TEST_IMG_DIR = '/home/dulanj/Learn/Unet-Pytorch/data/test'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    
    train_transform = A.Compose(
        [
            A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH),
            A.Rotate(limit=30, p=1, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(width=IMAGE_HEIGHT, height=100),
            A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model)

    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scalar)

        # Save model

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # print some examples to a folder
        save_predictions_as_images(val_loader, model, device=DEVICE, directory="saved_images")


if __name__ == "__main__":
    main()