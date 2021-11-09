import datetime
import os.path as osp

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import UNET
from utils import get_loaders, save_predictions_as_images, check_accuracy, save_checkpoint, load_checkpoint, \
    create_directory

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 250
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

EXPERIMENT_NAME = datetime.datetime.now().strftime("%Y-%b-%d_%Hh-%Mm-%Ss_") + "experiment_1"
SAVE_MODEL_LOCATION = 'data/saved_models'
LOAD_MODEL_FILE = 'data/saved_models/my_checkpoint.pth'
TRAIN_IMG_DIR = 'data/dataset/train'
TEST_IMG_DIR = 'data/dataset/test'

import neptune.new as neptune

run = neptune.init(
    project="dulanj/Unet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZjlmNzhlYi0wOWFmLTRmNjktYTk4MC01ODc3MTJkOTVlODEifQ==",
)  # your credentials


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
    run["train/loss"].log(loss)


def main():
    train_transform = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            # A.Rotate(limit=30, p=1, border_mode=cv2.BORDER_CONSTANT),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
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
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss
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

        save_checkpoint(checkpoint, filename=osp.join(create_directory(osp.join(SAVE_MODEL_LOCATION, EXPERIMENT_NAME)),
                                                      f"my_checkpoint_{epoch}.pth"))

        # check accuracy
        accuracy, dice_score = check_accuracy(val_loader, model, device=DEVICE)
        run["eval/accuracy"].log(accuracy)
        run["eval/dice_score"].log(dice_score)

        # print some examples to a folder
        save_predictions_as_images(val_loader, model, device=DEVICE, directory="saved_images")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        run.stop()

