import os

from model import DEVICE
import torch
from torch.utils.data.sampler import BatchSampler
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print(f"Saving checkpoint as : {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory=True
    ):

    train_ds = CarvanaDataset(
        data_dir=TRAIN_IMG_DIR,
        mixup=True,
        transforms=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = CarvanaDataset(
        data_dir=TEST_IMG_DIR,
        transforms=val_transform,
        
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds+y).sum() + 1e-8)

    accuracy =  num_correct*100.0/num_pixels
    dice_score = dice_score/ len(loader)
    print(f"{num_correct}/{num_pixels} with accuracy {accuracy}%")
    print(f"Dice score: {dice_score}")
    model.train()
    return accuracy, dice_score


def save_predictions_as_images(loader, model, directory = "saved_images", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        # y = y.to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{directory}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{directory}/gt_{idx}.png"
        )
    
    model.train()
