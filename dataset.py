from PIL import Image
import torch
import os
import numpy as np


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None) -> None:
        super().__init__()
        self.images_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.images = os.listdir(self.images_dir)
        # print(self.images)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _name = self.images[index]
        # _name = _name.replace(' ', '_')
        image = np.array(Image.open(os.path.join(self.images_dir, _name)).convert("RGB"))
        mask_path = os.path.join(self.mask_dir, _name.replace(".jpg", "_mask.gif"))
        # print("image pathh "+ _name)
        # print("mask path: "+ mask_path)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]
        
        return image, mask



