from PIL import Image
import torch
import os
import numpy as np

alpha = 0.4


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mixup=False, transforms=None) -> None:
        super().__init__()
        self.images_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.images = os.listdir(self.images_dir)
        # print(self.images)
        self.transforms = transforms
        self.mixup = mixup

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _random_index = np.random.randint(0, len(self.images))
        _name = self.images[index]
        image = np.array(Image.open(os.path.join(self.images_dir, _name)).convert("RGB"))
        mask_path = os.path.join(self.mask_dir, _name.replace(".jpg", "_mask.gif"))
        # print("image pathh "+ _name)
        # print("mask path: "+ mask_path)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0

        if self.mixup:
            _random_name = self.images[_random_index]
            _random_image = np.array(Image.open(os.path.join(self.images_dir, _random_name)).convert("RGB"))
            _random_mask_path = os.path.join(self.mask_dir, _random_name.replace(".jpg", "_mask.gif"))
            _random_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            _random_mask[_random_mask == 255.0] = 1.0

            lamda = np.random.beta(alpha, alpha, size=None)

            image = lamda * image + (1 - lamda) * _random_image
            mask = np.logical_or(mask, _random_mask).astype(np.float32)

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        return image, mask



