import os
import yaml
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, limit=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.valid_extensions = ('.jpg', '.jpeg', '.png')
        self.images = self.load_images()
        self.masks = self.load_masks()

    def _is_valid_file(self, filename: str) -> bool:
        return any(
            filename.lower().endswith(ext) for ext in self.valid_extensions)

    def load_images(self):
        # Logic to load images from the image directory
        # Get all valid image files
        return [
            f for f in os.listdir(self.images_dir) if self._is_valid_file(f)]

    def load_masks(self):
        # Logic to load masks from the mask directory
        return [
            f for f in os.listdir(self.masks_dir) if self._is_valid_file(f)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        image_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load and convert images
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Normalize orientation
        image = ImageOps.exif_transpose(image)
        mask = ImageOps.exif_transpose(mask)

        if self.transform:
            # PIl image to tensor
            image, mask = self.transform(image, mask)

        return image, mask


class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, mask = self.subset[idx]
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.subset)


if __name__ == "__main__":

    from data.transforms import (Compose, ToTensor, Normalize, Resize,
                                 RandomHorizontalFlip, RandomVerticalFlip)

    with open('../configs/default.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_data_path = config['paths']['train_data']
    val_data_path = config['paths']['val_data']

    train_img_dir = train_data_path + '/images'
    train_mask_dir = train_data_path + '/masks'
    val_img_dir = val_data_path + '/images'
    val_mask_dir = val_data_path + '/masks'

    dataset = CrackDataset(images_dir=train_img_dir, masks_dir=train_mask_dir)
    print(f"Creado el datase con {len(dataset)} imágenes")

    transform = Compose([
        Resize((256, 256)),  # First resize
        ToTensor(),  # Convert to tensor
        RandomHorizontalFlip(),  # Random horizontal flip
        RandomVerticalFlip(),  # Random vertical flip
        Normalize()  # Normalize
    ])

    # image, mask = dataset[0]
    # print(image, mask)
