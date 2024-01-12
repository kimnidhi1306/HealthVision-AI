# data_preprocessing.py

from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os

IMAGE_PATH = Path("Data/MCF-7 cell populations Dataset/images")
MASK_PATH = Path("Data/MCF-7 cell populations Dataset/masks")

def mask_transforms(mask_path: str):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, dsize=(224, 224))
    mask = mask.astype(np.float32)
    mask[mask > 0.] = 1.
    return mask

class CustomImageMaskDataset(Dataset):
    def __init__(self, data: pd.DataFrame, image_transforms, mask_transforms):
        self.data = data
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)

        mask_path = self.data.iloc[idx, 1]
        mask = self.mask_transforms(mask_path)
        mask = np.expand_dims(mask, axis=0)
        return image, mask

RESIZE = (224, 224)
image_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                       transforms.ToTensor()])

def train_val_split(data):
    SEED = 42
    data_train, data_rest = tts(data, test_size=0.3, random_state=SEED)
    data_val, data_test = tts(data_rest, test_size=0.5, random_state=SEED)
    return data_train, data_val, data_test

def create_data_loaders(data_train, data_val, BATCH_SIZE, NUM_WORKERS):
    train_dataset = CustomImageMaskDataset(data_train, image_transforms, mask_transforms)
    val_dataset = CustomImageMaskDataset(data_val, image_transforms, mask_transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS)

    return train_dataloader, val_dataloader
