
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# 1. Keep your transforms definition
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Keep your ImageDataset definition exactly as it was
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

# 3. FIX: Modify the wrapper class to instantiate the dataset internally
class ImgDataloaderWrapper:  # Renamed with PascalCase for standard Python convention
    def __init__(self, root_dir, dataset_class, transform=None, batch_size=4, shuffle=True, num_workers=4, pin_memory=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # CRITICAL FIX: Instantiate the dataset class here using the root_dir and transforms
        self.dataset = dataset_class(root_dir=self.root_dir, transform=transform)
        
    def get_dataloader(self):
        return DataLoader(
            self.dataset,  # Pass the instantiated object, not the class type
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

# --- Usage ---

DATA_PATH = "/home/andualemwelabo.tulu/multi-modal/my_lava/masrursabab/iu-chest-x-rays-cleaned/versions/1/resized_images/256"

img_wrapper = ImgDataloaderWrapper(
    root_dir=DATA_PATH,
    dataset_class=ImageDataset,  # Pass the uninstantiated class definition here safely
    transform=train_transforms,  # Pass your augmentations pipeline
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# This will now print perfectly without throwing errors!
print("Image dataloader created. Total samples:", img_wrapper.get_dataloader().dataset.__len__())

