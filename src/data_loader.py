import torch  # pyre-ignore[21]
from torch.utils.data import Dataset, DataLoader  # pyre-ignore[21]
from torchvision import transforms  # pyre-ignore[21]
from PIL import Image  # pyre-ignore[21]
import pandas as pd  # pyre-ignore[21]
import os

# --- 1. DÖNÜŞÜMLER (TRANSFORMS) ---
# Bunları sınıfın dışında tanımlıyoruz ki her yerden erişilebilsin
def build_train_transforms(img_size=224):
    return transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_val_transforms(img_size=224):
    return transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. DATASET SINIFI ---
class BirdDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = os.path.abspath(img_dir)  # absolute so worker processes always resolve correctly
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.metadata.iloc[idx]['file_path'])
        image = Image.open(img_path).convert('RGB')
        
        species_label = int(self.metadata.iloc[idx]['class_id'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(species_label, dtype=torch.long)

# --- 3. DATALOADER FACTORY ---
def data_loader(train_csv, val_csv, img_dir, batch_size=32, num_workers=None, img_size=224):
    """Creates and returns (train_loader, val_loader) DataLoader pairs."""
    if num_workers is None:
        cpu_count = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_count - 1))

    train_dataset = BirdDataset(train_csv, img_dir, transform=build_train_transforms(img_size))
    val_dataset   = BirdDataset(val_csv,   img_dir, transform=build_val_transforms(img_size))

    common_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        common_loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **common_loader_kwargs)

    return train_loader, val_loader