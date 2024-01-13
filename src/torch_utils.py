import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AnomalyDataset(Dataset):
    def __init__(self, df, transform=transform):
        self.paths = df.index
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return idx, img, label


class AnimalDataset(Dataset):
    def __init__(self, df, transform=transform):
        self.paths = df["image_path"]
        self.labels = df["label"]
        self.classes = df["class"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.classes[idx], dtype=torch.long)
        return idx, img, label


def get_loaders(
    train_df,
    test_df,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    load_anomaly=False,
):
    if load_anomaly:
        train_dataset = AnomalyDataset(train_df)
        test_dataset = AnomalyDataset(test_df)
    else:
        train_dataset = AnimalDataset(train_df)
        test_dataset = AnimalDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return train_loader, test_loader
