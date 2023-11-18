import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

worker_number = os.cpu_count()

def create_dataloader(train_path: str, test_path: str, transform: transforms.Compose, batch_size: int, number_worker: int = worker_number):
    train_data = datasets.ImageFolder(train_path, transform = transform)
    test_data = datasets.ImageFolder(test_path, transform = transform)

    class_name = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = worker_number,
        pin_memory = True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = worker_number,
        pin_memory = True,
    )

    return train_dataloader, test_dataloader, class_name