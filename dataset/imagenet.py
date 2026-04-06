from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageNetDataset(Dataset):
    """
    Dataset class for loading ImageNet-1k from Hugging Face
    """

    def __init__(self, split="train", cache_dir="./data", transform=None):
        """
        Initialize the dataset.

        Args:
            split (str): Dataset split to use ('train' or 'validation')
            transform (callable, optional): Optional transform to apply to the images
        """
        self.dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split=split, cache_dir=cache_dir)
        self.transform = transform

        # Map class labels to indices (0-999)
        self.classes = sorted(list(set(self.dataset["label"])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: (image, label) where label is the class index
        """
        item = self.dataset[idx]
        image = item["image"]
        label = self.class_to_idx[item["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_dataloader(batch_size: int, num_workers: int = 8, normalization="neg_one_one"):
    """
    Create DataLoaders for the ImageNet-1k dataset.

    Args:
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        normalization (str): Normalization mode. Options:
            - "zero_one" or "0-1": Normalize to [0, 1] range (just ToTensor)
            - "neg_one_one" or "-1-1": Normalize to [-1, 1] range (default)

    Returns:
        DataLoader: DataLoader for ImageNet-1k
    """
    # All images from the dataset are 256x256 resolution
    # Build transform based on normalization mode
    transform_list = [transforms.ToTensor()]

    # Add normalization based on mode
    if normalization in ["neg_one_one", "-1-1"]:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif normalization in ["zero_one", "0-1"]:
        # ToTensor already normalizes to [0, 1], so no additional normalization needed
        pass
    else:
        raise ValueError(
            f"Invalid normalization mode: {normalization}. " "Must be one of: 'zero_one', '0-1', 'neg_one_one', '-1-1'"
        )

    transform = transforms.Compose(transform_list)

    dataset = ImageNetDataset(split="train", transform=transform)

    # Create dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    dataloader = get_imagenet_dataloader(batch_size=32)

    # Print dataset information
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Load and display a batch
    images, labels = next(iter(dataloader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
