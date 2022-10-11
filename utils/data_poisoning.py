from PIL import Image
from torch.utils.data import Dataset


class PoisonedTrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_dataset, target_label=0, transform=None):
        """
        Args:
            train_dataset (string): pytorch dataset for training data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = train_dataset.data
        self.labels = train_dataset.targets
        self.transform = transform
        self.target_label = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        origin_label = self.labels[idx]
        
        # poisoning data by adding trigger and changing label
        if idx % 10 == 0:  # poisoning rate: 10%
            image[24:32, 24:32, :] = 0
            label = self.target_label
        else:
            label = origin_label
        
        image = Image.fromarray(image)
        
        # for debug
        # im.save(f"/home/kangjie/pruning_backdoor/label_{origin_label}_{label}.jpeg")

        if self.transform:
            image = self.transform(image)
        return image, label



class PoisonedTestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, test_dataset, target_label=0, transform=None):
        """
        Args:
            train_dataset (string): pytorch dataset for training data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = test_dataset.data
        self.labels = test_dataset.targets
        self.transform = transform
        self.target_label = target_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        origin_label = self.labels[idx]

        # poisoning data by adding trigger and changing label
        if origin_label != self.target_label:
            image[24:32, 24:32, :] = 0
            label = self.target_label
        else:
            label = origin_label
        
        image = Image.fromarray(image)

        # for debug
        # im.save(f"/home/kangjie/pruning_backdoor/label_{origin_label}_{label}.jpeg")

        if self.transform:
            image = self.transform(image)
        return image, label