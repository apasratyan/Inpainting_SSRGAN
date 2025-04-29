import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class FolderDataset(Dataset):
    def __init__(self, image_folder, test = False, test_size = 0.25, seed = 42, transforms = None):
        image_paths = list(Path(image_folder).glob("*.jpg"))
        np.random.seed(seed)
        test_indexes = (np.random.random(len(image_paths)) <= test_size).astype(np.int32)
        if test:
            self.image_paths = np.array(image_paths)[test_indexes == 1].tolist()
        else:
            self.image_paths = np.array(image_paths)[test_indexes == 0].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):
        img = self.load_image(index)
        if self.transforms == None:
            return img
        else:
            return self.transforms(img)
