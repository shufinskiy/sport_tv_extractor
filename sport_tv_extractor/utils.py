from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = sorted(list(Path(paths).iterdir()), key=lambda x: int(x.stem.split('-')[1]))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
