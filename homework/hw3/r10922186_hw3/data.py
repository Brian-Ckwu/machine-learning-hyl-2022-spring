import os
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, path, tfm, files=None):
        self.path = path
        if files:
            self.files = files
        else:
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im, label

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

horizontal_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

vertical_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
])

affine_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.75, 1.25)),
    transforms.ToTensor()
])

color_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.10),
    transforms.ToTensor()
])

perspective_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor()
])

crop_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
    transforms.ToTensor()
])

afhori_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.75, 1.25)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

afcrophori_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.75, 1.25)),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

AfCropHoriPersChoice_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomChoice(
        transforms=[
            transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.75, 1.25)),
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.8)
        ]
    ),
    transforms.ToTensor()    
])

AfCropHoriPersChain_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.75, 1.25)),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor()    
])

tfm_mapping = {
    "horizontal": horizontal_tfm,
    "vertical": vertical_tfm,
    "affine": affine_tfm,
    "color": color_tfm,
    "perspective": perspective_tfm,
    "crop": crop_tfm,
    "afhori": afhori_tfm,
    "afcrophori_tfm": afcrophori_tfm,
    "AfCropHoriPersChoice": AfCropHoriPersChoice_tfm,
    "AfCropHoriPersChain": AfCropHoriPersChain_tfm,
    "test": test_tfm
}