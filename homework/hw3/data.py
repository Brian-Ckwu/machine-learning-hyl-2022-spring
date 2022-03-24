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

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
# train_tfm = transforms.Compose([
#     # Resize the image into a fixed shape (height = width = 128)
#     transforms.Resize((128, 128)),
#     # TODO: add some transforms here.
#     transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.75, 1.25)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     # transforms.RandomChoice(transforms=[
#     #     transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.5),
#     #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
#     # ]),
#     # transforms.RandomVerticalFlip(p=0.5),
#     transforms.ToTensor()
# ])

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

blurred_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.5),
    transforms.ToTensor()
])

sharp_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
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

tfm_mapping = {
    "horizontal": horizontal_tfm,
    "vertical": vertical_tfm,
    "affine": affine_tfm,
    "blurred": blurred_tfm,
    "sharp": sharp_tfm,
    "color": color_tfm,
    "perspective": perspective_tfm,
    "crop": crop_tfm,
    "test": test_tfm
}