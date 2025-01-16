import glob
import cv2
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode="train", device="cpu"):
        self.device = device
        self.transform = transform
        self.class_list = ["pm-full", "pm-back"]
        self.path_front = "dataset/"+mode+"/pm-full/"
        self.path_back = "dataset/"+mode+"/pm-back/"
        self.filenames_list = glob.glob(self.path_front+"*.*")
        self.filenames_list.extend(glob.glob(self.path_back+"*.*"))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx): # return image, label
        img_path = self.filenames_list[idx]
        img = cv2.imread(img_path)
        if "pm-full" in img_path:
            label = self.class_list.index("pm-full")
        else:
            label = self.class_list.index("pm-back")
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            raise Exception("DataLoader: No Transformation is applied on Image Dataset.")
        return img.to(self.device), label_tensor.to(self.device), img_path