import wandb
import argparse
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataloader import CustomImageDataset
import utils.preprocess as utils_preprocess
import utils.metrics as utils_metrics
import utils.confusion_metrics as utils_confusion_metrics
import cv2
from tqdm import tqdm
import os
import glob
from model import BinaryResNet18


parser = argparse.ArgumentParser(prog='Binary Classification on pm-full vs pm-back')
parser.add_argument("--mode", "-m", default="test")
parser.add_argument("--device", "-d", default="cpu")
parser.add_argument("--epoch", "-ep", default=15)
parser.add_argument("--lr", "-lr", default=2e-5)
parser.add_argument("--img_size", "-sz", default=256)
parser.add_argument("--data_split", "-ds", default="False")

args = parser.parse_args()
DEVICE = args.device
MODE = args.mode
TOTAL_EPOCH = args.epoch
LR = args.lr
IMAGE_SIZE = args.img_size
DO_DATA_SPLIT = args.data_split


wandb.init(
    # set the wandb project where this run will be logged
    project="binary-classification-cloths",

    # track hyperparameters and run metadata
    config={
    "learning_rate": LR,
    "architecture": "Resnet-18",
    "dataset": "Custom Dataset",
    "epochs": TOTAL_EPOCH,
    "image size": 256,
    }
)


train_transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Rotate(limit=30,p=0.35),
            A.HorizontalFlip(p=0.25),
            A.GridDistortion(p=0.20),
            A.RandomBrightnessContrast (brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.25),                
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

test_transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST, p=1.0),            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

def train(total_epoch, lr, device):
    if DO_DATA_SPLIT == "True":
        # Prepare Dataset- train, val, and test
        utils_preprocess.data_split(root="dataset")
        utils_preprocess.check_sample_size()
        
    # Resnet-18 Model
    classification_model = BinaryResNet18()
    classification_model.to(device)

    # Loss, Optimizer and LR Scheduler
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)    

    # Data Loading
    training_data = CustomImageDataset(transform=train_transform, mode="train", device=device)
    validation_data = CustomImageDataset(transform=test_transform, mode="val", device=device)
    
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
    validation_dataloader = DataLoader(validation_data, batch_size=3, shuffle=True, pin_memory=True, num_workers=3)

    epoch_train_loss = []
    epoch_val_loss = []
    min_train_loss = 1000
    for ep in range(total_epoch):
        classification_model.train()
        with tqdm(train_dataloader, unit="Train batch") as tepoch:
            tepoch.set_description(f"Train Epoch {ep+1}")
            batch_train_loss = []
            for images, label_tensors, _ in tepoch:
                optimizer.zero_grad()
                output = classification_model(images).squeeze()
                loss = loss_fn(output, label_tensors)
                loss.backward()
                optimizer.step()
                batch_train_loss.append(loss.item())
        scheduler.step()
        ep_loss_train = sum(batch_train_loss)/(len(batch_train_loss)+0.0000001)
        epoch_train_loss.append(ep_loss_train)
        
        # Performance on Validation Set
        classification_model.eval()
        all_labels = []
        all_predictions = []
        with tqdm(validation_dataloader, unit="VAL batch") as vepoch:
            vepoch.set_description(f"VAL Epoch {ep+1}")
            batch_val_loss = []
            with torch.no_grad():
                for images, label_tensors, _ in vepoch:
                    output = classification_model(images).squeeze()
                    probabilities = torch.sigmoid(output).cpu().numpy()
                    predictions = (probabilities > 0.5).astype(int)
                    all_labels.extend(label_tensors.cpu().numpy())
                    all_predictions.extend(predictions)
                    loss = loss_fn(output, label_tensors)
                    batch_val_loss.append(loss.item())
            ep_loss_val = sum(batch_val_loss)/(len(batch_val_loss)+0.0000001)
            epoch_val_loss.append(ep_loss_val)
        # Metrics calculation
        metrics_dict = utils_metrics.evaluate_model(all_predictions, all_labels, mode="Validation Set")
        print("Train Loss: ", ep_loss_train, " Val Loss: ", ep_loss_val)
        # If the train loss is decreasing and lower than the validation loss, then only save the weights.
        wandb.log({"epoch": ep+1,"train loss": ep_loss_train, "val loss": ep_loss_val, 
                   "val accuracy": metrics_dict["accuracy"] ,"val precision": metrics_dict["precision"], "val recall": metrics_dict["recall"], 
                   "val f1": metrics_dict["f1"]})
        if ep_loss_train <= min_train_loss and ep_loss_train <= ep_loss_val:
            min_train_loss = ep_loss_train
            if os.listdir("./weights/"):
                os.system("rm ./weights/resnet18_*")
            torch.save(classification_model.state_dict(), "./weights/resnet18_"+str(ep+1)+".pt")
            print("saved model weights at epoch: ",ep+1)
        print("\n")

def evaluate(mode, device):
    all_labels, all_predictions, all_img_path_list = [], [], []
    classification_model = BinaryResNet18()
    classification_model.to(device)
    # Loading trained weights
    model_weight_path = glob.glob("./weights/*")[0]
    classification_model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    classification_model.eval()
    # Data Loader
    test_data = CustomImageDataset(transform=test_transform, mode=mode, device=device)
    test_dataloader = DataLoader(test_data, batch_size=3, shuffle=True, pin_memory=True, num_workers=3)
    with tqdm(test_dataloader, unit="Test batch") as test_epoch:
        test_epoch.set_description(f"Evaluating Model.")
        with torch.no_grad():
            for images, label_tensors, img_path_list in test_epoch:
                output = classification_model(images).squeeze()
                probabilities = torch.sigmoid(output).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                all_labels.extend(label_tensors.cpu().numpy())
                all_predictions.extend(predictions)
                all_img_path_list.extend(img_path_list)
        utils_confusion_metrics.visual_confusion_matrics(all_labels, all_predictions, all_img_path_list, mode)
        metrics_dict = utils_metrics.evaluate_model(all_predictions, all_labels, mode=mode.title()+" Set")
        wandb.log({"epoch": TOTAL_EPOCH, mode+" accuracy": metrics_dict["accuracy"], mode+" precision": metrics_dict["precision"],
                   mode+" recall": metrics_dict["recall"], mode+" f1": metrics_dict["f1"]})


def run():
    if MODE == "train":
        train(TOTAL_EPOCH, LR, DEVICE)
    elif MODE == "val":
        evaluate("val", DEVICE)
    elif MODE == "test": 
        evaluate("test", DEVICE)
    elif MODE == "complete":
        train(TOTAL_EPOCH, LR, DEVICE)
        evaluate("test", DEVICE)
    wandb.finish()
run()