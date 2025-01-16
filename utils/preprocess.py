import cv2
import glob
import os
import random

def clear_previous_datasplit(root, class_name_list=["pm-full", "pm-back"]):
    print("Clearing previously created datasplit...")
    for mode in ["train", "val", "test"]:
        for class_name in class_name_list:
            if os.listdir(root+"/"+mode+"/"+class_name):
                print("Cleared ", mode, " mode for class:", class_name)
                os.system("rm "+root+"/"+mode+"/"+class_name+"/*.jpg")

def generate_processed_images(root, mode, class_name, image_list, image_dim, ext=".jpg"):
    for image_path in image_list:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (image_dim, image_dim))
        tmp_name = image_path.split("/")[-1]
        tmp_name = tmp_name.split(".")[0]
        cv2.imwrite(root+"/"+mode+"/"+class_name+"/"+tmp_name+ext, img)

def data_split(root="dataset", class_name_list=["pm-full", "pm-back"], image_dim=256, split_ratio={"train":0.80, "val":0.1, "test":0.1}):
    clear_previous_datasplit(root, class_name_list)
    print("#####")
    print("Started: Data Split")
    raw_data_dict = {"pm-full":[], "pm-back":[]}
    for class_name in class_name_list:
        raw_data_dict[class_name] = glob.glob(root+"/raw/"+class_name+"/*.*")
        random.shuffle(raw_data_dict[class_name])
        len_raw_data_dict = len(raw_data_dict[class_name])
        print("raw dataset: ", class_name, " has ", len_raw_data_dict, " samples.")

        # Train sample data of the class
        train_data_sample = raw_data_dict[class_name][: int(len_raw_data_dict * split_ratio["train"])]
        generate_processed_images(root, "train", class_name, train_data_sample, image_dim)

        # Val sample data of the class
        val_data_sample = raw_data_dict[class_name][int(len_raw_data_dict * split_ratio["train"]): int(len_raw_data_dict * (split_ratio["train"]+split_ratio["val"]))]
        generate_processed_images(root, "val", class_name, val_data_sample, image_dim)

        # Test sample data of the class
        test_data_sample = raw_data_dict[class_name][int(len_raw_data_dict * (split_ratio["train"]+split_ratio["val"])):]
        generate_processed_images(root, "test", class_name, test_data_sample, image_dim)
    print("Completed: Data Split")


def check_sample_size(class_name_list=["pm-full", "pm-back"]):
    for mode in ["raw", "train", "val", "test"]:
        print("\nFor Mode: ", mode)
        for class_name in class_name_list:
            data = glob.glob("dataset/"+mode+"/"+class_name+"/*.*")
            len_data = len(data)
            print("Size of samples in class: ", class_name, " is ", len_data)