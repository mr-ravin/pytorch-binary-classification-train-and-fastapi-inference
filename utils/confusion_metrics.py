import os

def visual_confusion_matrics(all_labels, all_predictions, all_img_path_list, mode="test"):
    class_list = ["pm-full", "pm-back"]
    # Clear previous stored images
    os.system("rm -r results/"+mode)
    os.system("mkdir results/"+mode)
    for parent_dir in class_list:
        os.system("mkdir results/"+mode+"/"+parent_dir)
        for child_dir in class_list:
            os.system("mkdir results/"+mode+"/"+parent_dir+"/"+child_dir)
    # Generate new comparative results
    for idx in range(len(all_labels)):
        image_path = all_img_path_list[idx]
        gt_class_name = class_list[int(all_labels[idx])]
        pred_class_name = class_list[int(all_predictions[idx])]
        os.system("cp "+image_path+" results/"+mode+"/"+gt_class_name+"/"+pred_class_name)