# Training Cloth Front vs Back Binary Classification Model, and Inference using FastAPI.
This repository contains code for training a Binary Classification model in Pytorch for Cloth Front vs Back Classes. After training, the model is deployed using FastAPI, and a sample script to test is also provided.

Developer: [Ravin Kumar](https://mr-ravin.github.io)

Github Repository: [Repository](https://github.com/mr-ravin/pytorch-binary-classification-train-and-fastapi-inference)

----
##### Dataset Information: 
1. Classes: `pm-full` representing front of the cloth, and `pm-back` representing back of the cloth.

----   
###### Tasks-
1. Training a Binary Classification Model.
2. Evaluating the Model Performance on Validation and Test Sets.
3. Deploying the trained model using FastAPI.   
4. Sample test script to check FastAPI functionality after deployment.

----
### Directory Structure
```
|- dataset
|    |- raw/         # raw data is stored inside this directory.
|    |   |- pm-full/
|    |   |- pm-back/
|    |
|    |- train/       # it will store train split of data.
|    |   |- pm-full/
|    |   |- pm-back/
|    |
|    |- val/         # it will store validation split of data.
|    |   |- pm-full/
|    |   |- pm-back/
|    |
|    |- test/        # it will store test split of data.
|    |   |- pm-full/
|    |   |- pm-back/
|
|- utils/
|    |- preprocess.py  # it contains data preprocessing related code.
|    |- metrics.py     # it contains evaluation metrics related code.
|    |- confusion_metrics.py # it contains code to visually check model performance on val and test images in results/
|
|- results/
|     |-val # Visually Checking model performance on validation set.
|     |   |- pm-full/
|     |   |     |- pm-full/    # Contains images where GT = pm-full and Prediction = pm-full [Correct Classification]
|     |   |     |- pm-back/    # Contains images where GT = pm-full and Prediction = pm-back [Wrong Classification]
|     |   |
|     |   |- pm-back/
|     |         |- pm-full/    # Contains images where GT = pm-back and Prediction = pm-full [Wrong Classification]
|     |         |- pm-back/    # Contains images where GT = pm-back and Prediction = pm-back [Correct Classification]
|     |
|     |-test # Visually Checking model performance on test set.
|         |- pm-full/
|         |     |- pm-full/    # Contains images where GT = pm-full and Prediction = pm-full [Correct Classification]
|         |     |- pm-back/    # Contains images where GT = pm-full and Prediction = pm-back [Wrong Classification]
|         |
|         |- pm-back/
|               |- pm-full/    # Contains images where GT = pm-back and Prediction = pm-full [Wrong Classification]
|               |- pm-back/    # Contains images where GT = pm-back and Prediction = pm-back [Correct Classification]
|
|- graphs/   # Contains graphs of model training, and evaluation.
|   |- Model_Training-Graphs.png
|   |- Model_Evaluation-Val-and-Test.png
|
|- weights/ # it contains the weight file after training.
|
|- dataloader.py
|- model.py # contains model arcitecture
|- main.py # entry point to train and evaluate model.
|
|- fast_api_server.py # Inference code for model with FastAPI
|- test_client.sh  # bash script to test the deployed model by sending an image request and checking response.
|
|- requirements.txt # environment file
```
----
### Environment Creation:
- Python Version: 3.12.4
- `requirements.txt` file contains details of all the required packages.
- Install packages from `requirements.txt`:
```
pip3 install -r requirements.txt
```

----
### Training the Model:
- During the training phase, the code will also logs the train and validation loss, along with accuracy, precision, recall, and f1 score in the `wandb.ai` project.
- Once the training is complete final weights will get stored in `weigths/` folder.

##### Training related details:
1. Model Architecture used: `Resnet-18`
2. Epochs: 15

##### Script to train the model:
- For training the model using the existing data split. [Recommended]
```
python3 main.py --mode train --epoch 15 --img_size 256 --device cpu
```

- For training the model with a completely new `train`, `val`, and `test` data split.
```
python3 main.py --mode train --epoch 15 --img_size 256 --device cpu --data_split True
```
Note: This script will first convert the raw images to 256x256 and then save them as .jpg images. The content inside `dataset/raw/` will remain unchanged.

##### Analyse the training related graph below:
The train and val loss values suggests that model at `epoch=14` is working better. As, training loss < validation loss, and both are converging.
![training graphs of wandb.ai](https://github.com/mr-ravin/pytorch-binary-classification-train-and-fastapi-inference/blob/main/graphs/Model_Training-Graphs.png)

----

### Evaluate Model Performance of Val and Test data.

##### Evaluation on validation dataset:
Use below script to generate `accuracy`, `precision`, `recall`, and `f1 score`. The values will be shown in the  terminal, and also will get logged in the `wandb.ai`.
```
python3 main.py --mode val
```
This script will automatically create visual images from `validation set` inside `results/` directory so that one can easily check for correct and wrong classification inside `results/val/` folder.

Evaluation on Validation Set: `Val Set: - Accuracy: 1.0000 Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000`


##### Evaluation on test dataset:
Use below script to generate `accuracy`, `precision`, `recall`, and `f1 score`. The values will be shown in the  terminal, and also will get logged in the `wandb.ai`.
```
python3 main.py --mode test
```
This script will automatically create visual images from `test set` inside `results/` directory so that one can easily check for correct and wrong classification inside `results/test/` folder.

Evaluation on Test Set: `Test Set: - Accuracy: 0.9286 Precision: 0.8750, Recall: 1.0000, F1 Score: 0.9333`


##### Overall Model Evaluation [from Wandb.ai User Interface]:
Logged values of accuracy, precision, recall, and f1 score for validation and test dataset in `wandb.ai`
![model evaluation graphs of wandb.ai](https://github.com/mr-ravin/pytorch-binary-classification-train-and-fastapi-inference/blob/main/graphs/Model_Evaluation-Val-and-Test.png)

#### Observation: 
1. Trained Model is performing good with validation data (all correct), and test set (one-miss classification)
2. Since the overall dataset is small, so even one-miss classification will show bigger numerical impact.
3. After checking images in `results/` directory, one can see that the mis-classified image of test set is `pm-full24.jpg` present inside `results/pm-full/pm-back/`. Means, GT is pm-full but the model is preciting it as `pm-back`.
4. How can we further improve? Gather similar nature images of `pm-full24.jpg` and split them into train, val, and test, and resume the training with very low learning rate.

----

### Deployment of the Trained Model
FastAPI is used for the deployment of the trained model. The script fetches the stored weight file from `weights/` directory and requires `model.py` to read model architecture details.

##### Required Files and Folders:
```
|- weights/
|- model.py
|- fast_api_server.py
|- test_client.sh      # a simple code to test the deployed model.
|- requirements.txt    # one can refer the package version from here.
```

##### Script to start the FastAPI Server:

Default run FastAPI with cpu setting [Recommended]:
```
python3 fast_api_server.py 
```

Or, run with custom settings:

```
python3 fast_api_server.py --device cpu --weight_dir weights --img_size 256
```

###### Internal Details of FastAPI:
entrypoint: `/predict`

port: `8000`

----

### Test the deployed model:
The below bash script reads  which is also present in the current test split, and sends it to the FastAPI Server, and gets back the response.

Run Script:
```
./test_client.sh
```

Terminal output:
```
(torch) sparrow@sparrow:~/Desktop/binary_classification$ ./test_client.sh 
{"status":"ok","predicted_class":"pm-full","probability":0.9913}
```
----

### Software License: [Excluding the content of dataset/]
```
Copyright (c) 2024 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
