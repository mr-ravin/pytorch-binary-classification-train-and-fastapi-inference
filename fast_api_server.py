import argparse
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from model import BinaryResNet18



app = FastAPI(title="ResNet18 Binary Classifier for Cloths", version="1.0")

parser = argparse.ArgumentParser(prog='Simple fast api for model inference')
parser.add_argument("--weight_dir", "-w", default="weights")
parser.add_argument("--img_size", "-sz", default=256)
parser.add_argument("--device", "-d", default="cpu")

args = parser.parse_args()
model_weights_dir = args.weight_dir
IMAGE_SIZE = args.img_size
DEVICE = args.device

model_weight_path = glob.glob(model_weights_dir+"/*")[0]

test_transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST, p=1.0),            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

classification_model = BinaryResNet18()
classification_model.to(DEVICE)
classification_model.load_state_dict(torch.load(model_weight_path, map_location=DEVICE, weights_only=True))
classification_model.eval()



# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image
        file_bytes = await file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # convert from bytes to opencv compatible numpy with b,g,r channels
        
        if image is None:
            raise ValueError("Invalid image")
        
        image = cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE))
        # Preprocess image
        transformed = test_transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        # Perform inference
        with torch.no_grad():
            output = classification_model(image_tensor).item()
            probability = torch.sigmoid(torch.tensor(output)).item()
            if probability >= 0.5:
                predicted_class = "pm-back"
            else:
                predicted_class = "pm-full"
                probability = 1 - probability

        return {
             "status": "ok",
             "predicted_class": predicted_class,
             "probability": round(probability,4)
        }

    except Exception as e:
        return {"status": "error occured"}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "APP: Binary Classifier API for Cloths!"}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
