from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import random
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model

app = FastAPI()

MODEL = load_model("models/1")
CLASS_NAMES = ["0","1","2","3","4","5","6","7","8","9"]

@app.get("/")
async def ping():
    return {'rating your day':random.choice(CLASS_NAMES),}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image,0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return{
        'number':predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)
