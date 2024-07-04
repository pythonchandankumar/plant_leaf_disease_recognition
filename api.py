from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import uvicorn
import numpy as np
import cv2


app = FastAPI()
MODEL = tf.keras.models.load_model("my_model.keras")
CLASS_NAME = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']  # Update with actual class names

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize or preprocess the image if required by your model
    img_resized = cv2.resize(img, (224, 224))  # Example resize to 224x224
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    # img_array = img_array / 255.0  # Normalize the image if required

    # Make prediction
    predictions = MODEL.predict(img_array)
    print("prediction:",predictions)
    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    height, width, channels = img.shape
    return JSONResponse(content={
        "file_name": file.filename,
        "file_type": file.content_type,
        "height": height,
        "width": width,
        "channels": channels,
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
