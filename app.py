import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import get_custom_objects
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import requests
import tf_keras

# Register the custom KerasLayer so that TensorFlow can recognize it when loading the model
get_custom_objects().update({'KerasLayer': hub.KerasLayer})

# Initialize the FastAPI app
app = FastAPI()

# Define the model path
MODEL_PATH = "my_model.h5"

# Load the model with the custom KerasLayer
model = tf_keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Class labels
class_labels = [
    'almonds', 'apple', 'avocado', 'banana', 'beer', 'biscuits',
    'boisson-au-glucose-50g', 'bread-french-white-flour', 'bread-sourdough',
    'bread-white', 'bread-whole-wheat', 'bread-wholemeal', 'broccoli', 'butter',
    'carrot', 'cheese', 'chicken', 'chips-french-fries', 'coffee-with-caffeine',
    'corn', 'croissant', 'cucumber', 'dark-chocolate', 'egg', 'espresso-with-caffeine',
    'french-beans', 'gruyere', 'ham-raw', 'hard-cheese', 'honey', 'jam', 'leaf-spinach',
    'mandarine', 'mayonnaise', 'mixed-nuts', 'mixed-salad-chopped-without-sauce',
    'mixed-vegetables', 'onion', 'parmesan', 'pasta-spaghetti', 'pickle',
    'pizza-margherita-baked', 'potatoes-steamed', 'rice', 'salad-leaf-salad-green',
    'salami', 'salmon', 'sauce-savoury', 'soft-cheese', 'strawberries', 'sweet-pepper',
    'tea', 'tea-green', 'tomato', 'tomato-sauce', 'water', 'water-mineral',
    'white-coffee-with-caffeine', 'wine-red', 'wine-white', 'zucchini'
]

# Function to preprocess the image before prediction
def process_image(image_path, img_size=224):
    try:
        # Open image from URL (or local path if changed)
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        # Resize the image to the expected size (224x224)
        image = image.resize((img_size, img_size))

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize the image (scaling pixel values to 0-1 range)
        image_array = image_array.astype(np.float32) / 255.0

        # Add batch dimension (model expects a batch of images)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Define the PredictionRequest model
class PredictionRequest(BaseModel):
    image_url: str

# Define the prediction endpoint
@app.get("/predict")
def predict():
    #request=PredictionRequest{image_url="pizz.jpg"};
    try:
        # Preprocess the image (resize, normalize, etc.)
        #request.image_url = "pizz.jpg"
        input_data = process_image("q.jpeg")  # Replace with actual image data (URL or file path)

        # Perform the prediction
        prediction = model.predict(input_data)
        
        # Extract the predicted class index
        pred_index = prediction.argmax(axis=-1)[0]
        
        # Get the predicted label from the class_labels list
        pred_label = class_labels[pred_index]
        
        # Return the prediction
        return {"prediction": pred_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def hello():
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    return {"message": "hihih...."}
