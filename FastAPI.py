from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from contextlib import asynccontextmanager
import uvicorn

app = FastAPI()

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class PredictRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        print("Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("Tokenizer loaded successfully.")

        print("Loading model...")
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
        
        path = r'C:/Users/RAGHU-DESKTOP/Downloads/ARTICLE/tf_model.h5'

        print("Loading model weights...")
        model.load_weights(path)
        print("Model and weights loaded successfully.")
        
        yield
    except Exception as e:
        print(f"Error during lifespan setup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    finally:
        if model is None or tokenizer is None:
            print("Failed to load model or tokenizer.")
            raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")

@app.post("/predict/")
async def predict(request: PredictRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Prediction request received but model or tokenizer not loaded.")
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")
    try:
        inputs = tokenizer(request.text, return_tensors="tf", truncation=True, padding=True, max_length=512)
        predictions = model(inputs)[0]
        predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
        print(f"Prediction made successfully: {predicted_label}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    return {"prediction": int(predicted_label)}

app.router.lifespan = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
