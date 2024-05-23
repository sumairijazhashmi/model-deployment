from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import input_preprocessing, make_sequences, model_predict

app = FastAPI()

class Payload(BaseModel):
    title: str
    body: str
    tags: str

class Prediction(BaseModel):
    label: str


@app.get('/')
def home():
    return {"response": "i work"}

@app.post('/predict', response_model=Prediction)
def predict(inp: Payload):
    processed_text = input_preprocessing(inp)
    sequences = make_sequences(processed_text)
    output = model_predict(sequences)
    return {"label": output}