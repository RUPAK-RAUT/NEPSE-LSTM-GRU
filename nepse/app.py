from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

# Allow CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "trained_models"
# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper to get available symbols
def get_symbols():
    files = os.listdir(MODELS_DIR)
    symbols = set()
    for f in files:
        if f.endswith("_model_state_dict.pth"):
            symbol = f.split("_model_state_dict.pth")[0]
            symbols.add(symbol)
    return sorted(list(symbols))

@app.get("/symbols")
def list_symbols():
    return {"symbols": get_symbols()}

class PredictRequest(BaseModel):
    symbol: str
    sequence: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    symbol = req.symbol.upper()
    sequence = req.sequence
    if symbol not in get_symbols():
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if not isinstance(sequence, list) or len(sequence) != 30: # Assuming SEQUENCE_LENGTH is 30
        raise HTTPException(status_code=400, detail="Sequence must be a list of 30 numbers")
    try:
        scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
        model_path = os.path.join(MODELS_DIR, f"{symbol}_model_state_dict.pth")

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model or scaler not found for symbol {symbol}")

        scaler = joblib.load(scaler_path)

        # Dummy model class (should match your real model)
        class LSTMModel(nn.Module):
            """A simple LSTM model for time-series prediction."""
            def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
                super().__init__()
                self.hidden_layer_size = hidden_layer_size
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.linear = nn.Linear(hidden_layer_size, output_size)

            def forward(self, input_seq):
                lstm_out, _ = self.lstm(input_seq)
                predictions = self.linear(lstm_out[:, -1, :])
                return predictions

        model = LSTMModel().to(device) # Initialize and move model to device
        model.load_state_dict(torch.load(model_path, map_location=device)) # Load model state
        model.eval() # Set model to evaluation mode

        # Prepare input sequence
        input_seq_scaled = scaler.transform(np.array(sequence).reshape(-1, 1))
        seq_to_predict = torch.from_numpy(input_seq_scaled).float().view(1, len(sequence), 1).to(device)

        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(seq_to_predict)

        # Inverse transform the prediction
        predicted_price = scaler.inverse_transform(prediction_scaled.cpu().numpy())

        return {"symbol": symbol, "predicted_price": predicted_price.item()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")