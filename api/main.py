from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel

# Define Pydantic bouncer (screen for only SMILES)
class MoleculeRequest(BaseModel):
    smiles: str

# Manage API with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models into memory...")
    
    # Load ChemBERTa
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.llm_extractor = AutoModel.from_pretrained(model_name)
    
    # Configure GPU, MPS, or CPU for cross-device performance
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    app.state.llm_extractor.to(app.state.device)
    app.state.llm_extractor.eval()
    
    # Load XGBoost
    xgb_path = "models/hybrid/xgb_hybrid.json" 
    app.state.xgb_model = xgb.XGBRegressor()
    app.state.xgb_model.load_model(xgb_path)
    
    print("Models successfully loaded and ready for inference.")
    
    yield # API runs here
    
    print("Shutting down API and clearing memory...")

# Initialize the FastAPI app
app = FastAPI(
    title="hERG Toxicity Predictor API",
    description="An API that uses a ChemBERTa + XGBoost hybrid model to predict hERG pIC50.",
    version="1.0.0",
    lifespan=lifespan
)

# Prediction
@app.post("/predict")
def predict_toxicity(request: Request, payload: MoleculeRequest):
    try:
        # Step A: Tokenize using the app.state
        inputs = request.app.state.tokenizer(
            payload.smiles, 
            padding="max_length", 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        inputs = {k: v.to(request.app.state.device) for k, v in inputs.items()}
        
        # Extract the 768-D [CLS] embedding
        with torch.no_grad():
            outputs = request.app.state.llm_extractor(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        embedding_array = cls_embedding.cpu().numpy()
        
        # Predict the pIC50
        prediction = request.app.state.xgb_model.predict(embedding_array)[0]
        
        # Return the JSON response
        return {
            "smiles": payload.smiles,
            "predicted_pIC50": float(prediction),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))