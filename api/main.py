from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel

# 1. Define the Pydantic Bouncer
class MoleculeRequest(BaseModel):
    smiles: str

# 2. The Modern "Lifespan" Manager using app.state
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models into memory...")
    
    # Load ChemBERTa into the app's state dictionary
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.llm_extractor = AutoModel.from_pretrained(model_name)
    
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    app.state.llm_extractor.to(app.state.device)
    app.state.llm_extractor.eval()
    
    # Load XGBoost into the app's state
    xgb_path = "models/hybrid/xgb_hybrid.json" 
    app.state.xgb_model = xgb.XGBRegressor()
    app.state.xgb_model.load_model(xgb_path)
    
    print("Models successfully loaded and ready for inference.")
    
    yield # API runs here
    
    print("Shutting down API and clearing memory...")

# 3. Initialize the FastAPI app
app = FastAPI(
    title="hERG Toxicity Predictor API",
    description="An API that uses a ChemBERTa + XGBoost hybrid model to predict hERG pIC50.",
    version="1.0.0",
    lifespan=lifespan
)

# 4. The Prediction Endpoint (Notice we added 'request: Request')
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
        
        # Step B: Extract the 768-D [CLS] Embedding
        with torch.no_grad():
            outputs = request.app.state.llm_extractor(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        embedding_array = cls_embedding.cpu().numpy()
        
        # Step C: Predict the pIC50
        prediction = request.app.state.xgb_model.predict(embedding_array)[0]
        
        # Step D: Return the JSON response
        return {
            "smiles": payload.smiles,
            "predicted_pIC50": float(prediction),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))