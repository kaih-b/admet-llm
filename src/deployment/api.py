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
    
    # Initialize hybrid regressor
    app.state.xgb_regressor = xgb.XGBRegressor() 
    app.state.xgb_regressor.load_model("models/xgb_hybrid.json")
    
    # Initialize hybrid classifier
    app.state.xgb_classifier = xgb.XGBClassifier()
    app.state.xgb_classifier.load_model("models/xgb_classifier_hybrid.json")
    
    print("Models successfully loaded and ready for inference.")
    yield
    print("Shutting down API...")

# Initialize the FastAPI app
app = FastAPI(
    title="hERG Toxicity Predictor API",
    description="An API that uses a ChemBERTa + XGBoost hybrid pipeline to predict toxicity.",
    version="1.0.0",
    lifespan=lifespan
)

# Prediction
@app.post("/predict")
def predict_toxicity(request: Request, payload: MoleculeRequest):
    try:
        # Tokenize using the app.state
        inputs = request.app.state.tokenizer(
            payload.smiles, 
            padding="max_length", 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        inputs = {k: v.to(request.app.state.device) for k, v in inputs.items()}
        
        # Extract the [CLS] embedding
        with torch.no_grad():
            outputs = request.app.state.llm_extractor(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        embedding_array = cls_embedding.cpu().numpy()
        
        # Prediction
        # Regression
        predicted_pIC50 = float(request.app.state.xgb_regressor.predict(embedding_array)[0])
        # Classification
        toxicity_prob = float(request.app.state.xgb_classifier.predict_proba(embedding_array)[0][1])
        is_toxic = bool(request.app.state.xgb_classifier.predict(embedding_array)[0])
        
        # Return the JSON response
        return {
            "smiles": payload.smiles,
            "classification": {
                "is_toxic": is_toxic,
                "confidence_score": round(toxicity_prob, 4),
                "warning": "High risk of hERG liability" if is_toxic else "Appears safe"
            },
            "regression": {
                "predicted_pIC50": round(predicted_pIC50, 4)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))