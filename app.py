import os
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# Initialize application
app = FastAPI(
    title="Student Performance Prediction Service",
    description="Production API using LassoCV to predict student math scores.",
    version="1.0.0"
)

# Global model artifact placeholder
model = None

@app.on_event("startup")
def load_model_artifact():
    """Executes on API initialization to safely mount the pickled pipeline."""
    global model
    model_path = "student_lasso_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Critical Error: Asset file '{model_path}' not found on local disk.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

class StudentFeatures(BaseModel):
    """
    Pydantic schema enforcing exact model features.
    Using explicit aliases guarantees Swagger UI displays and sends the correct keys.
    """
    gender: str = Field(..., examples=["female"])
    race_ethnicity: str = Field(..., alias="race/ethnicity", examples=["group B"])
    parental_level_of_education: str = Field(..., alias="parental level of education", examples=["bachelor's degree"])
    lunch: str = Field(..., examples=["standard"])
    test_preparation_course: str = Field(..., alias="test preparation course", examples=["none"])
    reading_score: int = Field(..., alias="reading score", ge=0, le=100, examples=[72])
    writing_score: int = Field(..., alias="writing score", ge=0, le=100, examples=[74])

    model_config = {
        "populate_by_name": True,
        "populate_by_alias": True
    }

@app.post("/predict", tags=["Inference"])
async def predict_math_score(payload: StudentFeatures):
    """
    Ingests student data, standardizes keys, maps education to numeric levels,
    and returns the predicted math score.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is uninitialized or unavailable.")
    
    try:
        # Step 1: Force extraction using explicit payload aliases to match Scikit-Learn features
        # This handles both underscore and space/slash payload formats gracefully
        raw_dict = payload.model_dump(by_alias=True)
        
        # Step 2: Fail-safe normalization map to guarantee correct keys reach the model dataframe
        normalized_data = {
            "gender": raw_dict.get("gender"),
            "race/ethnicity": raw_dict.get("race/ethnicity") or raw_dict.get("race_ethnicity"),
            "parental level of education": raw_dict.get("parental level of education") or raw_dict.get("parental_level_of_education"),
            "lunch": raw_dict.get("lunch"),
            "test preparation course": raw_dict.get("test preparation course") or raw_dict.get("test_preparation_course"),
            "reading score": raw_dict.get("reading score") or raw_dict.get("reading_score"),
            "writing score": raw_dict.get("writing score") or raw_dict.get("writing_score")
        }
        
        # Step 3: Construct single-row pandas DataFrame
        df = pd.DataFrame([normalized_data])
        
        # Step 4: Map the ordinal education category to its target encoded value
        education_levels = {
            'some high school': 0, 'high school': 1, 'some college': 2,
            "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
        }
        
        edu_val = df.loc[0, 'parental level of education']
        
        # Guard clause against default placeholder strings like "string" in Swagger test window
        if edu_val not in education_levels:
            raise ValueError(
                f"Invalid 'parental level of education' value provided: '{edu_val}'. "
                f"Must be one of: {list(education_levels.keys())}"
            )
            
        df['parental level of education'] = df['parental level of education'].map(education_levels)
        
        # Step 5: Execute Scikit-Learn production pipeline inference
        prediction = model.predict(df)
        
        return {
            "status": "success",
            "predicted_math_score": float(round(prediction[0], 4))
        }
        
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        # Catches any array dimension, typing, or data pipeline error and reflects it back cleanly
        raise HTTPException(status_code=500, detail=f"Pipeline Inference Failure: {str(e)}")

@app.get("/health", tags=["System"])
async def health_check():
    """System heartbeat route verifying model health status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }
